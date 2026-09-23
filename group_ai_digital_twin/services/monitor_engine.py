import operator
import threading
from datetime import datetime, timezone
from services.analytics import METRIC_LABELS

OPS={'>':operator.gt,'>=':operator.ge,'<':operator.lt,'<=':operator.le,'==':operator.eq}

class EventStore:
    def __init__(self,max_items=300):
        self.max_items=max_items
        self._events=[]
        self._keys={}
        self._lock=threading.Lock()
        self._seq=0

    def add(self,event,dedupe_key=None):
        with self._lock:
            if dedupe_key and dedupe_key in self._keys:
                old_id=self._keys[dedupe_key]
                for e in self._events:
                    if e['id']==old_id:
                        e.update(event);e['updated_at']=datetime.now(timezone.utc).isoformat();return e
            self._seq+=1
            event={'id':self._seq,'created_at':datetime.now(timezone.utc).isoformat(),**event}
            self._events.append(event)
            self._events=self._events[-self.max_items:]
            if dedupe_key:self._keys[dedupe_key]=event['id']
            return event

    def latest(self):
        with self._lock:return dict(self._events[-1]) if self._events else None

    def list(self,limit=50):
        with self._lock:return [dict(x) for x in self._events[-limit:]][::-1]

class MonitorEngine:
    def __init__(self,data_service,ai_gateway,event_store):
        self.ds=data_service;self.ai=ai_gateway;self.events=event_store

    def run_all(self):
        self.ds.reload()
        created=[]
        rules=self.ds.rules[self.ds.rules.enabled.astype(int)==1]
        for _,company in self.ds.companies.iterrows():
            for _,rule in rules.iterrows():
                metric=str(rule.metric);op=str(rule.op)
                try:value=self.ds.metric_value(int(company.company_id),metric)
                except KeyError:continue
                threshold=float(rule.threshold)
                if op in OPS and OPS[op](value,threshold):
                    reason=f"{METRIC_LABELS.get(metric,metric)}当前值 {value:.2f}，触发规则 {op} {threshold:.2f}"
                    ai=self.ai.monitor({'company_id':int(company.company_id),'metric':metric,'current_value':value,
                                        'threshold':threshold,'severity':str(rule.severity),'reason':reason})
                    event=self.events.add({
                        'company_id':int(company.company_id),'company_name':str(company.company_name),'domain':str(rule.domain),
                        'metric':metric,'value':value,'threshold':threshold,'op':op,'severity':str(rule.severity),
                        'speak':bool(int(rule.speak)),'auto_decision':bool(int(rule.auto_decision)),
                        'title':ai['title'],'reason':ai['reason'],'status':'OPEN'
                    },dedupe_key=f"{int(company.company_id)}:{metric}:{op}:{threshold}")
                    created.append(event)
        return created

    def manual_event(self,company_id,metric='profit_margin',severity='CRITICAL'):
        company=self.ds.company(company_id);value=self.ds.metric_value(company_id,metric)
        return self.events.add({'company_id':int(company_id),'company_name':str(company.company_name),'domain':'AI演示',
            'metric':metric,'value':value,'threshold':value+1,'op':'<','severity':severity,'speak':True,'auto_decision':True,
            'title':f"{METRIC_LABELS.get(metric,metric)}智能预警",'reason':'演示事件：AI检测到连续趋势异常，需要进入决策分析。','status':'OPEN'})
