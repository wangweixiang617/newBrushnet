import json, threading, time
from config import EVENT_COOLDOWN_SECONDS,MONITOR_INTERVAL
from .bus import BUS
from .database import execute,row,rows,log_action
from .data_service import get_companies,get_rules
from .utils import now_iso


def _hit(value,op,threshold):
    if op=='<':return value<threshold
    if op=='<=':return value<=threshold
    if op=='>':return value>threshold
    if op=='>=':return value>=threshold
    if op=='==':return value==threshold
    return False


class MonitorEngine:
    def __init__(self,interval=MONITOR_INTERVAL):
        self.interval=max(3.0,float(interval));self.stop_event=threading.Event();self.thread=None

    def start(self):
        if self.thread and self.thread.is_alive():return
        self.thread=threading.Thread(target=self._loop,name='ai-monitor',daemon=True);self.thread.start()

    def stop(self):self.stop_event.set()

    def _loop(self):
        time.sleep(6.0)
        while not self.stop_event.is_set():
            try:self.scan()
            except Exception as e:log_action('monitor_error',json.dumps({'error':str(e)},ensure_ascii=False))
            self.stop_event.wait(self.interval)

    def scan(self,force=False):
        created=[];rules=[r for r in get_rules() if int(r['enabled'])]
        for c in get_companies():
            for rule in rules:
                metric=rule['metric'];value=c.get(metric)
                if value is None:continue
                value=float(value);threshold=float(rule['threshold'])
                if not _hit(value,rule['op'],threshold):continue
                key=f"{c['company_id']}:{metric}:{rule['severity']}"
                existing=row("SELECT id,created_at FROM events WHERE event_key=? AND status='OPEN' ORDER BY id DESC LIMIT 1",(key,))
                if existing and not force:continue
                title=f"{rule['domain']}指标异常 · {c['company_name']}"
                message=f"{metric} 当前值 {value:g}，触发规则 {rule['op']} {threshold:g}。"
                eid=execute('''INSERT INTO events(event_key,company_id,company_name,domain,metric,value,threshold,op,severity,title,message,status,created_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',(key,c['company_id'],c['company_name'],rule['domain'],metric,value,threshold,rule['op'],rule['severity'],title,message,'OPEN',now_iso()))
                event={'type':'alert','id':eid,'event_key':key,'company_id':c['company_id'],'company_name':c['company_name'],'domain':rule['domain'],'metric':metric,'value':value,'threshold':threshold,'op':rule['op'],'severity':rule['severity'],'title':title,'message':message,'speak':bool(int(rule['speak'])),'auto_decision':bool(int(rule['auto_decision'])),'created_at':now_iso()}
                created.append(event)
        if created:
            order={'CRITICAL':3,'WARNING':2,'NOTICE':1,'INFO':0}
            primary=sorted(created,key=lambda e:(order.get(e.get('severity'),0),e.get('id',0)),reverse=True)[0]
            primary['batch_count']=len(created)
            BUS.publish(primary)
            if len(created)>1:BUS.publish({'type':'monitor_batch','count':len(created),'primary_id':primary['id'],'created_at':now_iso()})
        return created

    def demo_alert(self,company_id=1001,metric='profit_margin',severity='CRITICAL'):
        c=row('SELECT * FROM companies WHERE company_id=?',(company_id,))
        if not c:raise ValueError('company not found')
        value=float(c.get(metric,0) or 0);threshold=8.0 if metric=='profit_margin' else 65.0
        key=f"demo:{int(time.time())}:{company_id}:{metric}"
        title=f"AI 风险预警 · {c['company_name']}"
        message=f"检测到 {metric} 持续偏离集团基准，系统已进入自动诊断流程。"
        eid=execute('''INSERT INTO events(event_key,company_id,company_name,domain,metric,value,threshold,op,severity,title,message,status,created_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',(key,company_id,c['company_name'],'AI演示',metric,value,threshold,'<',severity,title,message,'OPEN',now_iso()))
        event={'type':'alert','id':eid,'event_key':key,'company_id':company_id,'company_name':c['company_name'],'domain':'AI演示','metric':metric,'value':value,'threshold':threshold,'op':'<','severity':severity,'title':title,'message':message,'speak':True,'auto_decision':True,'created_at':now_iso()}
        BUS.publish(event);return event

    def acknowledge(self,event_id):
        execute("UPDATE events SET status='ACK',acknowledged_at=? WHERE id=?",(now_iso(),event_id))
        BUS.publish({'type':'event_ack','id':event_id});return {'ok':True,'id':event_id}

ENGINE=MonitorEngine()
