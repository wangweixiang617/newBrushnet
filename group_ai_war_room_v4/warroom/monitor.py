import operator
from .database import execute
from .data_service import get_companies,get_rules,latest_metric,metric_info
from .bus import BUS
from .utils import now_iso

OPS={'<':operator.lt,'>':operator.gt,'<=':operator.le,'>=':operator.ge}

class MonitorEngine:
    def scan(self,force=False):
        created=[]
        for rule in get_rules():
            if not rule['enabled']:continue
            info=metric_info(rule['metric'])
            if not info:continue
            for c in get_companies():
                x=latest_metric(c['company_id'],rule['metric'])
                if not x:continue
                hit=OPS.get(rule['op'],operator.lt)(float(x['value']),float(rule['threshold']))
                if hit or (force and c['company_id']==1 and rule['id']==1):
                    title=f"{info['metric_name']}触发{rule['severity']}预警"
                    message=f"当前 {x['value']:.2f}{info['unit']}，规则 {rule['op']} {rule['threshold']}。AI已建立诊断任务。"
                    eid=execute('''INSERT INTO events(company_id,metric,severity,title,message,value,threshold,op,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)''',(c['company_id'],rule['metric'],rule['severity'],title,message,x['value'],rule['threshold'],rule['op'],'OPEN',now_iso()))
                    event={'id':eid,'type':'alert','company_id':c['company_id'],'company_name':c['company_name'],'metric':rule['metric'],'severity':rule['severity'],'title':title,'message':message,'value':x['value'],'threshold':rule['threshold'],'op':rule['op'],'speak':bool(rule['speak']),'auto_decision':bool(rule['auto_decision'])}
                    BUS.publish(event);created.append(event)
                    if not force:return created
        return created
    def demo_alert(self,company_id,metric='roe',severity='CRITICAL'):
        c=next(x for x in get_companies() if x['company_id']==company_id);info=metric_info(metric);x=latest_metric(company_id,metric);threshold=8 if metric=='roe' else x['value']*.9
        title=f"{info['metric_name']}出现高等级异常";message=f"{info['metric_name']}连续偏离集团基准，当前 {x['value']:.2f}{info['unit']}。AI正在进入诊断模式。"
        eid=execute('''INSERT INTO events(company_id,metric,severity,title,message,value,threshold,op,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)''',(company_id,metric,severity,title,message,x['value'],threshold,'<','OPEN',now_iso()))
        event={'id':eid,'type':'alert','company_id':company_id,'company_name':c['company_name'],'metric':metric,'severity':severity,'title':title,'message':message,'value':x['value'],'threshold':threshold,'op':'<','speak':True,'auto_decision':True};BUS.publish(event);return event
    def acknowledge(self,event_id):
        execute("UPDATE events SET status='ACK' WHERE id=?",(event_id,));return {'ok':True,'event_id':event_id}
ENGINE=MonitorEngine()
