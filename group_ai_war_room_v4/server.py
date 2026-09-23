#!/usr/bin/env python3
import argparse, json, mimetypes, queue, sys, time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,urlparse

from config import HOST,PORT,STATIC_DIR,GROUP_NAME,DEFAULT_COMPANY_ID,SSE_HEARTBEAT_SECONDS
from warroom.database import init_db,execute,log_action
from warroom.data_service import *
from warroom.profile_service import profile_payload
from warroom.ai_engine import diagnose,generate_options,full_decision,forecast,interpret_command
from warroom.simulation import simulate_options
from warroom.monitor import ENGINE
from warroom.bus import BUS
from warroom.state import SCENE
from warroom.utils import json_dumps,now_iso,safe_float

ROOT=Path(__file__).resolve().parent


def read_json(handler):
    n=int(handler.headers.get('Content-Length','0') or 0)
    if n<=0:return {}
    raw=handler.rfile.read(min(n,4_000_000))
    return json.loads(raw.decode('utf-8')) if raw else {}


class Handler(BaseHTTPRequestHandler):
    server_version='GroupAIWarRoom/4.0';protocol_version='HTTP/1.1'
    def log_message(self,fmt,*args):sys.stdout.write('%s - - [%s] %s\n'%(self.client_address[0],self.log_date_time_string(),fmt%args))
    def _json(self,obj,status=200):
        data=json_dumps(obj).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def _text(self,text,status=200,ctype='text/plain; charset=utf-8'):
        data=text.encode();self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def _file(self,path):
        path=Path(path)
        if not path.exists() or not path.is_file():return self._text('Not found',404)
        ctype=mimetypes.guess_type(path.name)[0] or 'application/octet-stream';data=path.read_bytes();self.send_response(200);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-cache' if path.suffix in {'.html','.js','.css'} else 'public,max-age=3600');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        p=urlparse(self.path);path=p.path;qs=parse_qs(p.query)
        if path=='/':self.send_response(302);self.send_header('Location','/war-room');self.end_headers();return
        if path=='/war-room':return self._file(STATIC_DIR/'index.html')
        if path=='/control':return self._file(STATIC_DIR/'control.html')
        if path.startswith('/static/'):
            rel=Path(path[8:])
            if '..' in rel.parts:return self._text('Bad path',400)
            return self._file(STATIC_DIR/rel)
        if path=='/api/health':return self._json({'ok':True,'version':'4.0.0','time':now_iso(),'scene':SCENE.get()})
        if path=='/api/bootstrap':
            cid=int(qs.get('company_id',[SCENE.get()['company_id']])[0]);c=get_company(cid) or get_company(DEFAULT_COMPANY_ID)
            return self._json({'group_name':GROUP_NAME,'summary':group_summary(),'group_history':group_history('revenue'),'companies':get_companies(),'company':c,'rankings':{k:get_rankings(k) for k in RANKING_METRICS},'catalog':metric_catalog(),'scene':SCENE.get(),'rules':get_rules(),'events':get_events(20),'investment':investment_tree(c['company_id'])})
        if path=='/api/companies':return self._json(get_companies(qs.get('q',[''])[0]))
        if path.startswith('/api/company/'):
            try:cid=int(path.split('/')[-1])
            except:return self._json({'error':'invalid company'},400)
            c=get_company(cid)
            if not c:return self._json({'error':'not found'},404)
            return self._json(company_snapshot(cid))
        if path=='/api/rankings':return self._json(get_rankings(qs.get('metric',['composite_score'])[0]))
        if path=='/api/metrics/catalog':return self._json(metric_catalog(qs.get('profile',[None])[0]))
        if path.startswith('/api/metric/') and path.endswith('/compare'):
            key=path.split('/')[3];return self._json(metric_compare(key))
        if path.startswith('/api/metric/'):
            key=path.split('/')[3];cid=int(qs.get('company_id',[SCENE.get()['company_id']])[0]);d=metric_detail(cid,key);d['forecast']=forecast(d['series']);return self._json(d)
        if path.startswith('/api/profile/'):
            try:cid=int(path.split('/')[-1])
            except:return self._json({'error':'invalid company'},400)
            profile=qs.get('profile',['operation'])[0];return self._json(profile_payload(cid,profile))
        if path=='/api/investments/tree':return self._json(investment_tree(int(qs.get('company_id',[SCENE.get()['company_id']])[0])))
        if path=='/api/investments':
            cid=qs.get('company_id',[None])[0];return self._json(get_investments(int(cid)) if cid else get_investments())
        if path=='/api/monitor/rules':return self._json(get_rules())
        if path=='/api/events':return self._json(get_events(int(qs.get('limit',[50])[0])))
        if path=='/api/decisions':return self._json(get_decisions(int(qs.get('limit',[50])[0])))
        if path=='/api/scene':return self._json(SCENE.get())
        if path=='/api/events/stream':return self._sse()
        return self._text('Not found',404)
    def do_POST(self):
        path=urlparse(self.path).path
        try:payload=read_json(self)
        except Exception as e:return self._json({'error':f'invalid json: {e}'},400)
        try:
            if path=='/api/scene':
                allowed={'scene','company_id','metric','ambient','action','option','event_id'};patch={k:payload.get(k) for k in allowed if k in payload}
                if 'scene' in payload and 'action' not in payload:patch['action']=None
                return self._json(SCENE.set(patch))
            if path=='/api/monitor/scan':return self._json({'created':ENGINE.scan(force=bool(payload.get('force',False)))})
            if path=='/api/alerts/demo':
                event=ENGINE.demo_alert(int(payload.get('company_id',SCENE.get()['company_id'])),payload.get('metric','roe'),payload.get('severity','CRITICAL'));SCENE.set({'scene':'monitor','company_id':event['company_id'],'metric':event['metric'],'event_id':event['id'],'ambient':False});return self._json(event)
            if path=='/api/events/ack':return self._json(ENGINE.acknowledge(int(payload['event_id'])))
            if path=='/api/monitor/rules':
                rid=execute('''INSERT INTO monitor_rules(domain,metric,op,threshold,severity,enabled,speak,auto_decision) VALUES(?,?,?,?,?,?,?,?)''',(payload.get('domain','自定义'),payload['metric'],payload.get('op','<'),safe_float(payload.get('threshold')),payload.get('severity','WARNING'),int(bool(payload.get('enabled',True))),int(bool(payload.get('speak',False))),int(bool(payload.get('auto_decision',True)))))
                BUS.publish({'type':'rules_changed','id':rid});return self._json({'ok':True,'id':rid},201)
            if path=='/api/ai/diagnose':
                cid=int(payload.get('company_id',SCENE.get()['company_id']));metric=payload.get('metric',SCENE.get()['metric']);result=diagnose(cid,metric,payload.get('event'));SCENE.set({'scene':'decision','company_id':cid,'metric':metric,'ambient':False,'action':None},publish=False);return self._json(result)
            if path=='/api/ai/options':return self._json(generate_options(int(payload.get('company_id',SCENE.get()['company_id'])),payload.get('metric',SCENE.get()['metric'])))
            if path=='/api/ai/simulate':
                cid=int(payload.get('company_id',SCENE.get()['company_id']));metric=payload.get('metric',SCENE.get()['metric']);runs=max(500,min(20000,int(payload.get('runs',5000))));opts=payload.get('options') or generate_options(cid,metric);return self._json(simulate_options(get_company(cid),opts,runs))
            if path=='/api/ai/decision':
                cid=int(payload.get('company_id',SCENE.get()['company_id']));metric=payload.get('metric',SCENE.get()['metric']);runs=max(500,min(20000,int(payload.get('runs',5000))));result=full_decision(cid,metric,runs,payload.get('event'));SCENE.set({'scene':'decision','company_id':cid,'metric':metric,'ambient':False,'action':None},publish=False);return self._json(result)
            if path=='/api/ai/chat':
                intent=interpret_command(payload.get('text',''));patch={'scene':intent['scene'],'metric':intent['metric'],'ambient':False,'action':intent['action'],'option':intent['option']}
                if intent['company_id']:patch['company_id']=intent['company_id']
                SCENE.set(patch);reply=f"已识别：{intent['scene']} / {intent['action'] or 'view'}。"+(f" 当前对象切换为{intent['company_name']}。" if intent['company_name'] else '')
                return self._json({'intent':intent,'reply':reply})
            if path=='/api/decision':
                cid=int(payload.get('company_id',SCENE.get()['company_id']));option=payload.get('option',{});did=execute('''INSERT INTO decisions(company_id,event_id,option_id,option_name,operator,rationale,snapshot_json,created_at) VALUES(?,?,?,?,?,?,?,?)''',(cid,payload.get('event_id'),option.get('id'),option.get('name'),payload.get('operator','集团作战室'),payload.get('rationale','管理层确认'),json_dumps(payload),now_iso()));BUS.publish({'type':'decision_confirmed','id':did,'company_id':cid,'option':option});log_action('decision',json_dumps(payload));return self._json({'ok':True,'decision_id':did})
        except Exception as e:return self._json({'error':str(e),'path':path},500)
        return self._text('Not found',404)
    def _sse(self):
        self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8');self.send_header('Cache-Control','no-cache');self.send_header('Connection','keep-alive');self.send_header('X-Accel-Buffering','no');self.end_headers();q=BUS.subscribe()
        try:
            hello={'type':'hello','time':now_iso(),'scene':SCENE.get()};self.wfile.write(f"data: {json_dumps(hello)}\n\n".encode());self.wfile.flush()
            last=time.time()
            while True:
                try:event=q.get(timeout=1);self.wfile.write(f"data: {json_dumps(event)}\n\n".encode());self.wfile.flush()
                except queue.Empty:
                    if time.time()-last>SSE_HEARTBEAT_SECONDS:self.wfile.write(b': ping\n\n');self.wfile.flush();last=time.time()
        except (BrokenPipeError,ConnectionResetError):pass
        finally:BUS.unsubscribe(q)


def main():
    parser=argparse.ArgumentParser(description='Group AI Digital Twin War Room v4');parser.add_argument('--host',default=HOST);parser.add_argument('--port',type=int,default=PORT);parser.add_argument('--reset-db',action='store_true');args=parser.parse_args();init_db(reset=args.reset_db)
    httpd=ThreadingHTTPServer((args.host,args.port),Handler);print(f'Group AI War Room v4 running: http://127.0.0.1:{args.port}/war-room');print(f'Control console: http://127.0.0.1:{args.port}/control')
    try:httpd.serve_forever()
    except KeyboardInterrupt:print('\nStopping...')
    finally:httpd.server_close()

if __name__=='__main__':main()
