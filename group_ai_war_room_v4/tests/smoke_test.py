import json,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def fetch(url,payload=None):
    if payload is None:req=urllib.request.Request(url)
    else:req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=10) as r:return json.loads(r.read())

def main():
    subprocess.check_call([sys.executable,str(ROOT/'scripts/init_db.py'),'--reset'],cwd=ROOT)
    p=subprocess.Popen([sys.executable,str(ROOT/'server.py'),'--host','127.0.0.1','--port','8769'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:fetch('http://127.0.0.1:8769/api/health');break
            except Exception:time.sleep(.15)
        b=fetch('http://127.0.0.1:8769/api/bootstrap');assert len(b['companies'])>=18 and len(b['catalog'])>=18
        q=fetch('http://127.0.0.1:8769/api/rankings?metric=operation_score');assert q[0]['value']>=q[-1]['value']
        p1=fetch('http://127.0.0.1:8769/api/profile/1?profile=operation');assert len(p1['metrics'])==4
        m=fetch('http://127.0.0.1:8769/api/metric/roe?company_id=1');assert len(m['series'])==60 and len(m['forecast']['points'])==12
        inv=fetch('http://127.0.0.1:8769/api/investments/tree?company_id=1');assert inv['kpi']['project_count']>=12
        dec=fetch('http://127.0.0.1:8769/api/ai/decision',{'company_id':1,'metric':'roe','runs':600});assert len(dec['options'])==3
        print('SMOKE TEST PASS')
    finally:
        p.terminate();p.wait(timeout=5)
if __name__=='__main__':main()
