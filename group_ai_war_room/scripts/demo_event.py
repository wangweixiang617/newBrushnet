#!/usr/bin/env python3
import json, sys, urllib.request
url=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8050/api/alerts/demo'
payload=json.dumps({'company_id':1001,'metric':'profit_margin','severity':'CRITICAL'}).encode()
req=urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'},method='POST')
print(urllib.request.urlopen(req,timeout=10).read().decode())
