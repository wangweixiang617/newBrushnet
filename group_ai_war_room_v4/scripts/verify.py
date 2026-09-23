#!/usr/bin/env python3
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from warroom.database import init_db,rows
from warroom.data_service import get_companies,metric_catalog,metric_detail,profile_detail,investment_tree
from warroom.ai_engine import full_decision

def main():
    init_db(reset=True)
    companies=get_companies();assert len(companies)>=18
    assert len(metric_catalog())>=18
    d=metric_detail(1,'revenue');assert len(d['series'])==60 and d['rank']
    p=profile_detail(1,'operation');assert len(p['metrics'])==4
    inv=investment_tree(1);assert inv['children'] and inv['kpi']['project_count']>=12
    dec=full_decision(1,'roe',700);assert len(dec['options'])==3 and dec['diagnosis']['steps']
    print(json.dumps({'ok':True,'companies':len(companies),'metrics':len(metric_catalog()),'months':len(d['series']),'options':len(dec['options'])},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
