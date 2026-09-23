import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from warroom.database import init_db
from warroom.data_service import get_companies,get_history,group_summary
from warroom.ai_engine import diagnose,full_decision,interpret_command

init_db()
cs=get_companies();assert len(cs)>=8
h=get_history(1001);assert len(h)>=10
s=group_summary();assert s['revenue']>0
r=diagnose(1001,'profit_margin');assert r['causes'] and len(r['steps'])>=5
d=full_decision(1001,'profit_margin',runs=700);assert len(d['options'])==3 and all('npv_m' in x for x in d['options'])
i=interpret_command('分析上海分公司为什么利润下降');assert i['scene']=='decision' and i['company_id']==1001
print('SMOKE TEST OK')
