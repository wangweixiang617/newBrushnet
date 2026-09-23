from collections import defaultdict
from .database import row,rows,PROFILE_NAMES
from .utils import mean,median,percentile

PROFILE_METRICS={
    'basic':['total_assets','registered_capital','net_assets','employee_count'],
    'operation':['revenue','net_profit','net_assets','roe'],
    'technology':['rd_investment','rd_ratio','patent_count','rd_staff_ratio'],
    'supervision':['supervision_score','debt_ratio','audit_issues','risk_events'],
}

RANKING_METRICS={'composite_score','basic_score','operation_score','technology_score','supervision_score'}


def get_companies(query=''):
    if query:
        q=f'%{query.strip()}%'
        return rows('''SELECT * FROM companies WHERE company_name LIKE ? OR city LIKE ? OR region LIKE ? ORDER BY composite_score DESC''',(q,q,q))
    return rows('SELECT * FROM companies ORDER BY composite_score DESC')


def get_company(company_id):
    return row('SELECT * FROM companies WHERE company_id=?',(company_id,))


def get_rankings(metric='composite_score'):
    if metric not in RANKING_METRICS:metric='composite_score'
    return rows(f'SELECT company_id,company_name,city,region,{metric} AS value FROM companies ORDER BY {metric} DESC')


def metric_catalog(profile=None):
    if profile:return rows('SELECT * FROM metric_catalog WHERE profile=? ORDER BY sort_order',(profile,))
    return rows('SELECT * FROM metric_catalog ORDER BY sort_order')


def metric_info(metric_key):
    return row('SELECT * FROM metric_catalog WHERE metric_key=?',(metric_key,))


def metric_series(company_id,metric_key):
    return rows('''SELECT period,value FROM metric_values WHERE company_id=? AND metric_key=? ORDER BY period''',(company_id,metric_key))


def latest_metric(company_id,metric_key):
    return row('''SELECT period,value FROM metric_values WHERE company_id=? AND metric_key=? ORDER BY period DESC LIMIT 1''',(company_id,metric_key))


def metric_compare(metric_key):
    info=metric_info(metric_key)
    items=rows('''SELECT c.company_id,c.company_name,c.city,c.region,m.value
                 FROM companies c JOIN metric_values m ON m.company_id=c.company_id
                 WHERE m.metric_key=? AND m.period=(SELECT MAX(period) FROM metric_values WHERE metric_key=?)
                 ORDER BY m.value DESC''',(metric_key,metric_key))
    if info and info['direction']=='lower':items=list(reversed(items))
    vals=[x['value'] for x in items]
    for idx,x in enumerate(items,1):
        x['rank']=idx;x['percentile']=percentile(vals,x['value'])
    return {'metric':info,'items':items,'mean':round(mean(vals),2),'median':round(median(vals),2),'max':max(vals) if vals else 0,'min':min(vals) if vals else 0}


def metric_detail(company_id,metric_key):
    company=get_company(company_id);info=metric_info(metric_key);series=metric_series(company_id,metric_key);compare=metric_compare(metric_key)
    latest=series[-1]['value'] if series else 0
    prev=series[-13]['value'] if len(series)>=13 else (series[0]['value'] if series else latest)
    yoy=0 if not prev else (latest-prev)/abs(prev)*100
    item=next((x for x in compare['items'] if x['company_id']==company_id),None)
    return {'company':company,'metric':info,'series':series,'latest':latest,'yoy':round(yoy,2),'rank':item['rank'] if item else None,'percentile':item['percentile'] if item else None,'compare':compare}


def profile_detail(company_id,profile):
    if profile not in PROFILE_METRICS:profile='operation'
    company=get_company(company_id);metrics=[]
    for key in PROFILE_METRICS[profile]:metrics.append(metric_detail(company_id,key))
    score_key={'basic':'basic_score','operation':'operation_score','technology':'technology_score','supervision':'supervision_score'}[profile]
    return {'company':company,'profile':profile,'profile_name':PROFILE_NAMES[profile],'score':company[score_key],'metrics':metrics}


def company_snapshot(company_id):
    c=get_company(company_id)
    latest={}
    for m in metric_catalog():
        x=latest_metric(company_id,m['metric_key']);latest[m['metric_key']]=x['value'] if x else None
    return {'company':c,'latest':latest}


def group_summary():
    companies=get_companies();latest_period=row('SELECT MAX(period) AS period FROM metric_values')['period']
    revenue=row("SELECT SUM(value) AS v FROM metric_values WHERE metric_key='revenue' AND period=?",(latest_period,))['v'] or 0
    profit=row("SELECT SUM(value) AS v FROM metric_values WHERE metric_key='net_profit' AND period=?",(latest_period,))['v'] or 0
    sentiment=row("SELECT AVG(value) AS v FROM metric_values WHERE metric_key='employee_sentiment' AND period=?",(latest_period,))['v'] or 0
    risk=mean([c['risk_score'] for c in companies])
    return {'company_count':len(companies),'revenue':round(revenue,1),'net_profit':round(profit,1),'sentiment':round(sentiment,1),'risk_index':round(risk,1),'period':latest_period,'avg_score':round(mean([c['composite_score'] for c in companies]),1)}


def group_history(metric_key='revenue'):
    return rows('''SELECT period,SUM(value) AS value FROM metric_values WHERE metric_key=? GROUP BY period ORDER BY period''',(metric_key,))


def get_investments(company_id=None):
    if company_id:return rows('SELECT * FROM investments WHERE company_id=? ORDER BY category,subcategory,id',(company_id,))
    return rows('SELECT * FROM investments ORDER BY company_id,category,subcategory,id')


def investment_tree(company_id):
    items=get_investments(company_id);company=get_company(company_id)
    cats=defaultdict(lambda:defaultdict(list))
    for x in items:cats[x['category']][x['subcategory']].append(x)
    children=[]
    for cat,subs in cats.items():
        subnodes=[]
        for sub,projects in subs.items():
            subnodes.append({'name':sub,'value':round(sum(p['amount'] for p in projects),2),'children':[{'name':p['project_name'],'value':p['amount'],'roi':p['roi'],'npv':p['npv'],'risk':p['risk'],'progress':p['progress'],'id':p['id']} for p in projects]})
        children.append({'name':cat,'value':round(sum(n['value'] for n in subnodes),2),'children':subnodes})
    total=round(sum(x['amount'] for x in items),2);avg_roi=round(mean([x['roi'] for x in items]),1);npv=round(sum(x['npv'] for x in items),2)
    return {'name':company['company_name'],'value':total,'children':children,'kpi':{'investment':total,'npv':npv,'avg_roi':avg_roi,'project_count':len(items)}}


def get_rules():return rows('SELECT * FROM monitor_rules ORDER BY id')
def get_events(limit=50):return rows('''SELECT e.*,c.company_name FROM events e LEFT JOIN companies c ON c.company_id=e.company_id ORDER BY e.id DESC LIMIT ?''',(limit,))
def get_decisions(limit=50):return rows('''SELECT d.*,c.company_name FROM decisions d LEFT JOIN companies c ON c.company_id=d.company_id ORDER BY d.id DESC LIMIT ?''',(limit,))
