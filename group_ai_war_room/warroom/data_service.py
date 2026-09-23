from .database import rows,row

SCORE_KEYS=['finance_score','operation_score','hr_score','innovation_score','investment_score','market_score','compliance_score','esg_score']
METRICS={
 'revenue':('营业收入','亿元'),'profit_margin':('营业利润率','%'),'cash_flow':('现金流','亿元'),
 'growth':('增长率','%'),'debt_ratio':('资产负债率','%'),'receivables_growth':('应收账款增速','%'),
 'employee_turnover':('员工流失率','%'),'employee_sentiment':('员工情绪指数',''),
 'project_delay_days':('项目延期','天'),'operation_score':('经营能力','分'),'finance_score':('财务能力','分'),
 'hr_score':('人力能力','分'),'innovation_score':('创新能力','分'),'investment_score':('投资能力','分'),
 'market_score':('市场能力','分'),'compliance_score':('合规能力','分'),'esg_score':('ESG','分'),'risk_score':('风险指数','分')}


def composite_score(c):
    # 风险越低越好；其他能力分取均值。
    base=sum(float(c[k]) for k in SCORE_KEYS)/len(SCORE_KEYS)
    risk_adj=(100-float(c['risk_score']))*0.16
    growth_adj=max(-6,min(6,float(c['growth'])*0.25))
    return round(base*0.84+risk_adj+growth_adj,1)


def get_companies():
    out=rows('SELECT * FROM companies ORDER BY company_id')
    for c in out:
        c['score']=composite_score(c)
        c['status']='CRITICAL' if c['profit_margin']<7 or c['employee_sentiment']<58 or c['project_delay_days']>12 else 'WARNING' if c['profit_margin']<9 or c['employee_sentiment']<68 or c['project_delay_days']>8 else 'NORMAL'
    return out


def get_company(company_id):
    c=row('SELECT * FROM companies WHERE company_id=?',(company_id,))
    if not c:return None
    c['score']=composite_score(c)
    c['status']='CRITICAL' if c['profit_margin']<7 or c['employee_sentiment']<58 or c['project_delay_days']>12 else 'WARNING' if c['profit_margin']<9 or c['employee_sentiment']<68 or c['project_delay_days']>8 else 'NORMAL'
    return c


def get_history(company_id):return rows('SELECT * FROM history WHERE company_id=? ORDER BY month',(company_id,))
def get_investments(company_id=None):return rows('SELECT * FROM investments'+(' WHERE company_id=?' if company_id else '')+' ORDER BY npv DESC',((company_id,) if company_id else ()))
def get_rules():return rows('SELECT * FROM monitor_rules ORDER BY id')
def get_events(limit=50):return rows('SELECT * FROM events ORDER BY id DESC LIMIT ?',(limit,))
def get_decisions(limit=50):return rows('SELECT * FROM decisions ORDER BY id DESC LIMIT ?',(limit,))


def get_group_history():
    return rows('''SELECT month, ROUND(SUM(revenue),3) revenue, ROUND(AVG(profit_margin),3) profit_margin, ROUND(SUM(cash_flow),3) cash_flow, ROUND(AVG(employee_sentiment),3) employee_sentiment, ROUND(AVG(employee_turnover),3) employee_turnover, ROUND(AVG(project_progress),3) project_progress, ROUND(SUM(operating_cost),3) operating_cost FROM history GROUP BY month ORDER BY month''')


def get_rankings(metric='score'):
    cs=get_companies()
    if metric=='score':key=lambda x:x['score']
    else:key=lambda x:float(x.get(metric,0) or 0)
    reverse=metric not in {'risk_score','debt_ratio','employee_turnover','project_delay_days'}
    return sorted(cs,key=key,reverse=reverse)


def group_summary():
    cs=get_companies()
    total_revenue=sum(float(c['revenue']) for c in cs)
    avg_profit=sum(float(c['profit_margin']) for c in cs)/len(cs)
    avg_score=sum(c['score'] for c in cs)/len(cs)
    risk_count=sum(c['status']!='NORMAL' for c in cs)
    inv=get_investments()
    return {
      'revenue':round(total_revenue,1),'profit_margin':round(avg_profit,1),'score':round(avg_score,1),
      'risk_count':risk_count,'subsidiaries':len(cs),'investment':round(sum(float(i['amount']) for i in inv),1),
      'npv':round(sum(float(i['npv']) for i in inv),1)
    }
