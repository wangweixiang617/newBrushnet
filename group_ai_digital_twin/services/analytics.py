import numpy as np
import pandas as pd

SCORE_COLS=['operation_score','finance_score','hr_score','innovation_score','investment_score','market_score','compliance_score','esg_score']

METRIC_LABELS={
    'revenue':'营业收入','profit_margin':'利润率','cash_flow':'现金流','growth':'增长率','debt_ratio':'资产负债率',
    'receivables_growth':'应收账款增速','employee_turnover':'员工流失率','employee_sentiment':'员工状态指数',
    'project_delay_days':'项目延期天数','operation_score':'经营能力','finance_score':'财务能力','hr_score':'人力能力',
    'innovation_score':'创新能力','investment_score':'投资能力','market_score':'市场能力','compliance_score':'合规能力',
    'esg_score':'ESG能力','risk_score':'风险指数','project_progress':'项目进度','operating_cost':'经营成本'
}

DOMAIN_LABELS={
    'operation_score':'经营','finance_score':'财务','hr_score':'人力','innovation_score':'创新',
    'investment_score':'投资','market_score':'市场','compliance_score':'合规','esg_score':'ESG'
}

def composite_score(row):
    vals=[float(row[c]) for c in SCORE_COLS if c in row.index]
    risk_penalty=max(0,(float(row.get('risk_score',50))-50)*0.10)
    return round(float(np.mean(vals))-risk_penalty,1)

def add_scores(df):
    out=df.copy()
    out['composite_score']=out.apply(composite_score,axis=1)
    out['growth_index']=np.clip(50+out['growth']*2.0,0,100)
    out['value_index']=np.clip(0.45*out['finance_score']+0.30*out['investment_score']+0.25*out['market_score'],0,100)
    return out

def rank_table(df, metric='composite_score', ascending=False):
    data=add_scores(df) if metric=='composite_score' and metric not in df.columns else df.copy()
    data=data.sort_values(metric,ascending=ascending).reset_index(drop=True)
    data['rank']=np.arange(1,len(data)+1)
    return data

def status_from_score(score):
    if score>=82:return 'EXCELLENT'
    if score>=72:return 'NORMAL'
    if score>=62:return 'NOTICE'
    return 'WARNING'

def linear_forecast(history, metric='profit_margin', steps=6):
    s=history[['month',metric]].dropna().copy()
    if len(s)<2:return []
    x=np.arange(len(s),dtype=float)
    coef=np.polyfit(x,s[metric].astype(float),1)
    future_x=np.arange(len(s),len(s)+steps,dtype=float)
    y=np.polyval(coef,future_x)
    start=pd.Timestamp(s.month.max())+pd.offsets.MonthBegin(1)
    dates=pd.date_range(start,periods=steps,freq='MS')
    return [{'month':d,'value':float(v)} for d,v in zip(dates,y)]

def anomaly_zscore(history, metric, window=12):
    s=history[metric].dropna().astype(float).tail(window)
    if len(s)<4:return 0.0
    std=float(s.std(ddof=0))
    return 0.0 if std==0 else float((s.iloc[-1]-s.mean())/std)
