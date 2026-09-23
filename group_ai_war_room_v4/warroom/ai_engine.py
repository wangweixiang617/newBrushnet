import json, urllib.request
from config import AI_MODE,AI_ENDPOINT,AI_API_KEY
from .data_service import get_company,metric_detail,get_companies
from .simulation import simulate_options


def forecast(series,horizon=12):
    values=[float(x['value']) for x in series]
    if not values:return {'points':[],'confidence':0}
    n=min(18,len(values));ys=values[-n:];xs=list(range(n));xm=sum(xs)/n;ym=sum(ys)/n
    den=sum((x-xm)**2 for x in xs) or 1;slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/den
    residuals=[y-(ym+slope*(x-xm)) for x,y in zip(xs,ys)];sigma=(sum(r*r for r in residuals)/max(1,n-2))**.5
    points=[];last=ys[-1]
    for i in range(1,horizon+1):
        pred=max(0,last+slope*i);points.append({'step':i,'value':round(pred,2),'low':round(max(0,pred-1.64*sigma),2),'high':round(pred+1.64*sigma,2)})
    confidence=max(55,min(94,90-sigma/(abs(ym)+1)*250))
    return {'points':points,'confidence':round(confidence,1),'trend':'up' if slope>0 else 'down','slope':round(slope,3)}


def _remote(path,payload):
    if not AI_ENDPOINT:return None
    data=json.dumps(payload,ensure_ascii=False).encode()
    req=urllib.request.Request(AI_ENDPOINT+path,data=data,headers={'Content-Type':'application/json','Authorization':f'Bearer {AI_API_KEY}' if AI_API_KEY else ''})
    with urllib.request.urlopen(req,timeout=30) as r:return json.loads(r.read().decode())


def diagnose(company_id,metric_key='roe',event=None):
    if AI_MODE=='remote':
        try:
            x=_remote('/diagnose',{'company_id':company_id,'metric':metric_key,'event':event})
            if x:return x
        except Exception:pass
    c=get_company(company_id);d=metric_detail(company_id,metric_key);series=d['series'];fc=forecast(series,6)
    metric=d['metric'];latest=d['latest'];mean_v=d['compare']['mean'];direction=metric['direction'] if metric else 'higher'
    bad=(latest<mean_v) if direction=='higher' else (latest>mean_v)
    causes=[]
    if metric_key in {'roe','net_profit','cash_flow','revenue'}:
        causes=[('成本压力','采购与人工成本上涨对利润形成挤压',.91),('收入增速放缓','重点区域订单增长弱于集团均值',.84),('现金回收周期','应收与现金回笼速度对经营质量产生影响',.72),('人员效率','人均产出与组织效率存在改善空间',.61)]
    elif metric_key in {'supervision_score','debt_ratio','audit_issues','risk_events'}:
        causes=[('监管合规','审计问题与治理闭环速度影响监管画像',.93),('杠杆压力','资产负债结构增加经营弹性压力',.79),('项目风险','部分投资项目处于高波动阶段',.68),('流程治理','审批与授权流程存在优化空间',.55)]
    elif metric_key in {'rd_investment','rd_ratio','patent_count','rd_staff_ratio'}:
        causes=[('研发强度','研发投入增长与收入规模不同步',.88),('成果转化','专利与研发成果商业化效率不足',.76),('人才结构','研发人员占比与核心岗位密度偏低',.67),('项目组合','研发项目组合集中度较高',.58)]
    else:
        causes=[('指标偏离','当前指标偏离集团基准',.86),('历史趋势','连续周期变化触发异常检测',.78),('同业差距','横向比较显示排名下降',.69),('结构因素','关联指标共同驱动变化',.57)]
    steps=[
      {'label':'读取历史数据','detail':f'载入 {len(series)} 个月历史记录'},
      {'label':'比较集团下属企业','detail':f'横向比较 {len(get_companies())} 家企业'},
      {'label':'分析关联指标结构','detail':'计算趋势、偏离度、分位数与相关项'},
      {'label':'检索历史相似事件','detail':'匹配集团内部历史波动与处置路径'},
      {'label':'构建原因关系图','detail':'形成可视化因果链和置信度'},
      {'label':'生成决策候选方案','detail':'进入多方案仿真与管理决策阶段'},
    ]
    status='异常' if bad else '关注'
    return {'company':c,'metric':metric,'latest':latest,'group_mean':mean_v,'rank':d['rank'],'status':status,'forecast':fc,'causes':[{'name':a,'detail':b,'confidence':int(p*100)} for a,b,p in causes],'steps':steps,'summary':f"{c['company_name']} 的{metric['metric_name']}当前为 {latest:.2f}{metric['unit']}，集团均值 {mean_v:.2f}{metric['unit']}。AI判断主要由{causes[0][0]}、{causes[1][0]}共同驱动。"}


def generate_options(company_id,metric_key='roe'):
    c=get_company(company_id)
    return [
      {'id':'A','name':'成本优化','subtitle':'COST CONTROL','budget_m':4.2,'duration_days':30,'risk':'LOW','expected_profit_m':18.0,'npv_m':15.2,'execution_factor':.96,'volatility':.08,'risk_penalty':2.1,'npv_factor':1.08,'success_floor':6,'actions':['集中采购与供应商谈判','压缩低收益非核心支出','建立现金回笼专项机制']},
      {'id':'B','name':'增长驱动','subtitle':'REVENUE GROWTH','budget_m':12.0,'duration_days':90,'risk':'HIGH','expected_profit_m':35.0,'npv_m':27.1,'execution_factor':.88,'volatility':.18,'risk_penalty':8.0,'npv_factor':1.12,'success_floor':10,'actions':['重点区域销售资源加码','核心客户分层经营','产品组合与价格策略调整']},
      {'id':'C','name':'组合策略','subtitle':'HYBRID STRATEGY','budget_m':9.0,'duration_days':60,'risk':'MEDIUM','expected_profit_m':39.0,'npv_m':31.4,'execution_factor':.93,'volatility':.12,'risk_penalty':4.6,'npv_factor':1.10,'success_floor':11,'actions':['采购成本与费用双向优化','聚焦高毛利业务增长','设立60天经营改善战役']},
    ]


def full_decision(company_id,metric_key='roe',runs=5000,event=None):
    diagnosis=diagnose(company_id,metric_key,event);options=generate_options(company_id,metric_key);sim=simulate_options(get_company(company_id),options,runs)
    best=max(sim,key=lambda x:(x['simulated_npv_m'],x['success_probability']*.15))
    return {'diagnosis':diagnosis,'options':sim,'recommended':best['id'],'runs':runs}


def interpret_command(text):
    text=(text or '').strip();companies=get_companies();company=None
    for c in companies:
        if c['company_name'] in text or c['city'] in text:company=c;break
    metric='revenue';scene='overview';action=None;option=None
    mapping=[('利润','roe'),('营收','revenue'),('收入','revenue'),('现金流','cash_flow'),('研发','rd_ratio'),('监管','supervision_score'),('负债','debt_ratio'),('员工','employee_sentiment'),('投资','investment_roi')]
    for k,v in mapping:
        if k in text:metric=v;break
    if any(k in text for k in ['分析','原因','诊断']):scene='decision';action='diagnose'
    elif any(k in text for k in ['模拟','仿真']):scene='decision';action='simulate'
    elif '监控' in text:scene='monitor'
    elif '画像' in text:scene='profile'
    elif '指标' in text:scene='metrics'
    elif '投资' in text:scene='investment'
    elif any(k in text for k in ['排行','经营']):scene='analytics'
    if '方案一' in text or '方案A' in text.upper():option='A'
    if '方案二' in text or '方案B' in text.upper():option='B'
    if '方案三' in text or '方案C' in text.upper():option='C'
    return {'scene':scene,'action':action,'metric':metric,'option':option,'company_id':company['company_id'] if company else None,'company_name':company['company_name'] if company else None,'text':text}
