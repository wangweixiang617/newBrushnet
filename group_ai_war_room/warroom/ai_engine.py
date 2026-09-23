import json, re, urllib.request
from config import AI_MODE,AI_ENDPOINT,AI_API_KEY
from .data_service import get_company,get_history,get_companies
from .simulation import simulate_options
from .utils import clamp


def _corr(xs,ys):
    if len(xs)<3 or len(xs)!=len(ys):return 0.0
    mx=sum(xs)/len(xs);my=sum(ys)/len(ys)
    a=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    b=sum((x-mx)**2 for x in xs);c=sum((y-my)**2 for y in ys)
    return a/((b*c)**.5) if b and c else 0.0


def _remote(path,payload):
    url=AI_ENDPOINT.rstrip('/')+'/'+path.lstrip('/')
    body=json.dumps(payload,ensure_ascii=False).encode('utf-8')
    headers={'Content-Type':'application/json'}
    if AI_API_KEY:headers['Authorization']='Bearer '+AI_API_KEY
    req=urllib.request.Request(url,data=body,headers=headers,method='POST')
    with urllib.request.urlopen(req,timeout=20) as r:return json.loads(r.read().decode('utf-8'))


def metric_label(metric):
    return {
      'profit_margin':'营业利润率','cash_flow':'现金流','employee_sentiment':'员工情绪指数','employee_turnover':'员工流失率',
      'risk_score':'风险指数','project_delay_days':'项目延期','receivables_growth':'应收账款增速','operation_score':'经营能力',
      'finance_score':'财务能力','investment_score':'投资能力','compliance_score':'合规能力'
    }.get(metric,metric)


def diagnose(company_id,metric='profit_margin',event=None):
    if AI_MODE=='remote' and AI_ENDPOINT:return _remote('/diagnose',{'company_id':company_id,'metric':metric,'event':event})
    c=get_company(company_id);h=get_history(company_id)
    if not c:raise ValueError('company not found')
    series=[float(x.get(metric,c.get(metric,0)) or 0) for x in h] if h and metric in h[0] else []
    steps=[
      {'id':'load','label':'读取当前企业与历史数据','detail':f"载入 {len(h)} 个月历史记录"},
      {'id':'compare','label':'比较集团下属企业','detail':f"横向比较 {len(get_companies())} 家分公司"},
      {'id':'structure','label':'分析关联指标结构','detail':'计算趋势、偏离与关键相关项'},
      {'id':'similar','label':'检索历史相似事件','detail':'匹配同类风险事件与处置路径'},
      {'id':'causal','label':'构建原因关系图','detail':'形成可视化原因链和置信度'},
      {'id':'options','label':'生成决策候选方案','detail':'产生多方案并交由仿真引擎评价'}]
    causes=[]
    if metric=='profit_margin':
        pm=[float(x['profit_margin']) for x in h];cost=[float(x['operating_cost']) for x in h];rev=[float(x['revenue']) for x in h];cash=[float(x['cash_flow']) for x in h]
        causes=[
          {'id':'cost','label':'运营成本上升','confidence':round(clamp(abs(_corr(pm,cost))*100+24,42,94),1),'value':f"{cost[-1]:.2f} 亿元/月"},
          {'id':'revenue','label':'收入增长承压','confidence':round(clamp(abs(_corr(pm,rev))*100+18,38,89),1),'value':f"{rev[-1]:.2f} 亿元/月"},
          {'id':'cash','label':'现金流同步走弱','confidence':round(clamp(abs(_corr(pm,cash))*100+16,35,86),1),'value':f"{cash[-1]:.2f} 亿元"},
          {'id':'receivable','label':'应收账款增长偏快','confidence':round(clamp(float(c['receivables_growth'])+49,40,88),1),'value':f"{c['receivables_growth']:.1f}%"}]
        root='利润率下降';summary='利润率异常主要由成本端压力、收入增速承压与现金回收效率共同驱动。'
    elif metric in {'employee_sentiment','employee_turnover'}:
        causes=[
          {'id':'turnover','label':'员工流失率偏高','confidence':82.0,'value':f"{c['employee_turnover']:.1f}%"},
          {'id':'delay','label':'项目延期带来工作负荷','confidence':74.0,'value':f"{c['project_delay_days']:.1f} 天"},
          {'id':'growth','label':'业务变化增加组织压力','confidence':63.0,'value':f"{c['growth']:.1f}%"},
          {'id':'management','label':'管理与沟通需要复核','confidence':58.0,'value':'待访谈/问卷验证'}]
        root='员工状态异常';summary='员工状态变化需要结合流失率、项目负荷和匿名调查趋势综合判断。'
    else:
        causes=[
          {'id':'history','label':'历史趋势偏离','confidence':84.0,'value':'显著'},
          {'id':'peer','label':'低于集团同类基准','confidence':78.0,'value':'显著'},
          {'id':'risk','label':'关联风险指标上升','confidence':69.0,'value':f"{c['risk_score']:.1f}"}]
        root=metric_label(metric)+'异常';summary='系统检测到指标偏离历史与集团基准，建议进入多方案仿真。'
    return {'company':c,'metric':metric,'metric_label':metric_label(metric),'steps':steps,'root':root,'causes':causes,'summary':summary,'trend':series[-12:]}


def generate_options(company_id,metric='profit_margin'):
    c=get_company(company_id)
    if AI_MODE=='remote' and AI_ENDPOINT:return _remote('/options',{'company':c,'metric':metric})
    if metric=='profit_margin':
        return [
          {'id':'A','name':'成本优化','subtitle':'COST CONTROL','budget_m':4.2,'expected_profit_lift':2.1,'duration_days':30,'risk':'LOW','actions':['集中采购与重点供应商谈判','冻结低收益非核心支出','建立成本异常周度复盘']},
          {'id':'B','name':'收入增长','subtitle':'REVENUE GROWTH','budget_m':12.0,'expected_profit_lift':3.4,'duration_days':90,'risk':'HIGH','actions':['聚焦重点客户和高毛利产品','调整华东区域销售资源','优化价格与客户留存策略']},
          {'id':'C','name':'混合策略','subtitle':'HYBRID STRATEGY','budget_m':9.0,'expected_profit_lift':4.0,'duration_days':60,'risk':'MEDIUM','actions':['采购降本与预算重排','重点客户增长计划','按月复盘利润率与现金流联动']}
        ]
    if metric in {'employee_sentiment','employee_turnover'}:
        return [
          {'id':'A','name':'组织负荷优化','subtitle':'WORKLOAD RESET','budget_m':1.6,'expected_profit_lift':1.0,'duration_days':45,'risk':'LOW','actions':['识别高负荷团队','优化项目排期与加班','建立匿名脉冲问卷']},
          {'id':'B','name':'关键人才保留','subtitle':'RETENTION','budget_m':3.8,'expected_profit_lift':1.5,'duration_days':90,'risk':'MEDIUM','actions':['关键岗位保留计划','管理者沟通训练','内部流动与成长路径']},
          {'id':'C','name':'组织健康组合方案','subtitle':'ORG HEALTH','budget_m':4.5,'expected_profit_lift':2.0,'duration_days':75,'risk':'MEDIUM','actions':['负荷治理','人才保留','匿名员工体验闭环']}
        ]
    return [
      {'id':'A','name':'稳健修复','subtitle':'STABILIZE','budget_m':3.0,'expected_profit_lift':1.4,'duration_days':45,'risk':'LOW','actions':['修复关键偏差','强化周度跟踪']},
      {'id':'B','name':'专项提升','subtitle':'ACCELERATE','budget_m':7.0,'expected_profit_lift':2.5,'duration_days':75,'risk':'MEDIUM','actions':['集中资源专项突破','设置阶段性里程碑']},
      {'id':'C','name':'结构调整','subtitle':'TRANSFORM','budget_m':10.0,'expected_profit_lift':3.2,'duration_days':120,'risk':'HIGH','actions':['结构性调整','跨部门协同治理']}
    ]


def full_decision(company_id,metric='profit_margin',runs=5000,event=None):
    diagnosis=diagnose(company_id,metric,event)
    options=generate_options(company_id,metric)
    simulated=simulate_options(diagnosis['company'],options,runs=runs)
    return {'diagnosis':diagnosis,'options':simulated}


def interpret_command(text):
    text=(text or '').strip()
    companies=get_companies();company=None
    for c in companies:
        aliases=[c['company_name'],c['city'],c['province'],c['region']]
        if any(a and a in text for a in aliases):company=c;break
    metric='profit_margin'
    metric_terms={
      '利润':'profit_margin','现金流':'cash_flow','员工情绪':'employee_sentiment','员工状态':'employee_sentiment','流失':'employee_turnover',
      '投资':'investment_score','风险':'risk_score','项目':'project_delay_days','经营':'operation_score','财务':'finance_score','合规':'compliance_score'}
    for k,v in metric_terms.items():
        if k in text:metric=v;break
    scene='overview';action='navigate'
    if any(k in text for k in ['分析','诊断','为什么','原因']):scene='decision';action='diagnose'
    elif any(k in text for k in ['模拟','仿真']):scene='decision';action='simulate'
    elif any(k in text for k in ['数字孪生','孪生']):scene='twin'
    elif any(k in text for k in ['画像','四象限']):scene='profile'
    elif any(k in text for k in ['监控','检测','预警']):scene='monitor'
    elif '投资' in text:scene='investment'
    elif any(k in text for k in ['排行','排名','指标']):scene='analytics'
    option=None
    if re.search(r'(方案|option)\s*(一|1|a)',text,re.I):option='A'
    elif re.search(r'(方案|option)\s*(二|2|b)',text,re.I):option='B'
    elif re.search(r'(方案|option)\s*(三|3|c)',text,re.I):option='C'
    return {'text':text,'company_id':company['company_id'] if company else None,'company_name':company['company_name'] if company else None,'metric':metric,'scene':scene,'action':action,'option':option}
