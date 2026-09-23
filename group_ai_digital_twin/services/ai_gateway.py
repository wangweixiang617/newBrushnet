import requests
from config import AI_MODE, AI_ENDPOINT
from services.analytics import METRIC_LABELS

class AIGateway:
    """AI统一接口。默认mock模式；remote模式可替换为集团内部LLM/智能体服务。"""
    def __init__(self, mode=AI_MODE, endpoint=AI_ENDPOINT):
        self.mode=mode
        self.endpoint=endpoint.rstrip('/') if endpoint else ''

    def _remote(self, path, payload):
        if not self.endpoint:
            raise RuntimeError('AI_ENDPOINT is empty')
        r=requests.post(f'{self.endpoint}{path}',json=payload,timeout=30)
        r.raise_for_status()
        return r.json()

    def monitor(self, payload):
        if self.mode=='remote':return self._remote('/monitor',payload)
        label=METRIC_LABELS.get(payload.get('metric'),payload.get('metric'))
        return {'status':'warning','severity':payload.get('severity','WARNING'),'score':0.88,
                'title':f'{label}触发监测规则','reason':payload.get('reason','指标超过预设阈值'),'need_decision':True}

    def diagnose(self, company, event, history=None):
        if self.mode=='remote':return self._remote('/diagnose',{'company':dict(company),'event':event,'history':history or []})
        metric=event.get('metric','profit_margin')
        causes={
            'profit_margin':['经营成本上升','收入增速放缓','应收账款压力增加'],
            'cash_flow':['经营回款下降','资金占用扩大','项目投入集中'],
            'employee_sentiment':['员工状态指数下降','组织变动或工作负荷上升','关键岗位稳定性需关注'],
            'employee_turnover':['关键岗位流失加快','招聘补位速度不足','员工状态指数下行'],
            'project_delay_days':['关键里程碑延期','资源配置不足','跨部门协同效率下降'],
        }.get(metric,['指标偏离集团基准','历史趋势持续恶化','需要结合业务数据进一步核验'])
        return {'summary':f"{company['company_name']}的{METRIC_LABELS.get(metric,metric)}出现异常，建议进入决策模拟。",
                'causes':causes,'confidence':0.86,'evidence':['历史趋势','集团横向对比','当前规则阈值']}

    def chat(self, query, companies):
        if self.mode=='remote':return self._remote('/chat',{'query':query})
        q=query.strip()
        target=None
        for _,row in companies.iterrows():
            if str(row.company_name) in q or str(row.city) in q:
                target=int(row.company_id);break
        page='overview'
        if any(k in q for k in ['利润','原因','诊断','异常']):page='decision'
        elif any(k in q for k in ['孪生','关系','数据流']):page='twin'
        elif any(k in q for k in ['投资','方案','模拟']):page='investment'
        elif any(k in q for k in ['排名','排行']):page='ranking'
        elif any(k in q for k in ['画像','四象限']):page='portrait'
        elif any(k in q for k in ['指标','趋势','雷达']):page='profile'
        return {'reply':f'已解析指令：{q}。我会切换到相关分析页面并保留当前数据上下文。','page':page,'company_id':target}
