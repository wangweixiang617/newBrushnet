import numpy as np
from config import SIMULATION_RUNS

class DecisionEngine:
    def __init__(self, runs=SIMULATION_RUNS):
        self.runs=runs

    @staticmethod
    def _npv(rate,cashflows):
        return float(sum(cf/((1+rate)**i) for i,cf in enumerate(cashflows)))

    def generate_options(self, company, trigger_metric='profit_margin'):
        base_profit=max(float(company['revenue'])*float(company['profit_margin'])/100,0.1)
        seed=int(company['company_id'])+sum(map(ord,trigger_metric))
        configs=[
            ('OPTION A','成本优化','集中采购、压缩低效支出、优化预算执行',0.035,0.11,0.55,'LOW'),
            ('OPTION B','增长提升','重点客户策略、区域增量计划、优化价格与渠道',0.075,0.18,0.85,'HIGH'),
            ('OPTION C','组合治理','成本优化与收入提升并行，分阶段推进并设置止损点',0.055,0.15,0.72,'MEDIUM'),
        ]
        rng=np.random.default_rng(seed)
        options=[]
        for code,name,summary,investment_ratio,benefit_ratio,success_base,risk in configs:
            investment=float(company['revenue'])*investment_ratio
            annual_benefit=float(company['revenue'])*benefit_ratio
            samples=rng.normal(annual_benefit,annual_benefit*0.28,self.runs)
            cost_samples=rng.normal(investment,investment*0.12,self.runs)
            net=samples-cost_samples
            success=float((net>0).mean())
            cashflows=[-investment]+[annual_benefit*(1+0.03*i) for i in range(1,4)]
            npv=self._npv(0.08,cashflows)
            options.append({
                'code':code,'name':name,'summary':summary,'investment':round(investment,2),
                'annual_benefit':round(annual_benefit,2),'npv':round(npv,2),'success_prob':round(success*100,1),
                'risk':risk,'profit_uplift':round((annual_benefit/base_profit)*float(company['profit_margin'])*0.22,2),
                'distribution':np.percentile(net,[10,50,90]).round(2).tolist()
            })
        return options
