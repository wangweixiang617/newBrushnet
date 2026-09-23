import random
from .utils import clamp


def simulate_options(company,options,runs=5000):
    seed=company['company_id']*991+runs
    rng=random.Random(seed);out=[]
    for o in options:
        successes=0;npvs=[];profits=[]
        for _ in range(runs):
            execution=rng.gauss(o['execution_factor'],o['volatility'])
            market=rng.gauss(1.0,.08);risk_penalty=rng.uniform(0,o['risk_penalty'])
            profit=max(-10,o['expected_profit_m']*execution*market-risk_penalty)
            npv=profit*o['npv_factor']-o['budget_m']
            npvs.append(npv);profits.append(profit)
            if npv>0 and profit>o['success_floor']:successes+=1
        s=dict(o);s['success_probability']=round(successes/runs*100,1);s['simulated_npv_m']=round(sum(npvs)/len(npvs),1);s['simulated_profit_m']=round(sum(profits)/len(profits),1);s['npv_low_m']=round(sorted(npvs)[int(runs*.1)],1);s['npv_high_m']=round(sorted(npvs)[int(runs*.9)],1);out.append(s)
    return out
