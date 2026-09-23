import hashlib, math, random


def _npv(rate,cashflows):
    return sum(cf/((1+rate)**i) for i,cf in enumerate(cashflows))


def simulate_options(company,options,runs=5000):
    results=[]
    for option in options:
        seed=int(hashlib.sha256(f"{company['company_id']}:{option['id']}".encode()).hexdigest()[:8],16)
        rng=random.Random(seed)
        npvs=[];success=0
        budget=float(option['budget_m'])
        lift=float(option['expected_profit_lift'])
        base_rev=float(company['revenue'])
        risk_factor={'LOW':0.72,'MEDIUM':1.0,'HIGH':1.35}.get(option['risk'],1.0)
        for _ in range(runs):
            adoption=max(0.2,min(1.35,rng.gauss(0.88,0.13*risk_factor)))
            market=max(0.65,min(1.35,rng.gauss(1.0,0.09*risk_factor)))
            annual_benefit=base_rev*(lift/100.0)*10*adoption*market
            op_cost=budget*0.13*rng.uniform(0.85,1.22)
            cash=[-budget]
            for year in range(1,4):
                decay=max(0.72,1-(year-1)*0.07)
                cash.append(annual_benefit*decay-op_cost)
            value=_npv(0.08,cash)
            npvs.append(value)
            target={'LOW':0.74,'MEDIUM':0.81,'HIGH':0.89}.get(option['risk'],0.81)
            if value>0 and adoption*market>=target:success+=1
        npvs.sort()
        mean=sum(npvs)/runs
        q=lambda p:npvs[min(runs-1,max(0,int((runs-1)*p)))]
        results.append({
          **option,'runs':runs,'npv_m':round(mean,2),'success_probability':round(success/runs*100,1),
          'p10':round(q(.10),2),'p50':round(q(.50),2),'p90':round(q(.90),2),
          'annual_benefit_m':round(base_rev*(lift/100.0)*10,2)
        })
    return results
