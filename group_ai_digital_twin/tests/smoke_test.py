from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from services import DataService, DecisionEngine, AIGateway, EventStore, MonitorEngine
from services.analytics import add_scores


def main():
    ds=DataService(mode='csv')
    assert len(ds.companies)>=5
    scored=add_scores(ds.companies)
    assert 'composite_score' in scored.columns
    company=ds.company(int(ds.companies.iloc[0].company_id))
    options=DecisionEngine(runs=200).generate_options(company)
    assert len(options)==3
    events=EventStore();monitor=MonitorEngine(ds,AIGateway(mode='mock'),events)
    monitor.run_all()
    assert events.latest() is not None
    print('smoke test passed')

if __name__=='__main__':main()
