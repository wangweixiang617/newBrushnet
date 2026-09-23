from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from config import DATA_DIR, DATA_MODE, DATABASE_URL

class DataService:
    def __init__(self, mode=DATA_MODE, database_url=DATABASE_URL, data_dir=DATA_DIR):
        self.mode=mode
        self.database_url=database_url
        self.data_dir=Path(data_dir)
        self.engine=create_engine(database_url, future=True) if mode=='database' else None
        self.reload()

    def reload(self):
        if self.mode=='database':
            self.companies=pd.read_sql('companies',self.engine)
            self.history=pd.read_sql('history',self.engine,parse_dates=['month'])
            self.investments=pd.read_sql('investments',self.engine)
            self.rules=pd.read_sql('monitor_rules',self.engine)
        else:
            self.companies=pd.read_csv(self.data_dir/'companies.csv')
            self.history=pd.read_csv(self.data_dir/'history.csv',parse_dates=['month'])
            self.investments=pd.read_csv(self.data_dir/'investments.csv')
            self.rules=pd.read_csv(self.data_dir/'monitor_rules.csv')
        return self

    def company(self, company_id):
        row=self.companies[self.companies.company_id==int(company_id)]
        if row.empty:
            raise KeyError(f'company_id={company_id} not found')
        return row.iloc[0]

    def company_history(self, company_id):
        return self.history[self.history.company_id==int(company_id)].sort_values('month').copy()

    def company_investments(self, company_id):
        return self.investments[self.investments.company_id==int(company_id)].copy()

    def metric_value(self, company_id, metric):
        company=self.company(company_id)
        if metric in company.index:
            return float(company[metric])
        hist=self.company_history(company_id)
        if metric in hist.columns and not hist.empty:
            return float(hist.iloc[-1][metric])
        raise KeyError(f'metric={metric} not found')

    def metric_history(self, company_id, metric):
        hist=self.company_history(company_id)
        if metric not in hist.columns:
            return []
        return hist[['month',metric]].dropna().to_dict('records')

    def save_rule(self, rule):
        row=pd.DataFrame([rule])
        if self.mode=='database':
            row.to_sql('monitor_rules',self.engine,if_exists='append',index=False)
        else:
            current=pd.read_csv(self.data_dir/'monitor_rules.csv')
            pd.concat([current,row],ignore_index=True).to_csv(self.data_dir/'monitor_rules.csv',index=False,encoding='utf-8-sig')
        self.reload()

    def export_sqlite(self, sqlite_url='sqlite:///data/demo.db'):
        engine=create_engine(sqlite_url,future=True)
        self.companies.to_sql('companies',engine,if_exists='replace',index=False)
        self.history.to_sql('history',engine,if_exists='replace',index=False)
        self.investments.to_sql('investments',engine,if_exists='replace',index=False)
        self.rules.to_sql('monitor_rules',engine,if_exists='replace',index=False)
