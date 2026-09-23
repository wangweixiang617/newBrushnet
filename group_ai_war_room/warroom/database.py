import csv, sqlite3, threading
from pathlib import Path
from config import DATA_DIR, DB_PATH
from .utils import now_iso

_LOCK=threading.RLock()


def connect():
    conn=sqlite3.connect(DB_PATH,check_same_thread=False,timeout=30)
    conn.row_factory=sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    return conn


def _read_csv(name):
    with open(DATA_DIR/name,'r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def init_db(force=False):
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    if force and DB_PATH.exists():DB_PATH.unlink()
    with _LOCK,connect() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS companies(
          company_id INTEGER PRIMARY KEY, company_name TEXT, region TEXT, province TEXT, city TEXT,
          lat REAL, lon REAL, industry TEXT, revenue REAL, profit_margin REAL, cash_flow REAL, growth REAL,
          debt_ratio REAL, receivables_growth REAL, employee_turnover REAL, employee_sentiment REAL,
          project_delay_days REAL, operation_score REAL, finance_score REAL, hr_score REAL, innovation_score REAL,
          investment_score REAL, market_score REAL, compliance_score REAL, esg_score REAL, risk_score REAL
        );
        CREATE TABLE IF NOT EXISTS history(
          id INTEGER PRIMARY KEY AUTOINCREMENT, company_id INTEGER, month TEXT, revenue REAL, profit_margin REAL,
          cash_flow REAL, employee_sentiment REAL, employee_turnover REAL, project_progress REAL, operating_cost REAL
        );
        CREATE TABLE IF NOT EXISTS investments(
          project_id INTEGER PRIMARY KEY, company_id INTEGER, sector TEXT, subsector TEXT, project TEXT,
          amount REAL, roi REAL, npv REAL, risk TEXT
        );
        CREATE TABLE IF NOT EXISTS monitor_rules(
          id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT, metric TEXT, op TEXT, threshold REAL, severity TEXT,
          enabled INTEGER DEFAULT 1, speak INTEGER DEFAULT 0, auto_decision INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS events(
          id INTEGER PRIMARY KEY AUTOINCREMENT, event_key TEXT, company_id INTEGER, company_name TEXT, domain TEXT,
          metric TEXT, value REAL, threshold REAL, op TEXT, severity TEXT, title TEXT, message TEXT,
          status TEXT DEFAULT 'OPEN', created_at TEXT, acknowledged_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC);
        CREATE TABLE IF NOT EXISTS decisions(
          id INTEGER PRIMARY KEY AUTOINCREMENT, company_id INTEGER, event_id INTEGER, option_id TEXT,
          option_name TEXT, operator TEXT, rationale TEXT, snapshot_json TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS audit_log(
          id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT, payload_json TEXT, created_at TEXT
        );
        ''')
        count=c.execute('SELECT COUNT(*) FROM companies').fetchone()[0]
        if count==0:
            companies=_read_csv('companies.csv')
            cols=list(companies[0])
            q=','.join('?' for _ in cols)
            for r in companies:c.execute(f"INSERT INTO companies({','.join(cols)}) VALUES({q})",[r[k] for k in cols])
        count=c.execute('SELECT COUNT(*) FROM history').fetchone()[0]
        if count==0:
            rows=_read_csv('history.csv');cols=list(rows[0])
            q=','.join('?' for _ in cols)
            for r in rows:c.execute(f"INSERT INTO history({','.join(cols)}) VALUES({q})",[r[k] for k in cols])
        count=c.execute('SELECT COUNT(*) FROM investments').fetchone()[0]
        if count==0:
            rows=_read_csv('investments.csv');cols=list(rows[0])
            q=','.join('?' for _ in cols)
            for r in rows:c.execute(f"INSERT INTO investments({','.join(cols)}) VALUES({q})",[r[k] for k in cols])
        count=c.execute('SELECT COUNT(*) FROM monitor_rules').fetchone()[0]
        if count==0:
            rows=_read_csv('monitor_rules.csv');cols=list(rows[0])
            q=','.join('?' for _ in cols)
            for r in rows:c.execute(f"INSERT INTO monitor_rules({','.join(cols)}) VALUES({q})",[r[k] for k in cols])
        c.commit()


def rows(sql,args=()):
    with _LOCK,connect() as c:return [dict(x) for x in c.execute(sql,args).fetchall()]


def row(sql,args=()):
    with _LOCK,connect() as c:
        r=c.execute(sql,args).fetchone();return dict(r) if r else None


def execute(sql,args=()):
    with _LOCK,connect() as c:
        cur=c.execute(sql,args);c.commit();return cur.lastrowid


def log_action(action,payload_json='{}'):
    return execute('INSERT INTO audit_log(action,payload_json,created_at) VALUES(?,?,?)',(action,payload_json,now_iso()))
