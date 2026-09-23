import math, random, sqlite3
from datetime import date
from pathlib import Path
from config import DB_PATH,PROFILE_WEIGHTS
from .utils import now_iso

COMPANIES=[
    (1,'华北产业分公司','北京','华北',116.4074,39.9042,82,88,84,79,28),
    (2,'华东能源分公司','上海','华东',121.4737,31.2304,86,92,78,83,32),
    (3,'华南科技分公司','深圳','华南',114.0579,22.5431,83,87,96,85,21),
    (4,'华东创新分公司','杭州','华东',120.1551,30.2741,88,90,94,86,18),
    (5,'西南运营分公司','成都','西南',104.0665,30.5723,80,84,77,81,36),
    (6,'华中制造分公司','武汉','华中',114.3054,30.5931,78,82,75,76,42),
    (7,'西北资源分公司','西安','西北',108.9398,34.3416,81,79,72,84,38),
    (8,'东北工业分公司','沈阳','东北',123.4315,41.8057,77,76,70,74,48),
    (9,'华北新材料公司','天津','华北',117.2009,39.0842,84,85,89,82,26),
    (10,'华东数字科技公司','南京','华东',118.7969,32.0603,85,91,95,88,17),
    (11,'华南供应链公司','广州','华南',113.2644,23.1291,79,86,80,78,33),
    (12,'中部工程公司','长沙','华中',112.9388,28.2282,76,80,73,77,43),
    (13,'西南新能源公司','重庆','西南',106.5516,29.5630,87,89,90,85,23),
    (14,'华东装备公司','苏州','华东',120.5853,31.2989,82,83,81,86,29),
    (15,'华北基础设施公司','石家庄','华北',114.5149,38.0428,75,78,69,80,45),
    (16,'华南服务公司','厦门','华南',118.0894,24.4798,80,88,76,84,31),
    (17,'西部矿业公司','乌鲁木齐','西北',87.6168,43.8256,74,72,68,73,55),
    (18,'东北物流公司','大连','东北',121.6147,38.9140,78,81,71,79,40),
]

METRICS=[
    ('total_assets','总资产','basic','亿元','higher',0,0,1),
    ('registered_capital','注册资本','basic','亿元','higher',0,0,0),
    ('net_assets','净资产','basic','亿元','higher',0,0,1),
    ('employee_count','员工规模','basic','人','higher',0,0,0),
    ('revenue','营业收入','operation','亿元','higher',0,0,1),
    ('net_profit','净利润','operation','亿元','higher',0,0,1),
    ('roe','ROE','operation','%','higher',0,0,1),
    ('cash_flow','经营现金流','operation','亿元','higher',0,0,1),
    ('rd_investment','研发投入','technology','亿元','higher',0,0,1),
    ('rd_ratio','研发强度','technology','%','higher',0,0,1),
    ('patent_count','有效专利','technology','件','higher',0,0,0),
    ('rd_staff_ratio','研发人员占比','technology','%','higher',0,0,1),
    ('supervision_score','监管合规评分','supervision','分','higher',70,60,0),
    ('debt_ratio','资产负债率','supervision','%','lower',65,75,1),
    ('audit_issues','审计异常','supervision','项','lower',3,6,0),
    ('risk_events','风险事件','supervision','项','lower',2,5,0),
    ('employee_sentiment','员工状态指数','human','分','higher',65,55,1),
    ('investment_roi','投资ROI','investment','%','higher',8,4,1),
]

PROFILE_NAMES={'basic':'基础画像','operation':'经营画像','technology':'科技画像','supervision':'监管画像'}

INVESTMENT_CATEGORIES={
    '产业服务': [('供应链','供应链协同平台'),('数字运营','集团数字化升级'),('市场网络','区域服务网络')],
    '先进制造': [('智能装备','智能产线改造'),('新材料','高性能材料项目'),('工业软件','制造执行平台')],
    '数字产业': [('人工智能','AI决策平台'),('数据资产','集团数据湖'),('云平台','私有云升级')],
    '新能源': [('储能','储能示范项目'),('光伏','分布式光伏'),('氢能','氢能研发基地')],
}


def connect():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(DB_PATH,timeout=30,check_same_thread=False)
    conn.row_factory=sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL');conn.execute('PRAGMA foreign_keys=ON')
    return conn


def row(sql,args=()):
    with connect() as c:
        r=c.execute(sql,args).fetchone();return dict(r) if r else None


def rows(sql,args=()):
    with connect() as c:return [dict(r) for r in c.execute(sql,args).fetchall()]


def execute(sql,args=()):
    with connect() as c:
        cur=c.execute(sql,args);c.commit();return cur.lastrowid


def _month_iter(start_year=2021,months=60):
    y,m=start_year,1
    for _ in range(months):
        yield f'{y:04d}-{m:02d}'
        m+=1
        if m>12:y+=1;m=1


def _score(b,o,t,s):
    return round(b*PROFILE_WEIGHTS['basic']+o*PROFILE_WEIGHTS['operation']+t*PROFILE_WEIGHTS['technology']+s*PROFILE_WEIGHTS['supervision'],1)


def _metric_value(key,company,idx,rng):
    cid,name,city,region,lon,lat,basic,operation,tech,sup,risk=company
    x=idx/59
    season=math.sin(idx*math.pi/6)*0.035
    noise=lambda scale:rng.uniform(-scale,scale)
    revenue_base=60+operation*0.9+cid*1.2
    revenue=revenue_base*(0.72+0.28*x)*(1+season+noise(.018))
    net_profit=revenue*(0.055+(operation-70)*0.0025+noise(.006))
    net_assets=(55+basic*.95+cid*.8)*(0.86+0.14*x)*(1+noise(.01))
    total_assets=net_assets*(1.55+(100-sup)*.006)*(1+noise(.008))
    roe=100*net_profit/max(net_assets,1)
    cash=net_profit*(1.05+noise(.18))
    employees=(900+basic*22+cid*35)*(0.9+0.1*x)*(1+noise(.012))
    rd= revenue*(0.025+(tech-65)*0.0013)*(1+noise(.04))
    rd_ratio=100*rd/max(revenue,1)
    patents=(18+(tech-60)*2.4+idx*.65+cid*1.5)*(1+noise(.03))
    rd_staff=8+(tech-65)*.34+noise(.5)
    supervision=max(52,min(99,sup+math.sin(idx/8)*1.8+noise(1.1)))
    debt=max(20,min(84,52+(100-sup)*.37+math.sin(idx/9)*2+noise(1.8)))
    audit=max(0,round((100-sup)/8+noise(1.0)))
    risk_events=max(0,round((risk-20)/12+noise(.8)))
    sentiment=max(48,min(95,70+(operation-80)*.45+(sup-80)*.2+math.sin(idx/5)*2.5+noise(2)))
    roi=max(1,8+(operation-78)*.35+(tech-78)*.18+noise(1.2))
    values={
      'total_assets':total_assets,'registered_capital':15+basic*.38+cid*.7,'net_assets':net_assets,'employee_count':employees,
      'revenue':revenue,'net_profit':net_profit,'roe':roe,'cash_flow':cash,
      'rd_investment':rd,'rd_ratio':rd_ratio,'patent_count':patents,'rd_staff_ratio':rd_staff,
      'supervision_score':supervision,'debt_ratio':debt,'audit_issues':audit,'risk_events':risk_events,
      'employee_sentiment':sentiment,'investment_roi':roi}
    return round(values[key],2)


def init_db(reset=False):
    if reset and DB_PATH.exists():DB_PATH.unlink()
    with connect() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS companies(
          company_id INTEGER PRIMARY KEY,company_name TEXT NOT NULL,city TEXT,region TEXT,longitude REAL,latitude REAL,
          basic_score REAL,operation_score REAL,technology_score REAL,supervision_score REAL,composite_score REAL,risk_score REAL,
          status TEXT DEFAULT 'NORMAL',updated_at TEXT);
        CREATE TABLE IF NOT EXISTS metric_catalog(
          metric_key TEXT PRIMARY KEY,metric_name TEXT NOT NULL,profile TEXT NOT NULL,unit TEXT,direction TEXT,
          warning_threshold REAL,critical_threshold REAL,forecastable INTEGER DEFAULT 1,sort_order INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS metric_values(
          company_id INTEGER NOT NULL,metric_key TEXT NOT NULL,period TEXT NOT NULL,value REAL NOT NULL,
          PRIMARY KEY(company_id,metric_key,period),FOREIGN KEY(company_id) REFERENCES companies(company_id));
        CREATE TABLE IF NOT EXISTS investments(
          id INTEGER PRIMARY KEY AUTOINCREMENT,company_id INTEGER,category TEXT,subcategory TEXT,project_name TEXT,
          amount REAL,roi REAL,npv REAL,risk TEXT,progress REAL,FOREIGN KEY(company_id) REFERENCES companies(company_id));
        CREATE TABLE IF NOT EXISTS monitor_rules(
          id INTEGER PRIMARY KEY AUTOINCREMENT,domain TEXT,metric TEXT,op TEXT,threshold REAL,severity TEXT,enabled INTEGER,speak INTEGER,auto_decision INTEGER);
        CREATE TABLE IF NOT EXISTS events(
          id INTEGER PRIMARY KEY AUTOINCREMENT,company_id INTEGER,metric TEXT,severity TEXT,title TEXT,message TEXT,value REAL,threshold REAL,op TEXT,status TEXT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS decisions(
          id INTEGER PRIMARY KEY AUTOINCREMENT,company_id INTEGER,event_id INTEGER,option_id TEXT,option_name TEXT,operator TEXT,rationale TEXT,snapshot_json TEXT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT,action TEXT,payload TEXT,created_at TEXT);
        ''')
        if c.execute('SELECT COUNT(*) FROM companies').fetchone()[0]==0:
            now=now_iso()
            for comp in COMPANIES:
                cid,name,city,region,lon,lat,b,o,t,s,risk=comp
                c.execute('INSERT INTO companies VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(cid,name,city,region,lon,lat,b,o,t,s,_score(b,o,t,s),risk,'NORMAL',now))
        if c.execute('SELECT COUNT(*) FROM metric_catalog').fetchone()[0]==0:
            for order,m in enumerate(METRICS):c.execute('INSERT INTO metric_catalog VALUES(?,?,?,?,?,?,?,?,?)',(*m,order))
        if c.execute('SELECT COUNT(*) FROM metric_values').fetchone()[0]==0:
            for comp in COMPANIES:
                rng=random.Random(20260923+comp[0]*97)
                for idx,period in enumerate(_month_iter()):
                    for key,*_ in METRICS:
                        c.execute('INSERT INTO metric_values(company_id,metric_key,period,value) VALUES(?,?,?,?)',(comp[0],key,period,_metric_value(key,comp,idx,rng)))
        if c.execute('SELECT COUNT(*) FROM investments').fetchone()[0]==0:
            rng=random.Random(4281)
            for comp in COMPANIES:
                cid=comp[0]
                for category,subs in INVESTMENT_CATEGORIES.items():
                    for sub,project in subs:
                        amount=round(rng.uniform(2.8,16.5)*(0.8+comp[9]/100),2)
                        roi=round(rng.uniform(7,24)+(comp[8]-80)*.12,2)
                        npv=round(amount*rng.uniform(.75,1.8),2)
                        risk=rng.choice(['LOW','LOW','MEDIUM','MEDIUM','HIGH'])
                        progress=round(rng.uniform(25,96),1)
                        c.execute('INSERT INTO investments(company_id,category,subcategory,project_name,amount,roi,npv,risk,progress) VALUES(?,?,?,?,?,?,?,?,?)',(cid,category,sub,project,amount,roi,npv,risk,progress))
        if c.execute('SELECT COUNT(*) FROM monitor_rules').fetchone()[0]==0:
            rules=[('经营画像','roe','<',8,'WARNING',1,1,1),('监管画像','supervision_score','<',72,'CRITICAL',1,1,1),('人力状态','employee_sentiment','<',60,'WARNING',1,1,0),('经营画像','cash_flow','<',5,'CRITICAL',1,1,1),('监管画像','debt_ratio','>',72,'WARNING',1,0,1),('科技画像','rd_ratio','<',3.5,'NOTICE',1,0,0)]
            c.executemany('INSERT INTO monitor_rules(domain,metric,op,threshold,severity,enabled,speak,auto_decision) VALUES(?,?,?,?,?,?,?,?)',rules)
        c.commit()
    return DB_PATH


def log_action(action,payload):
    return execute('INSERT INTO audit_log(action,payload,created_at) VALUES(?,?,?)',(action,payload,now_iso()))
