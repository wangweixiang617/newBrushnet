## 集团级 AI 企业数字孪生监测与决策平台

这是一个可直接运行和部署的 Python 原型项目，定位为大型集团总部管理下属分公司使用的 **AI Enterprise Digital Twin / AI 决策驾驶舱**。

项目把三类能力放在同一个系统中：

- 企业经营分析：集团总览、全国分公司地图、排行、画像四象限、企业全景、雷达图、趋势、指标分析、投资 Sunburst。
- 高级数字孪生可视化：深色科幻 UI、发光面板、动态数据流、节点状态、实时 AI 事件弹窗。
- AI 监测与决策闭环：自定义监测规则、后台检测、异常预警、自动语音播报、AI 诊断、Option A/B/C、Monte Carlo 风格模拟、人工选择方案。

当前版本是**可运行的工程骨架 + 演示数据 + 可替换 AI 接口**。默认不依赖真实传感器，数据来自 CSV 或数据库。

### 1. 已实现功能

当前代码包包含以下功能：

1. **企业总览 Dashboard**：集团收入、利润率、当前企业评分、AI 事件总数、全国分公司分布、集团趋势和风险排行。
2. **全国企业地图**：使用分公司经纬度显示全国布局，可作为后续省份/地市下钻的入口。
3. **企业排行**：集团下属企业综合评分排行与明细表。
4. **企业画像四象限**：企业价值指数 × 成长指数，当前企业高亮显示。
5. **企业全景详情**：雷达图、利润/员工状态历史趋势、自定义指标趋势与简单预测。
6. **投资分析**：集团投资 Sunburst、分公司投资 NPV 排名、当前企业投资额与 ROI。
7. **Digital Twin 动态关系/数据流**：企业 → 财务/经营/人力/投资/项目/风险 Twin → AI CORE → DECISION ENGINE；连线动态流动，异常节点改变状态。
8. **AI Monitor**：可新增检测指标、设置阈值、严重级别、自动播报和自动决策入口。
9. **AI Event**：后台按规则持续扫描数据，生成事件并去重。
10. **AI 弹窗预警**：出现新事件时自动弹出高级风格 Modal。
11. **自动语音播报**：浏览器支持 Web Speech API 时，重要事件自动中文播报。
12. **AI 决策中心**：异常趋势、原因分析、三套 Option、模拟投入/NPV/成功概率/风险。
13. **AI Assistant**：文字输入 + 浏览器语音输入，可解析“查看画像、分析利润、进入数字孪生、看投资”等指令。
14. **REST AI 接口**：`/api/ai/monitor`、`/api/ai/diagnose`、`/api/ai/options`、`/api/ai/chat` 已预留。
15. **CSV / SQLite / PostgreSQL 路线**：默认 CSV；通过 `DATA_MODE=database` 和 SQLAlchemy URL 可切换数据库。
16. **Docker / Gunicorn 部署**：已经包含 `Dockerfile`、`docker-compose.yml` 和 `gunicorn.conf.py`。

### 2. 技术栈

核心保持 Python 为主：

```text
Dash                Web 主框架
Plotly              图表、地图、Sunburst、雷达图
Dash Cytoscape      数字孪生关系与动态数据流
Pandas / NumPy      数据处理
SciPy               后续优化/统计模型扩展
scikit-learn        后续聚类、相似企业、异常检测扩展
SQLAlchemy          数据库访问
APScheduler         第一阶段后台定时监测
Flask API           Dash 内部提供 REST AI 接口
CSS + 少量 JS       高级科幻视觉、TTS、语音输入
Gunicorn            Linux 生产部署
Docker               容器部署
```

第一阶段没有引入 Celery/Redis，便于学习和部署。正式集团生产环境建议后续把后台任务迁移到 Celery/RQ + Redis，并把事件存储迁移到数据库或消息队列。

### 3. 项目目录

```text
group_ai_digital_twin/
├── app.py                     # Dash 主入口 + API + callbacks
├── config.py                  # 环境配置
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── gunicorn.conf.py
├── .env
├── .env.example
│
├── assets/
│   ├── style.css              # 科幻 UI / 动画
│   └── app.js                 # 语音输入 + TTS
│
├── data/
│   ├── companies.csv          # 分公司主数据
│   ├── history.csv            # 月度历史指标
│   ├── investments.csv        # 投资项目
│   └── monitor_rules.csv      # AI 检测规则
│
├── services/
│   ├── data_service.py        # CSV / DB 数据层
│   ├── analytics.py           # 综合评分、预测等
│   ├── monitor_engine.py      # 后台 AI Monitor / EventStore
│   ├── ai_gateway.py          # AI 接口抽象；mock/remote
│   └── decision_engine.py     # Option A/B/C + 模拟
│
├── ui/
│   ├── components.py          # 公共 UI 组件
│   └── pages.py               # 各业务页面
│
├── scripts/
│   └── init_sqlite.py         # CSV 初始化 SQLite
│
└── tests/
    └── smoke_test.py
```

### 4. 最快运行方式：Python 本地启动

建议使用 Python 3.11。

Linux / macOS：

```bash
cd group_ai_digital_twin
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Windows PowerShell：

```powershell
cd group_ai_digital_twin
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

浏览器访问：

```text
http://127.0.0.1:8050
```

默认监听 `0.0.0.0:8050`，同一局域网其他机器可以使用：

```text
http://服务器IP:8050
```

### 5. Docker 一键部署

项目已经自带 `.env`，直接执行：

```bash
cd group_ai_digital_twin
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f
```

停止：

```bash
docker compose down
```

访问：

```text
http://服务器IP:8050
```

Docker 当前使用一个 Gunicorn worker + 多线程，原因是演示版后台 APScheduler 与内存 EventStore 需要保持单实例。正式集群部署时应把后台监测任务和事件存储拆出去，再增加 Web worker 数量。

### 6. Ubuntu 服务器部署

如果服务器已经安装 Docker：

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker

cd /opt
git clone <你的代码仓库地址> group_ai_digital_twin
cd group_ai_digital_twin
cp .env.example .env
docker compose up -d --build
```

防火墙放行演示端口：

```bash
sudo ufw allow 8050/tcp
```

生产环境不建议长期直接暴露 8050，建议使用：

```text
Internet
   ↓
Nginx / HTTPS
   ↓
Gunicorn
   ↓
Dash
```

### 7. 切换到 SQLite 数据库

先把当前 CSV 导入 SQLite：

```bash
python scripts/init_sqlite.py
```

修改 `.env`：

```text
DATA_MODE=database
DATABASE_URL=sqlite:///data/demo.db
```

重新启动即可。

### 8. 切换到 PostgreSQL

安装和创建 PostgreSQL 数据库后，将四张表导入：

```text
companies
history
investments
monitor_rules
```

然后修改 `.env`：

```text
DATA_MODE=database
DATABASE_URL=postgresql+psycopg://user:password@db-host:5432/twin
```

并在 `requirements.txt` 中增加：

```text
psycopg[binary]
```

`DataService` 使用 SQLAlchemy，因此页面层不需要因为数据库变化而重写。

### 9. 演示数据字段说明

`companies.csv` 主要字段：

```text
company_id
company_name
region / province / city
lat / lon
industry
revenue
profit_margin
cash_flow
growth
debt_ratio
receivables_growth
employee_turnover
employee_sentiment
project_delay_days
operation_score
finance_score
hr_score
innovation_score
investment_score
market_score
compliance_score
esg_score
risk_score
```

`history.csv` 保存月度历史数据：

```text
company_id
month
revenue
profit_margin
cash_flow
employee_sentiment
employee_turnover
project_progress
operating_cost
```

替换为真实集团数据时，推荐先保留这些字段名，使第一版代码不需要修改；后续再建立字段映射层。

### 10. AI Monitor 逻辑

当前 AI Monitor 流程：

```text
CSV / Database
      ↓
MonitorEngine
      ↓
读取 monitor_rules
      ↓
检查阈值
      ↓
AIGateway.monitor()
      ↓
EventStore
      ↓
网页轮询新事件
      ↓
弹窗 + TTS
      ↓
AI Diagnosis
      ↓
Option A / B / C
```

系统启动时会自动执行一次检测，随后根据：

```text
MONITOR_INTERVAL_SECONDS=30
```

持续检测。

### 11. 新增监测指标

AI Monitor 页面可以新增规则，也可以直接编辑 `data/monitor_rules.csv`。

示例：

```text
domain,metric,op,threshold,severity,enabled,speak,auto_decision
财务,profit_margin,<,8,CRITICAL,1,1,1
员工状态,employee_sentiment,<,65,WARNING,1,1,1
项目,project_delay_days,>,10,WARNING,1,0,1
```

目前已经示范：

- 财务：利润率、现金流、应收账款。
- 人力：流失率。
- 员工状态：聚合员工状态指数。
- 项目：延期天数。
- 风险：风险指数。
- 投资、经营、合规。

员工状态建议只使用**匿名、聚合、组织级**数据，例如匿名问卷、部门总体趋势，不建议对具体个人做情绪标签或自动人事决策。

### 12. AI 接口如何替换为真实模型

默认：

```text
AI_MODE=mock
```

`services/ai_gateway.py` 会返回本地模拟结果，使整个系统无外部 API 也能完整演示。

未来接集团内部 AI 服务时：

```text
AI_MODE=remote
AI_ENDPOINT=http://ai-service:9000
```

外部 AI 服务需要实现：

```text
POST /monitor
POST /diagnose
POST /chat
```

也可以直接修改 `AIGateway`，接入你自己的大模型、RAG、异常检测模型或企业智能体。

推荐架构：

```text
Dash
  ↓
受控 AI Gateway
  ↓
Intent / Agent Router
  ├── get_company
  ├── get_metric
  ├── compare_company
  ├── detect_anomaly
  ├── diagnose
  ├── generate_options
  └── simulate_option
```

不要让大模型直接拥有数据库写权限；真正的决策、审批、资金操作必须经过业务 API 和权限系统。

### 13. 当前 REST API

健康检查：

```bash
curl http://127.0.0.1:8050/api/health
```

监测接口：

```bash
curl -X POST http://127.0.0.1:8050/api/ai/monitor \
  -H 'Content-Type: application/json' \
  -d '{"company_id":1001,"metric":"profit_margin","current_value":6.7,"severity":"CRITICAL","reason":"连续下降"}'
```

诊断接口：

```bash
curl -X POST http://127.0.0.1:8050/api/ai/diagnose \
  -H 'Content-Type: application/json' \
  -d '{"company_id":1001,"event":{"metric":"profit_margin"}}'
```

方案接口：

```bash
curl -X POST http://127.0.0.1:8050/api/ai/options \
  -H 'Content-Type: application/json' \
  -d '{"company_id":1001,"metric":"profit_margin"}'
```

AI 助手：

```bash
curl -X POST http://127.0.0.1:8050/api/ai/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"分析上海分公司利润异常"}'
```

### 14. 语音输入和自动播报

语音逻辑在：

```text
assets/app.js
```

使用浏览器原生：

```text
SpeechRecognition / webkitSpeechRecognition
speechSynthesis
```

因此推荐使用最新版 Chrome / Edge。

注意：浏览器语音识别的支持程度取决于浏览器、操作系统和网络环境。如果集团内网浏览器不支持，可以把语音识别替换为服务端 STT：

```text
Microphone
   ↓
Audio Upload / WebSocket
   ↓
Whisper / 企业 STT
   ↓
文字
   ↓
AI Assistant
```

TTS 同理可以替换成集团内部语音服务。

### 15. 决策模拟的实现逻辑

当前 `DecisionEngine` 会为异常生成三类方案：

```text
OPTION A  成本优化
OPTION B  增长提升
OPTION C  组合治理
```

随后通过随机采样模拟：

```text
预期收益
投入成本
净收益分布
成功概率
3 年 NPV
风险等级
```

这是演示级逻辑，便于验证整个产品交互流程。真实项目应把 `DecisionEngine` 替换为集团认可的财务模型、预算模型、业务预测模型和约束优化模型。

### 16. Digital Twin 动态数据流

数字孪生页面使用 `dash-cytoscape`：

```text
当前企业
   ↓
财务 Twin
经营 Twin
人力 Twin
投资 Twin
项目 Twin
风险 Twin
   ↓
AI CORE
   ↓
DECISION ENGINE
```

`dcc.Interval` 持续改变 `line-dash-offset`，形成流动数据线效果。

后续可以继续增加：

- 供应链 Twin
- 市场 Twin
- 审计 Twin
- 法务 Twin
- ESG Twin
- 资产 Twin
- 客户 Twin

节点状态由真实监测数据驱动，而不是固定动画。

### 17. 如何学习这个项目

建议按下面顺序阅读，不要一上来从 `app.py` 的 callback 全部硬看。

第一步，看数据：

```text
data/companies.csv
data/history.csv
data/investments.csv
data/monitor_rules.csv
```

理解前端看到的数字从哪里来。

第二步，看数据层：

```text
services/data_service.py
```

理解 CSV 与数据库怎样统一成同一套 DataFrame 接口。

第三步，看普通分析：

```text
services/analytics.py
```

理解综合评分、排行、趋势预测。

第四步，看 AI 监测：

```text
services/monitor_engine.py
```

重点理解：

```text
规则 → 指标值 → 命中 → EventStore
```

第五步，看 AI 接口：

```text
services/ai_gateway.py
```

理解为什么前端不直接绑定某一个大模型。

第六步，看决策模拟：

```text
services/decision_engine.py
```

理解 Option A/B/C 的生成和模拟结果。

第七步，看页面：

```text
ui/pages.py
```

主要是 Plotly/Cytoscape 图形。

第八步，看交互：

```text
app.py
```

重点学习 Dash callback：

```text
Input
State
Output
ctx.triggered_id
```

第九步，看视觉：

```text
assets/style.css
assets/app.js
```

这两部分负责科幻 UI、扫描线、流动效果、语音输入和 TTS。

### 18. 推荐的二次开发顺序

不要一次把所有生产功能都加进去。推荐：

```text
阶段 1
替换真实集团/分公司数据
        ↓
阶段 2
确认财务、经营、人力、投资指标口径
        ↓
阶段 3
接 PostgreSQL / 数据仓库
        ↓
阶段 4
把监测规则改成真实规则
        ↓
阶段 5
接真实 AI / RAG
        ↓
阶段 6
接审批、权限和审计
        ↓
阶段 7
Celery + Redis + WebSocket
        ↓
阶段 8
SSO / HTTPS / 高可用
```

### 19. 正式集团部署前必须补的能力

当前项目是工程原型，不应该未经加固直接承载真实集团决策。正式生产至少需要增加：

- SSO / LDAP / OAuth2 身份认证。
- 集团、区域、分公司、部门级 RBAC 数据权限。
- PostgreSQL 等正式数据库。
- 审计日志不可篡改存储。
- AI 输出引用的数据来源、时间、计算版本。
- 决策方案人工确认与审批工作流。
- 模型版本管理和回溯。
- 告警去重、升级、关闭、责任人机制。
- Celery/RQ + Redis/Kafka 等独立任务与消息系统。
- WebSocket/SSE 实时事件推送。
- HTTPS、WAF、密钥管理、备份恢复。
- 多实例部署时移除内存 EventStore。
- 员工状态数据的匿名化、最小化和合规治理。

### 20. 常见问题

#### 页面能运行，但全国地图没有底图

`Scattergeo` 的地理资源在某些离线环境可能受 Plotly 地理数据加载方式影响。公网部署通常没有问题。纯内网环境建议将集团自有 GeoJSON/行政区边界文件放入 `assets/`，改为本地 GeoJSON 绘制。

#### 为什么 Gunicorn 只有 1 个 worker

因为演示版将 APScheduler 和 EventStore 放在 Web 进程里。多 worker 会导致重复监测和事件不共享。生产版应把它们拆成独立服务，然后将 Web worker 扩展到多个。

#### 为什么员工情绪没有分析具体个人

集团级系统建议使用匿名聚合的员工状态/满意度趋势，用于组织治理，而不是对个人自动做情绪标签或人事决策。

#### 能不能接 Excel

可以。`DataService` 当前以 CSV 为默认示例；可以用 `pandas.read_excel()` 增加 Excel loader，或者先将 Excel 导入数据库。项目已经安装 `openpyxl`。

#### 能不能换成 FastAPI

可以。当前为了降低部署复杂度，REST API 直接挂在 Dash 自带 Flask server。后续可以把 `services/` 原样保留，外面换成：

```text
FastAPI
   ├── REST / WebSocket
   └── Mount Dash / 独立前端
```

业务服务层无需推翻。

### 21. 快速验证

安装依赖后运行：

```bash
python tests/smoke_test.py
```

正常输出：

```text
smoke test passed
```

然后运行：

```bash
python app.py
```

进入 `AI监测` 页面，点击：

```text
触发演示预警
```

即可看到完整链路：

```text
预警事件
  ↓
弹窗
  ↓
自动语音播报
  ↓
进入 AI 分析
  ↓
趋势诊断
  ↓
Option A / B / C
  ↓
选择方案
```

这条链路是当前代码包最重要的学习入口。
