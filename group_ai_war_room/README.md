## 集团级 AI 数字孪生作战指挥平台

这是一个可以直接运行、直接部署、直接交付演示的 **集团级 AI Enterprise Digital Twin / War Room** 项目。

它不是普通 BI Dashboard。设计目标是：

- 集团总部在作战室大屏持续查看全国分公司经营态势；
- 企业像一个“数字生命体”一样由财务、经营、人力、投资、项目、风险、市场、合规、ESG 等 Twin 组成；
- AI Monitor 持续扫描监控规则，发现异常后触发大屏事件；
- 自动弹出风险预警并支持中文语音播报；
- AI 诊断按“读取数据 → 集团比较 → 结构分析 → 相似事件 → 原因图 → 方案生成”的过程动态展示；
- 自动生成 Option A/B/C，多方案进入 Monte Carlo 仿真；
- 管理人员最终选择方案，系统写入决策审计；
- 支持文字/浏览器语音指令控制大屏；
- 大屏端和操作端分离，适合集团作战室投屏。

项目核心业务逻辑使用 **Python**。浏览器端的 CSS / SVG / Canvas / JavaScript 只负责高帧率视觉呈现和语音交互，这样可以避免用 Python 定时回调逐帧驱动动画造成卡顿。

### 1. 已实现的完整功能

#### 1.1 集团企业分析

包含之前确定的全部分析能力：

- 企业总览 Dashboard；
- 全国分公司战略态势地图；
- 企业综合排行 / 营收排行 / 利润率排行 / 增长排行 / 风险排行；
- 企业画像四象限；
- 企业全景详情；
- 能力雷达图；
- 历史趋势；
- 指标分析与简单趋势预测；
- 投资结构 Sunburst 风格环图；
- NPV 排名；
- 当前企业投资项目分析。

#### 1.2 高级科幻视觉

已经包含：

- 深蓝黑作战室主题；
- HUD 边角；
- 动态背景粒子；
- 扫描纹理；
- 发光节点；
- 脉冲光环；
- 数据流虚线；
- 60 FPS SVG 路径粒子；
- AI Core 双向旋转环；
- Radar 扫描感；
- KPI 数字滚动；
- 折线逐步绘制；
- 雷达图展开；
- 企业画像 Orbit 动画；
- 战略地图节点扩散波纹；
- AI 预警全屏事件；
- Option A/B/C 顺序入场；
- Monte Carlo 数字递增与概率条收敛动画；
- Ambient Mode 自动巡航。

#### 1.3 Digital Twin 动态关系网络

数字孪生页面包含：

```text
DATA HUB
   │
   ├── FINANCE TWIN
   ├── OPERATION TWIN
   ├── HR / SENTIMENT TWIN
   ├── INVESTMENT TWIN
   ├── PROJECT TWIN
   ├── RISK TWIN
   └── MARKET TWIN
              │
              ▼
           AI CORE
              │
              ▼
       DECISION ENGINE
```

每条路径包含：

- 底层能量线；
- 动态 dash；
- SVG 粒子沿路径连续运动；
- 状态着色；
- 异常节点脉冲。

动画本身由浏览器 `requestAnimationFrame` 驱动，业务状态由 Python API 控制。

#### 1.4 AI Monitor 持续监控

默认监测：

- 财务；
- 经营；
- 人力；
- 员工状态；
- 投资；
- 项目；
- 风险；
- 市场；
- 合规；
- ESG。

监控规则支持：

- 指标字段；
- `<`、`>`、`<=`、`>=`；
- 阈值；
- NOTICE / WARNING / CRITICAL；
- 是否语音播报；
- 是否自动进入 AI 决策流程。

后台 Python `MonitorEngine` 会独立线程运行，不需要用户打开某张图才检测。

#### 1.5 AI 风险事件

检测到异常后：

```text
AI Monitor
    ↓
异常事件
    ↓
SSE 实时推送
    ↓
大屏自动弹窗
    ↓
中文 TTS 播报
    ↓
切换到 Monitor / Decision Scene
```

默认首次启动数秒后会执行第一轮扫描。为避免大屏一次被大量事件淹没，一轮扫描会在数据库保存全部新事件，但实时大屏只推送该批次中等级最高的主事件。

#### 1.6 AI 诊断过程

AI 决策页面不会直接把最终答案扔出来，而是按照视频式流程展示：

```text
01 读取企业与历史数据
02 比较集团下属企业
03 分析关联指标结构
04 检索历史相似事件
05 构建原因关系图
06 生成决策候选方案
```

每一步都会按时间线依次点亮。

当前本地诊断逻辑是真实 Python 逻辑，包含：

- 历史序列读取；
- 相关性计算；
- 集团横向比较；
- 规则分析；
- 因果图数据构建。

同时保留真实外部 AI 服务接口，可以以后替换本地规则/统计分析。

#### 1.7 Option A / B / C 决策

以利润率异常为例，默认产生：

- Option A：成本优化；
- Option B：收入增长；
- Option C：混合策略。

每个方案包含：

- 预计投入；
- 实施周期；
- 风险等级；
- 预计利润改善；
- 可执行动作；
- 仿真 NPV；
- 成功概率；
- P10 / P50 / P90。

#### 1.8 Monte Carlo 仿真

Python 实际执行 Monte Carlo，不是只在前端写死百分比。

默认：

```text
5,000 simulations / option
```

仿真考虑：

- 方案投入；
- 方案收益提升；
- 执行采用率；
- 市场扰动；
- 风险系数；
- 三年现金流；
- 8% 折现率 NPV。

前端再把已计算完成的真实结果做“收敛式动画”展示。

#### 1.9 人工最终决策与审计

用户选择方案时必须二次确认。

确认后写入 SQLite：

```text
company_id
event_id
option_id
option_name
operator
rationale
snapshot_json
created_at
```

控制台可以查看决策审计记录。

#### 1.10 语音助手

支持 Chrome / Edge 浏览器原生 Web Speech API：

- 中文语音转文字；
- 中文自动播报；
- 文字指令；
- 语音指令。

示例：

```text
分析上海分公司为什么利润下降
查看深圳分公司企业画像
进入数字孪生
查看员工情绪
模拟第二个方案
查看投资分析
```

系统会把自然语言解析成：

```text
company
metric
scene
action
option
```

然后自动控制大屏。

重要决策不会仅凭语音直接落库，仍然需要人工确认。

### 2. 大屏与控制端

项目提供两个入口。

#### 2.1 作战室大屏

```text
http://服务器IP:8050/war-room
```

适合：

- 4K 大屏；
- 超宽屏；
- 会议室投屏；
- 集团作战室。

大屏主要负责展示、场景动画和决策过程，不堆大量管理表单。

#### 2.2 操作控制台

```text
http://服务器IP:8050/control
```

适合笔记本 / 平板操作。

控制台可以：

- 切换当前分公司；
- 切换分析指标；
- 控制大屏场景；
- 触发演示预警；
- 启动 AI 诊断；
- 启动仿真；
- 立即执行 AI Monitor 扫描；
- 发送文字 / 语音命令；
- 查看监控规则；
- 新增监控规则；
- 查看决策审计记录。

控制台和大屏通过服务器状态 + SSE 实时事件同步。

### 3. 技术架构

```text
                    ┌────────────────────┐
                    │    War Room 4K     │
                    │ SVG/Canvas/CSS/JS  │
                    └─────────┬──────────┘
                              │
                              │ HTTP + SSE
                              │
        ┌─────────────────────▼──────────────────────┐
        │                 Python Server              │
        │                                            │
        │  Data Service          Scene State         │
        │  AI Engine             Event Bus           │
        │  Monitor Engine        Audit               │
        │  Simulation Engine     REST API            │
        └───────────────┬────────────────────────────┘
                        │
               ┌────────┴─────────┐
               │                  │
               ▼                  ▼
           SQLite / CSV       Remote AI API
                              （可选）
```

浏览器层只负责：

- 视觉布局；
- 60 FPS 动画；
- SVG / Canvas；
- TTS / Speech Recognition。

Python 层负责：

- 数据；
- 企业指标；
- 监控；
- AI 诊断；
- 方案；
- 仿真；
- 状态；
- 审计；
- API。

### 4. 项目目录

```text
group_ai_war_room/
├── server.py                 # Python HTTP/API/SSE 主入口
├── config.py                 # 配置
├── README.md
├── requirements.txt          # 默认运行无第三方 Python 依赖
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── start_linux.sh
├── start_windows.bat
├── .env.example
│
├── warroom/
│   ├── database.py           # SQLite 与初始化
│   ├── data_service.py       # 企业/趋势/投资/排行数据服务
│   ├── monitor.py            # AI Monitor 后台线程
│   ├── ai_engine.py          # 诊断、方案、语义指令
│   ├── simulation.py         # Monte Carlo + NPV
│   ├── bus.py                # SSE 事件总线
│   ├── state.py              # 作战室 Scene 状态
│   └── utils.py
│
├── static/
│   ├── index.html            # 4K War Room
│   ├── control.html          # 操作控制台
│   ├── css/style.css         # 科幻大屏主题
│   ├── js/
│   │   ├── warroom.js        # 场景、业务联动、Ambient Mode
│   │   ├── visuals.js        # 地图、Twin、Orbit、原因图、粒子
│   │   ├── charts.js         # Canvas 图表动画
│   │   ├── voice.js          # 语音输入、TTS
│   │   └── control.js        # 控制台逻辑
│   └── data/china.geojson    # 离线中国轮廓
│
├── data/
│   ├── companies.csv
│   ├── history.csv
│   ├── investments.csv
│   ├── monitor_rules.csv
│   └── group_twin.db         # 首次初始化生成/已随包提供
│
├── scripts/
│   ├── init_db.py
│   ├── demo_event.py
│   └── verify.py
│
└── tests/
    └── smoke_test.py
```

### 5. 最快运行

默认需要：

```text
Python 3.10+
Chrome / Edge 推荐
```

默认运行**没有必须安装的第三方 Python 包**。

#### Windows

双击：

```text
start_windows.bat
```

或者 PowerShell：

```powershell
cd group_ai_war_room
python scripts\init_db.py
python server.py
```

访问：

```text
http://127.0.0.1:8050/war-room
http://127.0.0.1:8050/control
```

#### Linux / Ubuntu

```bash
cd group_ai_war_room
chmod +x start_linux.sh
./start_linux.sh
```

或者：

```bash
python3 scripts/init_db.py
python3 server.py --host 0.0.0.0 --port 8050
```

访问：

```text
http://服务器IP:8050/war-room
http://服务器IP:8050/control
```

### 6. Docker 一键部署

```bash
cd group_ai_war_room
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
http://服务器IP:8050/war-room
```

### 7. Nginx 反向代理

项目提供 `nginx.conf` 示例。

Ubuntu：

```bash
sudo cp nginx.conf /etc/nginx/sites-available/group-ai-war-room
sudo ln -s /etc/nginx/sites-available/group-ai-war-room /etc/nginx/sites-enabled/group-ai-war-room
sudo nginx -t
sudo systemctl reload nginx
```

SSE 需要：

```nginx
proxy_buffering off;
proxy_read_timeout 3600s;
```

示例文件已经配置。

正式公网部署建议再配置 HTTPS。

### 8. 4K / 作战室部署建议

推荐：

```text
分辨率：3840×2160
缩放：100%
浏览器：Chrome / Edge
页面：/war-room
浏览器：全屏 / F11
```

超宽屏同样可以使用。

页面使用响应式比例布局，最低设计宽度为 1180px；真正作战室推荐 1920px 以上。

大屏无人操作时建议打开：

```text
AUTO / AMBIENT MODE
```

系统会每约 18 秒自动巡航：

```text
集团态势
→ 企业画像
→ Digital Twin
→ AI Monitor
→ 投资分析
→ 经营分析
```

一旦出现高等级事件：

```text
Ambient Mode
→ Event Mode
→ 异常弹窗
→ 语音播报
→ AI Monitor
→ AI Diagnosis
→ Options
→ Simulation
```

### 9. 数据说明

示例数据全部是演示数据。

#### companies.csv

包括：

```text
company_id
company_name
region
province
city
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

#### history.csv

月度历史：

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

#### investments.csv

```text
project_id
company_id
sector
subsector
project
amount
roi
npv
risk
```

#### monitor_rules.csv

```text
domain
metric
op
threshold
severity
enabled
speak
auto_decision
```

### 10. 替换成真实集团数据

最简单方式：

1. 按上述字段导出集团数据；
2. 替换 `data/*.csv`；
3. 执行：

```bash
python scripts/init_db.py --force
```

4. 重启服务。

如果你的真实数据字段不同，优先修改：

```text
warroom/data_service.py
```

而不是改 UI。

### 11. AI 接口

当前：

```text
AI_MODE=mock
```

本地模式并不是完全假数据：

- 规则检测真实执行；
- 历史相关分析真实执行；
- 原因置信度由数据计算；
- Option 根据问题类型生成；
- Monte Carlo 真实执行；
- NPV 真实计算。

“LLM / 集团内部 AI”部分使用接口预留。

配置：

```bash
export AI_MODE=remote
export AI_ENDPOINT=http://你的AI服务:9000
export AI_API_KEY=xxxx
python server.py
```

当前远端接口约定：

```text
POST {AI_ENDPOINT}/diagnose
POST {AI_ENDPOINT}/options
```

请求/返回使用 JSON。

如果你以后接 OpenAI、私有大模型、RAG 或本地模型，只需要主要修改：

```text
warroom/ai_engine.py
```

不要把 LLM 直接连生产数据库。更推荐：

```text
LLM
 ↓
受控工具 / Service API
 ↓
数据服务
 ↓
数据库
```

### 12. REST API

主要接口：

```text
GET  /api/health
GET  /api/bootstrap
GET  /api/companies
GET  /api/company/{id}
GET  /api/rankings?metric=score
GET  /api/investments
GET  /api/monitor/rules
GET  /api/events
GET  /api/decisions
GET  /api/scene
GET  /api/events/stream          SSE

POST /api/scene
POST /api/monitor/scan
POST /api/alerts/demo
POST /api/events/ack
POST /api/monitor/rules
POST /api/ai/monitor
POST /api/ai/diagnose
POST /api/ai/options
POST /api/ai/simulate
POST /api/ai/decision
POST /api/ai/chat
POST /api/decision
```

### 13. 快速体验完整 AI 决策流程

启动系统后：

1. 打开 `/war-room`；
2. 再打开 `/control`；
3. 控制台选择 `华东能源分公司`；
4. 点击 `触发 CRITICAL 演示预警`；
5. 大屏会自动出现全屏风险事件；
6. 自动中文播报；
7. 点击 `进入 AI 诊断`；
8. 观察 AI 时间线逐项运行；
9. 原因关系图逐步生长；
10. Option A/B/C 顺序出现；
11. 5,000 次 Monte Carlo 计数动画；
12. 查看成功概率和 NPV；
13. 选择一个方案；
14. 控制台查看决策审计记录。

也可以运行：

```bash
python scripts/demo_event.py
```

### 14. 学习代码的推荐顺序

如果目的是学习，不建议一上来读 `style.css`。

#### 第一步：理解 Python 数据层

先看：

```text
warroom/database.py
warroom/data_service.py
```

理解：

```text
CSV → SQLite → Python Dict → API
```

#### 第二步：理解 AI Monitor

看：

```text
warroom/monitor.py
```

重点：

```text
rule
→ scan
→ event
→ EventBus
→ SSE
```

#### 第三步：理解 AI 决策

看：

```text
warroom/ai_engine.py
warroom/simulation.py
```

重点理解：

```text
diagnose()
generate_options()
simulate_options()
full_decision()
```

#### 第四步：理解 Python API

看：

```text
server.py
```

重点关注：

```text
/api/ai/diagnose
/api/ai/decision
/api/scene
/api/events/stream
```

#### 第五步：理解大屏状态机

看：

```text
static/js/warroom.js
```

核心概念：

```text
scene
company
metric
ambient
alert
diagnosis
simulation
```

#### 第六步：理解动画

看：

```text
static/js/visuals.js
static/js/charts.js
static/css/style.css
```

其中：

- `FlowNetwork`：动态数据粒子；
- `renderStrategicMap()`：中国战略态势图；
- `renderEnterpriseOrbit()`：企业数字 DNA；
- `renderCausal()`：原因图；
- `setupMonitorWheel()`：AI Monitor 轮询动画；
- `requestAnimationFrame`：高帧率动画。

### 15. 为什么不再用 Python 每 450ms 驱动动画

之前版本的主要问题是：

```text
Python dcc.Interval
→ callback
→ 网络传输
→ DOM 更新
```

这种方式适合“每秒刷新数据”，不适合视频2那种持续流动的动画。

当前版本改成：

```text
Python
只传业务状态

Browser
requestAnimationFrame
→ 60 FPS 粒子
→ 光环
→ 流动路径
→ Timeline
→ 方案动画
```

这也是本版本科技感明显强化的核心原因。

### 16. 员工情绪指标建议

真实集团环境里，`employee_sentiment` 建议只使用：

- 匿名问卷；
- 部门 / 分公司聚合统计；
- 合法授权的员工反馈渠道；
- 趋势型汇总指标。

不建议系统对单个员工推断“情绪标签”。

### 17. 验证代码包

执行：

```bash
python scripts/verify.py
```

或：

```bash
python tests/smoke_test.py
```

当前 Smoke Test 会检查：

- 数据库；
- 企业数据；
- 历史数据；
- AI 诊断；
- 方案生成；
- Monte Carlo；
- 语义指令解析。

### 18. 正式集团生产环境还建议增加什么

当前包已经可以运行、部署、演示、二次开发。

如果真正进入大型集团生产，建议下一阶段增加：

- AD / LDAP / SSO；
- RBAC 权限；
- 集团 / 区域 / 分公司数据权限；
- PostgreSQL；
- Redis；
- 独立任务 Worker；
- 消息队列；
- 数据仓库接口；
- SAP / ERP / OA / HR / CRM 接口；
- 审计日志不可篡改存储；
- HTTPS；
- WAF；
- 备份；
- 高可用；
- 监控告警；
- AI 模型版本管理；
- RAG 知识库；
- 决策审批工作流。

不要把当前演示版的账号、权限、审计方式直接当成大型集团正式安全架构。

### 19. 配置

环境变量：

```text
HOST
PORT
GROUP_NAME
DEFAULT_COMPANY_ID
MONITOR_INTERVAL
EVENT_COOLDOWN_SECONDS
AI_MODE
AI_ENDPOINT
AI_API_KEY
```

例如：

```bash
GROUP_NAME="某某集团" PORT=9000 python server.py
```

### 20. 推荐浏览器

推荐：

```text
Chrome
Microsoft Edge
```

原因：

- SVG / Canvas 动画兼容性好；
- Speech Recognition 支持更完整；
- TTS 支持较好；
- 4K 全屏效果稳定。

Safari / Firefox 可以查看主要页面，但语音识别支持可能不同。

### 21. 交付说明

这个代码包不依赖原始视频文件，拿到代码的人只需要：

```text
Python 3.10+
浏览器
```

就可以启动默认版本。

如果用 Docker，目标机器连 Python 都不需要单独安装，只需要 Docker。

默认示例数据已包含在 `data/` 中，因此无需先准备数据库即可演示全部主要流程。


### 22. 培训与二次开发资料

为了方便直接交给其他人培训，本代码包还附带：

```text
docs/TRAINING_GUIDE.md   培训课程与练习
docs/ARCHITECTURE.md     系统架构和事件链
docs/API_REFERENCE.md    API 调用参考
docs/DEMO_SCRIPT.md      集团作战室现场演示脚本
```

建议培训人员先通读 README，再按照 `docs/TRAINING_GUIDE.md` 的半天路线学习。
