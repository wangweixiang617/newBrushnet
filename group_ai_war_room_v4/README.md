## 集团级 AI 企业数字孪生作战指挥平台 v4

这是一个可以直接运行、演示、培训和部署的集团级 AI 数字孪生作战室项目。项目以 Python 为后台核心，前端使用原生 HTML/CSS/SVG/Canvas/JavaScript 做高帧率大屏动画，不依赖前端构建工具，也不要求 Node.js 才能运行。

本版本重点恢复并扩展了企业经营分析业务深度，同时保留 Digital Twin、AI Monitor、AI Decision、语音助手和 War Room 大屏能力。

## 本版本包含的完整功能

### 集团态势

- 集团经营 KPI
- 全国企业战略态势图
- 分公司风险状态
- 集团 5 年营业收入趋势
- AI 实时事件流
- 分公司风险排行

### 经营分析

- 5 个固定排名维度：综合评分、基础得分、经营得分、科技得分、监管得分
- 四象限严格使用：X=经营得分，Y=基础得分
- 气泡大小由综合评分驱动，不是固定小点
- 四象限气泡带渐变、辉光、当前企业扩散环
- 气泡可点击，点击后自动切换企业并进入企业画像
- 右侧指标分析支持营业收入、净利润、ROE、现金流、研发强度、监管评分、员工状态等

### 企业画像

- 顶部 CURRENT 企业可搜索、可切换
- 企业 Digital DNA 由四类画像构成：基础画像、经营画像、科技画像、监管画像
- 四类画像颜色独立、节点可点击、数据粒子持续流动
- 点击画像后打开完整深度分析弹窗
- 每类画像 4 个核心指标
- 每个指标支持：当前值、集团排名、集团均值、同比、5 年历史趋势、AI 预测、预测置信度
- 全集团 Honeycomb 蜂巢分布，六边形代表企业
- Honeycomb 企业可点击并切换分析对象

默认画像指标：

| 画像 | 指标 |
|---|---|
| 基础画像 | 总资产、注册资本、净资产、员工规模 |
| 经营画像 | 营业收入、净利润、净资产、ROE |
| 科技画像 | 研发投入、研发强度、有效专利、研发人员占比 |
| 监管画像 | 监管合规评分、资产负债率、审计异常、风险事件 |

### 指标全景

独立 `08 指标全景` 页面，解决旧版本只有一条趋势线、缺少完整指标分析的问题。

- 指标体系树
- 指标搜索
- 当前企业指标详情
- 5 年历史数据
- AI 趋势预测
- 预测置信度
- 集团均值、排名、同比、分位数
- 所有企业横向比较
- 全集团 Honeycomb 分布
- 点击企业后同步切换整个系统当前企业

### 投资分析

- 左侧企业列表，可搜索和切换企业
- 中央大型交互式 Sunburst
- Sunburst 可点击、可下钻、可返回上一级
- 圆弧带辉光和层级动画感
- 右侧当前企业投资 KPI、项目柱图和项目列表
- 取消旧版本中间无必要的 NPV 排名区域
- 修复右侧 KPI 与图表重叠问题

### Digital Twin

- 六大 Twin 节点
- AI TWIN CORE
- SVG 流动路径
- 数据粒子沿路径持续移动
- 节点健康状态
- CRITICAL 节点脉冲扩散
- 实时数据吞吐和 AI 延迟展示

### AI Monitor

- 持续扫描视觉
- 基础、经营、科技、监管、投资、风险域
- 扫描进度
- 自定义监控规则
- 事件中心
- 演示预警
- SSE 实时推送
- 异常后自动弹窗和语音播报

### AI Decision

完整链路：

`AI检测 -> 诊断步骤 -> 因果图 -> Option A/B/C -> Monte Carlo -> 管理决策 -> 审计记录`

包含：

- AI CORE 多层旋转环
- 数据粒子进入 AI CORE
- 分阶段诊断 Timeline
- 动态因果图
- Option A/B/C 顺序生成
- 推荐方案高亮
- 5000 次 Monte Carlo 动画
- 成功概率、NPV、风险、投入、周期
- 人工最终确认
- 决策记录写入 SQLite

### AI Assistant

旧版本 AI 助手固定在右下角并遮挡页面。本版本改为：

- 默认只显示底部小型 `AI READY` 按钮
- 点击才从右侧展开 Drawer
- 默认绝不覆盖主业务内容
- 支持文字自然语言指令
- Chrome 支持语音识别
- 支持浏览器中文 TTS 自动播报

## 运行环境

推荐：

- Python 3.10+
- Chrome / Edge 最新版本
- macOS、Windows、Linux 均可
- 1920×1080 可以使用
- 集团作战室推荐 3840×2160 4K 或 3840×1080 超宽屏

项目默认 Python 后台只使用标准库，无需安装 Flask/Django/Dash 等第三方包。

## macOS / Linux 第一次运行

解压后进入项目目录：

```bash
cd group_ai_war_room_v4
```

初始化数据库：

```bash
python3 scripts/init_db.py
```

启动：

```bash
python3 server.py
```

浏览器打开：

```text
http://127.0.0.1:8050/war-room
```

控制台：

```text
http://127.0.0.1:8050/control
```

也可以直接：

```bash
./start_mac_linux.sh
```

`start_mac_linux.sh` 会检查并初始化数据库，但不会覆盖已经存在的数据。如需重置演示数据，执行 `python3 scripts/init_db.py --reset`。

## Windows 第一次运行

双击：

```text
start_windows.bat
```

或命令行：

```bat
python scripts\init_db.py
python server.py
```

访问：

```text
http://127.0.0.1:8050/war-room
```

## Docker 部署

```bash
docker compose up -d --build
```

访问：

```text
http://服务器IP:8050/war-room
```

控制台：

```text
http://服务器IP:8050/control
```

查看日志：

```bash
docker compose logs -f
```

停止：

```bash
docker compose down
```

## 局域网投屏

服务器或 Mac 启动：

```bash
python3 server.py --host 0.0.0.0 --port 8050
```

假设电脑局域网 IP 为 `192.168.1.25`，大屏打开：

```text
http://192.168.1.25:8050/war-room
```

操作人员电脑/平板打开：

```text
http://192.168.1.25:8050/control
```

建议作战室结构：

```text
控制笔记本 / 平板
       |
       | HTTP + SSE
       v
Python War Room Server
       |
       +--------> 4K 大屏 /war-room
       +--------> 控制端 /control
```

## 项目结构

```text
group_ai_war_room_v4/
├── server.py                  # HTTP API、SSE、静态资源服务
├── config.py                  # 系统配置
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── start_mac_linux.sh
├── start_windows.bat
│
├── warroom/
│   ├── database.py            # SQLite Schema + 5年模拟数据生成
│   ├── data_service.py        # 企业、指标、排名、投资数据服务
│   ├── profile_service.py     # 四画像服务
│   ├── ai_engine.py           # AI诊断、预测、方案、语义指令
│   ├── simulation.py          # Monte Carlo
│   ├── monitor.py             # AI后台规则检测
│   ├── bus.py                 # SSE事件总线
│   ├── state.py               # War Room场景状态
│   └── utils.py
│
├── static/
│   ├── index.html             # 作战大屏
│   ├── control.html           # 控制端
│   ├── css/style.css          # 全局科幻UI
│   └── js/
│       ├── charts.js          # Canvas折线/雷达/柱图
│       ├── visuals.js         # Bubble/Honeycomb/Sunburst/Twin/因果图
│       ├── voice.js           # STT/TTS
│       ├── warroom.js         # 大屏场景与交互
│       └── control.js         # 控制台
│
├── data/
│   ├── companies.csv          # 企业基础示例
│   ├── metric_catalog.csv     # 指标定义示例
│   └── group_twin.db          # init_db 后生成
│
├── scripts/
│   ├── init_db.py
│   ├── verify.py
│   └── demo_event.py
│
├── tests/
│   └── smoke_test.py
│
└── docs/
    ├── ARCHITECTURE.md
    ├── TRAINING_GUIDE.md
    ├── API_REFERENCE.md
    └── DEMO_SCRIPT.md
```

## 数据模型

### companies

保存集团企业与四画像分数：

```text
basic_score
operation_score
technology_score
supervision_score
composite_score
risk_score
```

四象限严格使用：

```text
X = operation_score
Y = basic_score
Bubble Size = composite_score
```

综合评分默认：

```text
20% 基础 + 35% 经营 + 25% 科技 + 20% 监管
```

可在 `config.py` 修改 `PROFILE_WEIGHTS`。

### metric_catalog

指标不是写死成数据库列，而使用指标目录：

```text
metric_key
metric_name
profile
unit
direction
warning_threshold
critical_threshold
forecastable
```

后续增加毛利率、客户流失率、员工情绪、项目延期等指标时，不需要重新设计主要页面。

### metric_values

统一时序表：

```text
company_id
metric_key
period
value
```

演示数据默认：

```text
18 家企业 × 18 个指标 × 60 个月
```

即完整 5 年数据。

## 如何换成真实集团数据

推荐保持前端 API 不变，仅替换 Python 数据层。

第一步可以把：

```text
warroom/data_service.py
```

中的 SQLite 查询替换为企业数据仓库查询。

真实系统常见数据源：

```text
SAP
Oracle
ERP
用友
金蝶
HR
CRM
OA
PostgreSQL
SQL Server
集团数据湖
Excel / CSV
```

前端不需要跟随数据库重写。

## 如何接真实 AI

默认：

```text
AI_MODE=mock
```

表示 AI 诊断和方案由本地 Python 逻辑完成，适合离线演示和培训。

环境变量：

```bash
export AI_MODE=remote
export AI_ENDPOINT=http://your-ai-server:9000
export AI_API_KEY=your-key
```

`warroom/ai_engine.py` 已保留远程 AI 调用入口。

生产系统建议明确划分：

```text
LLM / AI：解释、归纳、策略草案、自然语言交互
Python模型：评分、财务计算、预测、Monte Carlo、NPV、规则检测
数据库：事实数据
管理人员：最终决策
```

不要让 LLM 自行编造经营数值。

## 代码学习顺序

建议培训时按照：

1. `config.py`
2. `warroom/database.py`
3. `warroom/data_service.py`
4. `server.py`
5. `static/index.html`
6. `static/js/warroom.js`
7. `static/js/visuals.js`
8. `static/js/charts.js`
9. `warroom/ai_engine.py`
10. `warroom/simulation.py`
11. `warroom/monitor.py`

详细培训计划见 `docs/TRAINING_GUIDE.md`。

## 验证代码包

服务层验证：

```bash
python3 scripts/verify.py
```

完整 HTTP Smoke Test：

```bash
python3 tests/smoke_test.py
```

正常应显示：

```text
SMOKE TEST PASS
```

## 生产上线前必须增加的能力

本项目已经可直接部署、演示、培训和二次开发，但大型集团正式生产上线前仍应结合企业环境增加：

- SSO / LDAP / OAuth2
- RBAC 权限
- 集团 / 区域 / 分公司数据权限
- HTTPS
- WAF / API 鉴权
- PostgreSQL / Oracle 等正式数据库
- Redis / 消息队列
- 后台任务调度
- 完整审计日志
- 数据脱敏
- AI 调用审计
- 模型版本管理
- 灾备与数据库备份
- 高可用部署
- 4K 大屏设备实测

这些属于生产治理，不应该通过演示代码假装已经完成。

## 推荐演示流程

1. 打开 `01 集团态势`
2. 进入 `02 经营分析`，切换五维排名
3. 点击四象限中的某企业 Bubble，自动进入企业画像
4. 点击经营画像
5. 展示营业收入 / 净利润 / 净资产 / ROE
6. 开启 AI 预测
7. 点击 Honeycomb 中其他企业切换分析对象
8. 打开 `08 指标全景`，展示全集团横向比较
9. 打开 `04 投资分析`，点击 Sunburst 下钻
10. 打开 `05 数字孪生`
11. 在 `06 AI监控` 触发演示预警
12. 自动播报后进入 `07 AI决策`
13. 展示因果图、A/B/C 方案、Monte Carlo
14. 管理人员选择方案并写入决策记录

完整讲解稿见 `docs/DEMO_SCRIPT.md`。
