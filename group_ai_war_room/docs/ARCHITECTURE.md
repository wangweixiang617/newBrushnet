## 系统架构

### 1. 总体架构

```text
                 ┌───────────────────────────┐
                 │      War Room Screen      │
                 │   SVG / Canvas / CSS / JS │
                 └─────────────┬─────────────┘
                               │ HTTP + SSE
                 ┌─────────────▼─────────────┐
                 │        Python Server      │
                 │     API / Scene State     │
                 └───────┬─────────┬─────────┘
                         │         │
              ┌──────────▼───┐ ┌──▼──────────────┐
              │ MonitorEngine│ │ AI Decision     │
              │ Rule / Event │ │ Diagnose/Options│
              └──────┬───────┘ └──────┬──────────┘
                     │                │
                     └────────┬───────┘
                              ▼
                      ┌──────────────┐
                      │ SQLite / CSV │
                      └──────────────┘
```

### 2. 为什么动画不由 Python 逐帧驱动

Python 负责业务状态：

```text
company
metric
scene
alert
diagnosis
options
simulation
```

浏览器负责视觉状态：

```text
particles
pulse
glow
orbit
scan
chart animation
scene transition
```

这样做的优势：

- 动画可以接近浏览器刷新率；
- 不需要每 16 ms 请求 Python；
- 后端更稳定；
- 大屏端即使持续展示数小时也不会产生大量无意义 API 调用。

### 3. 核心模块

```text
server.py                  HTTP API / SSE / 静态文件
warroom/database.py        SQLite 初始化和数据操作
warroom/data_service.py    企业、历史、投资、排行读取
warroom/monitor.py         后台监控规则和异常事件
warroom/ai_engine.py       诊断、方案、语义命令
warroom/simulation.py      Monte Carlo + NPV
warroom/state.py           大屏 Scene State
warroom/bus.py             内存事件总线
static/js/warroom.js       主场景控制器
static/js/visuals.js       SVG / Canvas 动态视觉
static/js/charts.js        Canvas 图表
static/js/voice.js         STT / TTS
static/js/control.js       独立控制台
```

### 4. 事件链

```text
数据库指标
   ↓
MonitorEngine.scan()
   ↓
命中 monitor_rule
   ↓
写入 events
   ↓
BUS.publish()
   ↓
/api/events/stream (SSE)
   ↓
大屏收到 alert
   ↓
弹窗 / 播报 / 场景切换
```

### 5. AI 决策链

```text
异常事件
  ↓
diagnose()
  ↓
历史趋势 + 集团比较 + 相关分析
  ↓
原因图
  ↓
generate_options()
  ↓
simulate_options()
  ↓
Monte Carlo 5000 次 / Option
  ↓
NPV / P10 / P50 / P90 / 成功概率
  ↓
人工确认
  ↓
decisions + audit_log
```

### 6. 生产升级推荐

PoC 当前结构：

```text
Python ThreadingHTTPServer + SQLite + in-memory EventBus
```

大型集团生产建议替换为：

```text
Nginx
  ↓
FastAPI / Gunicorn / Uvicorn
  ↓
PostgreSQL
  ↓
Redis / Kafka / RabbitMQ
  ↓
Celery / Worker
```

SSE 可以保留，也可以升级为 WebSocket。
