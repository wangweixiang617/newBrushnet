## 系统架构

### 总体结构

```text
4K War Room /war-room             Control Console /control
          |                                |
          +---------------+----------------+
                          |
                    Python HTTP Server
                          |
          +---------------+----------------+
          |               |                |
      REST API           SSE            Static UI
          |               |
          v               v
    Data Service      Event Bus
          |
   +------+------+----------------+
   |             |                 |
SQLite       AI Engine        Monitor Engine
   |             |                 |
5年指标      Diagnose          Rule Scan
Investment   Forecast          Alert
Ranking      Options           TTS event
             Monte Carlo
```

### 为什么动画不放在 Python 循环里

Python 负责真实状态，不负责每一帧。

```text
Python -> 发送业务状态
Browser -> requestAnimationFrame / SVG animate / CSS animation
```

这样数据流、AI Core、Sunburst、Bubble、Honeycomb 可以保持流畅，同时避免旧版本 `dcc.Interval` 式低帧率闪动。

### 企业画像数据模型

```text
企业
├── 基础画像
│   ├── 总资产
│   ├── 注册资本
│   ├── 净资产
│   └── 员工规模
├── 经营画像
│   ├── 营业收入
│   ├── 净利润
│   ├── 净资产
│   └── ROE
├── 科技画像
│   ├── 研发投入
│   ├── 研发强度
│   ├── 有效专利
│   └── 研发人员占比
└── 监管画像
    ├── 监管合规评分
    ├── 资产负债率
    ├── 审计异常
    └── 风险事件
```

### AI 决策闭环

```text
监控规则
  -> Alert
  -> TTS
  -> AI Diagnosis
  -> Causal Graph
  -> Option A/B/C
  -> Monte Carlo
  -> Human Confirm
  -> Decision Audit
```

### 真实系统扩展建议

生产环境可将 Python HTTP 层替换为 FastAPI，SQLite 替换 PostgreSQL/Oracle，事件总线替换 Redis/Kafka，后台定时器替换 Celery/APS Scheduler。前端 API 契约可以保持不变。
