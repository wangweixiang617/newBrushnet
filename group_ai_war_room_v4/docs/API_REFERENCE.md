## API Reference

### Health

```text
GET /api/health
```

### Bootstrap

```text
GET /api/bootstrap
```

返回集团概要、企业、5维排行、指标目录、当前场景、规则、事件和当前投资结构。

### 企业

```text
GET /api/companies?q=上海
GET /api/company/1
GET /api/rankings?metric=operation_score
```

排名 metric 允许：

```text
composite_score
basic_score
operation_score
technology_score
supervision_score
```

### 指标

```text
GET /api/metrics/catalog
GET /api/metrics/catalog?profile=operation
GET /api/metric/roe?company_id=1
GET /api/metric/roe/compare
```

### 企业画像

```text
GET /api/profile/1?profile=operation
```

profile：

```text
basic
operation
technology
supervision
```

### 投资

```text
GET /api/investments/tree?company_id=1
GET /api/investments?company_id=1
```

### AI Monitor

```text
GET  /api/monitor/rules
POST /api/monitor/scan
POST /api/alerts/demo
POST /api/events/ack
```

### AI

```text
POST /api/ai/diagnose
POST /api/ai/options
POST /api/ai/simulate
POST /api/ai/decision
POST /api/ai/chat
```

AI Decision 示例：

```json
{
  "company_id": 1,
  "metric": "roe",
  "runs": 5000
}
```

### 场景控制

```text
GET  /api/scene
POST /api/scene
```

示例：

```json
{
  "scene": "profile",
  "company_id": 3,
  "ambient": false
}
```

场景：

```text
overview
analytics
profile
investment
twin
monitor
decision
metrics
```

### 实时事件

```text
GET /api/events/stream
```

使用 Server-Sent Events。
