## API 参考

默认地址：`http://127.0.0.1:8050`

### 1. 系统

```text
GET /api/health
GET /api/bootstrap?company_id=1001
GET /api/scene
POST /api/scene
```

`POST /api/scene` 示例：

```json
{
  "scene": "twin",
  "company_id": 1001,
  "metric": "profit_margin",
  "ambient": false
}
```

### 2. 企业与分析

```text
GET /api/companies
GET /api/company/{company_id}
GET /api/rankings?metric=score
GET /api/investments?company_id=1001
```

### 3. AI Monitor

```text
GET  /api/monitor/rules
POST /api/monitor/rules
POST /api/monitor/scan
POST /api/alerts/demo
GET  /api/events
POST /api/events/ack
GET  /api/events/stream
```

手工触发演示告警：

```bash
curl -X POST http://127.0.0.1:8050/api/alerts/demo \
  -H "Content-Type: application/json" \
  -d '{"company_id":1001,"metric":"profit_margin","severity":"CRITICAL"}'
```

### 4. AI 决策

```text
POST /api/ai/monitor
POST /api/ai/diagnose
POST /api/ai/options
POST /api/ai/simulate
POST /api/ai/decision
POST /api/ai/chat
```

完整决策：

```bash
curl -X POST http://127.0.0.1:8050/api/ai/decision \
  -H "Content-Type: application/json" \
  -d '{"company_id":1001,"metric":"profit_margin","runs":5000}'
```

### 5. 决策审计

```text
POST /api/decision
GET  /api/decisions
```

### 6. 接入外部 AI

环境变量：

```text
AI_MODE=remote
AI_ENDPOINT=http://your-ai-service:9000
AI_API_KEY=xxx
```

当前 `warroom/ai_engine.py` 会调用：

```text
POST {AI_ENDPOINT}/diagnose
POST {AI_ENDPOINT}/options
```

外部服务只要保持返回结构兼容，大屏和其他业务代码无需修改。
