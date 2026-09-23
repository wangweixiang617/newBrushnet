## 培训指南

### 第 1 课：把项目跑起来

```bash
python3 scripts/init_db.py
python3 server.py
```

理解 `/war-room` 和 `/control` 两个入口。

### 第 2 课：理解数据库

阅读：

```text
warroom/database.py
```

重点理解：

- companies
- metric_catalog
- metric_values
- investments
- monitor_rules
- events
- decisions

### 第 3 课：理解四象限

阅读：

```text
static/js/visuals.js -> quadrant()
```

业务规则：

```text
X = operation_score
Y = basic_score
Bubble Radius = composite_score
```

### 第 4 课：企业画像

阅读：

```text
warroom/profile_service.py
static/js/visuals.js -> profileOrbit()
static/js/warroom.js -> openProfileAnalysis()
```

学习企业搜索、四画像、5年趋势、AI预测和 Honeycomb。

### 第 5 课：指标全景

阅读：

```text
warroom/data_service.py -> metric_detail()
static/js/warroom.js -> renderMetricPanorama()
```

理解统一指标模型比把每个指标写死成页面代码更适合集团系统。

### 第 6 课：投资 Sunburst

阅读：

```text
warroom/data_service.py -> investment_tree()
static/js/visuals.js -> sunburst()
```

尝试增加一个产业类别。

### 第 7 课：AI Monitor

阅读：

```text
warroom/monitor.py
```

修改一个阈值并触发预警。

### 第 8 课：AI Decision

阅读：

```text
warroom/ai_engine.py
warroom/simulation.py
```

区分：

- AI 解释逻辑
- 财务/统计计算
- 人工最终决策

### 第 9 课：控制大屏

打开两个浏览器：

```text
/war-room
/control
```

在控制端切企业、切场景、触发预警，观察 SSE 如何同步大屏。

### 培训练习

1. 新增一个企业。
2. 新增“客户满意度”指标。
3. 把客户满意度加入 AI Monitor。
4. 在指标全景中查看所有企业比较。
5. 增加一个投资项目。
6. 修改方案 C 的 Monte Carlo 参数。
7. 使用语音说“分析上海分公司利润”。
