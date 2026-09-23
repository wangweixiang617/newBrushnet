## 培训指南

本指南用于把整个项目直接交给新的开发人员、实施人员或演示人员。默认培训目标不是要求学员一次理解全部前端动画，而是先建立“业务状态由 Python 产生，浏览器负责高帧率表现”的整体认识。

### 1. 培训目标

完成培训后，学员应能够：

- 在 Windows / Linux / Docker 中启动系统；
- 理解集团总览、企业分析、企业画像、投资分析、Digital Twin、AI Monitor、AI Decision 七个主场景；
- 修改集团名称、企业样例数据和监控阈值；
- 触发一个 AI 预警并完整走完诊断、方案生成、Monte Carlo 仿真和决策确认；
- 理解 Python API、SSE 事件流和浏览器 Scene Engine 的关系；
- 知道未来如何把 Mock / 本地 AI 替换成真实 AI 服务；
- 知道生产环境需要额外增加认证、权限、审计、高可用和数据安全能力。

### 2. 30 分钟演示培训流程

#### 2.1 启动

Linux / macOS：

```bash
cd group_ai_war_room
./start_linux.sh
```

Windows：

```bat
start_windows.bat
```

访问：

```text
大屏：http://127.0.0.1:8050/war-room
控制台：http://127.0.0.1:8050/control
```

#### 2.2 演示集团态势

进入“01 集团态势”，讲解：

- 左侧集团生命体征；
- 中部全国战略态势图；
- 分公司状态与风险波纹；
- 底部集团经营趋势；
- 右侧 AI Intelligence / Event Stream；
- AUTO 模式下场景自动巡航。

#### 2.3 演示经营分析

进入“02 经营分析”：

- 切换综合、营收、利润率、增长、风险排行；
- 点击分公司切换当前分析对象；
- 观察企业画像四象限；
- 切换指标并观察趋势和预测。

#### 2.4 演示企业数字画像

进入“03 企业画像”：

- 中央 Enterprise Digital DNA；
- 财务、经营、创新、投资、人力、市场、合规、ESG Orbit Node；
- 能力雷达展开；
- 利润率、员工状态、现金流历史趋势。

重点说明：这里不是单纯画雷达图，而是把企业看成持续变化的数字生命体。

#### 2.5 演示 Digital Twin

进入“05 数字孪生”：

- DATA HUB；
- FINANCE / OPERATION / HR / INVESTMENT / PROJECT / RISK / MARKET Twin；
- 数据粒子沿 SVG Path 持续流入 AI CORE；
- AI CORE 再流向 DECISION ENGINE；
- 异常域会进入 WARNING / CRITICAL 状态并出现脉冲。

#### 2.6 演示 AI Monitor

进入“06 AI监控”，点击“触发演示预警”。

讲解完整事件链：

```text
Monitor Rule
→ Python MonitorEngine
→ Event
→ SSE
→ War Room Alert
→ TTS
→ Scene Switch
```

#### 2.7 演示 AI Decision

进入“07 AI决策”，观察：

```text
读取历史数据
→ 集团横向比较
→ 指标结构分析
→ 相似事件检索
→ 原因图
→ Option A/B/C
→ Monte Carlo
→ Decision Matrix
```

最终选择方案时会二次确认并写入决策审计表。

### 3. 半天开发培训路线

#### 第一小时：数据与数据库

阅读：

```text
warroom/database.py
warroom/data_service.py
data/*.csv
```

练习：

1. 新增一家分公司；
2. 修改利润率；
3. 重置数据库；
4. 在大屏观察变化。

#### 第二小时：AI Monitor

阅读：

```text
warroom/monitor.py
data/monitor_rules.csv
```

练习：新增一条现金流监控规则，并确认能生成事件。

#### 第三小时：AI 决策

阅读：

```text
warroom/ai_engine.py
warroom/simulation.py
```

练习：给利润率异常增加一个新的原因项，并调整 Option C 的预算和预期改善幅度。

#### 第四小时：浏览器场景和动画

阅读：

```text
static/js/warroom.js
static/js/visuals.js
static/js/charts.js
static/css/style.css
```

重点理解：

- Python 不逐帧控制动画；
- `requestAnimationFrame` 控制高帧率粒子；
- Scene State 决定当前业务场景；
- SSE 负责把后台事件实时推向大屏。

### 4. 建议的二次开发练习

从易到难：

1. 修改集团名称和主题文案；
2. 新增监测指标；
3. 新增企业维度；
4. 增加一个 AI 诊断原因节点；
5. 增加一个新的决策方案；
6. 接入真实 PostgreSQL；
7. 接入真实 AI 服务；
8. 接入企业 SSO / LDAP；
9. 增加审批流和不可篡改审计；
10. 增加多屏联动和 4K 超宽屏布局。

### 5. 培训时应明确的边界

当前包适用于：

- 演示；
- 培训；
- PoC；
- 二次开发；
- 内部原型部署。

真正进入大型集团生产前，还需要完成：

- 认证与 RBAC；
- 分公司数据权限；
- 数据脱敏；
- HTTPS 和 WAF；
- 高可用数据库；
- Redis / MQ / Worker；
- 统一日志、指标监控和告警；
- AI 模型治理；
- 审批工作流；
- 安全测试和等保相关工作。
