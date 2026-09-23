from flask import jsonify, request
from dash import Dash, html, dcc, Input, Output, State, ctx, no_update, ALL
from dash.exceptions import PreventUpdate
from apscheduler.schedulers.background import BackgroundScheduler
import plotly.graph_objects as go

from config import APP_HOST, APP_PORT, DEBUG, DEFAULT_COMPANY_ID, ENABLE_BACKGROUND_MONITOR, MONITOR_INTERVAL_SECONDS
from services import DataService, AIGateway, DecisionEngine, EventStore, MonitorEngine
from services.analytics import METRIC_LABELS
from ui.components import NAV_ITEMS, nav_button, status_badge
from ui.pages import overview_page, ranking_page, portrait_page, profile_page, investment_page, twin_page, monitor_page, decision_page, base_fig, CYAN, ORANGE

# Core services
ds=DataService()
ai=AIGateway()
decision_engine=DecisionEngine()
events=EventStore()
monitor=MonitorEngine(ds,ai,events)
monitor.run_all()

scheduler=None
if ENABLE_BACKGROUND_MONITOR:
    scheduler=BackgroundScheduler(daemon=True)
    scheduler.add_job(monitor.run_all,'interval',seconds=MONITOR_INTERVAL_SECONDS,id='ai-monitor',replace_existing=True,max_instances=1)
    scheduler.start()

app=Dash(__name__,suppress_callback_exceptions=True,title='Group AI Digital Twin',update_title=None)
server=app.server

company_options=[{'label':f"{r.company_name} · {r.city}",'value':int(r.company_id)} for _,r in ds.companies.iterrows()]

app.layout=html.Div([
    dcc.Store(id='page-store',data='overview'),dcc.Store(id='company-store',data=DEFAULT_COMPANY_ID),dcc.Store(id='refresh-store',data=0),
    dcc.Store(id='last-event-store',data=0),dcc.Store(id='active-alert-store'),dcc.Store(id='decision-event-store'),dcc.Store(id='diagnosis-store'),dcc.Store(id='options-store'),
    dcc.Interval(id='event-poll',interval=3500,n_intervals=0),
    html.Div(id='tts-text',style={'display':'none'}),html.Div(id='assistant-tts-text',style={'display':'none'}),
    html.Div([
        html.Div([
            html.Div([html.Div('◈',className='brand-glyph'),html.Div([html.Div('GROUP AI',className='brand-main'),html.Div('DIGITAL TWIN',className='brand-sub')])],className='brand'),
            html.Div([nav_button(*x) for x in NAV_ITEMS],className='nav-list'),
            html.Div([html.Div('SYSTEM',className='mini-label'),html.Div([html.Span(className='live-dot'),html.Span('AI MONITOR ONLINE')],className='system-live'),html.Div('Python · Dash · Plotly',className='system-foot')],className='sidebar-foot')
        ],className='sidebar'),
        html.Div([
            html.Div([
                html.Div([html.Div('GROUP INTELLIGENCE COMMAND CENTER',className='top-eyebrow'),html.Div(id='page-title',className='top-title')]),
                html.Div([status_badge('LIVE','normal'),dcc.Dropdown(id='company-select',options=company_options,value=DEFAULT_COMPANY_ID,clearable=False,className='company-select')],className='top-actions')
            ],className='topbar'),
            html.Main(id='page-content',className='main-content')
        ],className='workspace')
    ],className='app-shell'),
    html.Div(id='alert-area'),
    html.Div([
        html.Button('◉',id='voice-btn',className='voice-orb',title='语音输入'),
        html.Div([
            html.Div([html.Span('AI ASSISTANT'),html.Span('VOICE READY',className='assistant-state')],className='assistant-head'),
            html.Div(id='assistant-response',children='可以输入：分析上海分公司利润异常 / 查看企业画像 / 模拟投资方案。',className='assistant-response'),
            html.Div([dcc.Input(id='assistant-query',placeholder='输入问题或点击语音按钮…',className='assistant-input',debounce=False),html.Button('发送',id='assistant-send',className='assistant-send')],className='assistant-input-row')
        ],className='assistant-box')
    ],className='assistant-wrap')
],className='app-root')

@server.get('/api/health')
def api_health():
    return jsonify({'status':'ok','service':'group-ai-digital-twin','events':len(events.list(300))})

@server.post('/api/ai/monitor')
def api_monitor():
    return jsonify(ai.monitor(request.get_json(silent=True) or {}))

@server.post('/api/ai/diagnose')
def api_diagnose():
    payload=request.get_json(silent=True) or {};company=ds.company(payload.get('company_id',DEFAULT_COMPANY_ID))
    return jsonify(ai.diagnose(company,payload.get('event',{}),payload.get('history',[])))

@server.post('/api/ai/options')
def api_options():
    payload=request.get_json(silent=True) or {};company=ds.company(payload.get('company_id',DEFAULT_COMPANY_ID))
    return jsonify({'options':decision_engine.generate_options(company,payload.get('metric','profit_margin'))})

@server.post('/api/ai/chat')
def api_chat():
    payload=request.get_json(silent=True) or {};return jsonify(ai.chat(payload.get('query',''),ds.companies))

@app.callback(Output('page-store','data'),*[Input(f'nav-{x[0]}','n_clicks') for x in NAV_ITEMS],prevent_initial_call=True)
def navigate(*_):
    if not ctx.triggered_id:raise PreventUpdate
    return ctx.triggered_id.replace('nav-','')

@app.callback(Output('company-store','data'),Input('company-select','value'))
def select_company(value):
    return int(value) if value is not None else DEFAULT_COMPANY_ID

@app.callback(Output('page-content','children'),Output('page-title','children'),Input('page-store','data'),Input('company-store','data'),Input('refresh-store','data'),State('decision-event-store','data'),State('diagnosis-store','data'),State('options-store','data'))
def render_page(page,company_id,_refresh,event,diagnosis,options):
    titles={'overview':'集团经营总览','ranking':'企业综合排行','portrait':'企业画像四象限','profile':'企业全景与指标分析','investment':'集团投资分析','twin':'Digital Twin 动态数据流','monitor':'AI 监测中心','decision':'AI 诊断与决策中心'}
    if page=='ranking':body=ranking_page(ds)
    elif page=='portrait':body=portrait_page(ds,company_id)
    elif page=='profile':body=profile_page(ds,company_id)
    elif page=='investment':body=investment_page(ds,company_id)
    elif page=='twin':body=twin_page(ds,company_id)
    elif page=='monitor':body=monitor_page(ds,events.list())
    elif page=='decision':body=decision_page(ds,company_id,event or {},diagnosis or {},options or [])
    else:body=overview_page(ds,company_id,events.list())
    return body,titles.get(page,'集团经营总览')

@app.callback(Output('metric-detail-graph','figure'),Input('metric-dropdown','value'),State('company-store','data'),prevent_initial_call=False)
def metric_detail(metric,company_id):
    hist=ds.company_history(company_id);fig=go.Figure()
    if metric in hist.columns:
        fig.add_trace(go.Scatter(x=hist.month,y=hist[metric],mode='lines+markers',line={'color':CYAN,'width':3},marker={'size':5},name=METRIC_LABELS.get(metric,metric)))
        forecast=[]
        from services.analytics import linear_forecast
        forecast=linear_forecast(hist,metric,4)
        if forecast:fig.add_trace(go.Scatter(x=[x['month'] for x in forecast],y=[x['value'] for x in forecast],line={'color':ORANGE,'width':2,'dash':'dot'},name='趋势预测'))
    return base_fig(fig)

@app.callback(Output('twin-cytoscape','stylesheet'),Input('twin-animation','n_intervals'),State('twin-cytoscape','stylesheet'),prevent_initial_call=True)
def animate_twin(n,stylesheet):
    if not stylesheet:raise PreventUpdate
    clean=[x for x in stylesheet if x.get('selector')!='edge[active = 1]']
    clean.append({'selector':'edge[active = 1]','style':{'line-dash-offset':int((n*4)%32),'line-color':'#37e7ff','target-arrow-color':'#37e7ff'}})
    return clean

@app.callback(Output('twin-node-detail','children'),Input('twin-cytoscape','tapNodeData'),prevent_initial_call=True)
def twin_node_detail(data):
    if not data:raise PreventUpdate
    score=data.get('score')
    return html.Div([html.Div(data.get('label',''),className='node-title'),html.Div(f"当前分值：{score}" if score is not None else '核心流程节点',className='node-score'),html.Div(f"状态：{data.get('state','ACTIVE').upper()}",className='code-line')])

@app.callback(Output('monitor-action-msg','children'),Output('refresh-store','data'),Input('add-rule-btn','n_clicks'),Input('run-monitor-btn','n_clicks'),Input('demo-alert-btn','n_clicks'),State('rule-domain','value'),State('rule-metric','value'),State('rule-op','value'),State('rule-threshold','value'),State('rule-severity','value'),State('rule-flags','value'),State('company-store','data'),State('refresh-store','data'),prevent_initial_call=True)
def monitor_actions(_add,_run,_demo,domain,metric,op,threshold,severity,flags,company_id,refresh):
    trigger=ctx.triggered_id
    if trigger=='add-rule-btn':
        if threshold is None:return '阈值不能为空。',refresh
        ds.save_rule({'domain':domain,'metric':metric,'op':op,'threshold':float(threshold),'severity':severity,'enabled':1,'speak':int('speak' in (flags or [])),'auto_decision':int('auto' in (flags or []))})
        return f'已新增监测规则：{METRIC_LABELS.get(metric,metric)} {op} {threshold}',refresh+1
    if trigger=='run-monitor-btn':
        found=monitor.run_all();return f'检测完成：当前规则命中 {len(found)} 次（重复事件会自动去重）。',refresh+1
    if trigger=='demo-alert-btn':
        monitor.manual_event(company_id);return '已触发一条演示级 AI 高风险预警。',refresh+1
    raise PreventUpdate

@app.callback(Output('alert-area','children'),Output('tts-text','children'),Output('last-event-store','data'),Output('active-alert-store','data'),Input('event-poll','n_intervals'),State('last-event-store','data'))
def poll_events(_n,last_id):
    event=events.latest()
    if not event or int(event['id'])<=int(last_id or 0):return no_update,no_update,last_id,no_update
    modal=html.Div([
        html.Div([
            html.Div([html.Div('AI RISK ALERT',className='alert-title'),html.Div(event['severity'],className=f"severity severity-{event['severity'].lower()}")],className='alert-head'),
            html.Div(event['company_name'],className='alert-company'),html.Div(event['title'],className='alert-subtitle'),html.Div(event['reason'],className='alert-reason'),
            html.Div([html.Button('进入AI分析',id='analyze-alert-btn',className='primary-btn'),html.Button('稍后处理',id='dismiss-alert-btn',className='ghost-btn')],className='action-row')
        ],className='alert-modal')
    ],className='alert-overlay')
    speech=f"AI风险预警。{event['company_name']}。{event['reason']}。建议进入决策分析。" if event.get('speak') else ''
    return modal,speech,event['id'],event

@app.callback(Output('alert-area','children',allow_duplicate=True),Output('page-store','data',allow_duplicate=True),Output('company-select','value',allow_duplicate=True),Output('decision-event-store','data'),Output('diagnosis-store','data'),Output('options-store','data'),Input('analyze-alert-btn','n_clicks'),Input('dismiss-alert-btn','n_clicks'),State('active-alert-store','data'),prevent_initial_call=True)
def handle_alert(analyze,dismiss,event):
    if not event:raise PreventUpdate
    if ctx.triggered_id=='dismiss-alert-btn':return None,no_update,no_update,no_update,no_update,no_update
    company_id=int(event['company_id']);company=ds.company(company_id)
    diagnosis=ai.diagnose(company,event,ds.metric_history(company_id,event.get('metric','profit_margin')))
    options=decision_engine.generate_options(company,event.get('metric','profit_margin'))
    return None,'decision',company_id,event,diagnosis,options

@app.callback(Output('assistant-response','children'),Output('assistant-tts-text','children'),Output('page-store','data',allow_duplicate=True),Output('company-select','value',allow_duplicate=True),Input('assistant-send','n_clicks'),State('assistant-query','value'),State('company-store','data'),prevent_initial_call=True)
def assistant_query(_n,query,current_company):
    if not query:return '请输入问题，或使用语音输入。','',no_update,no_update
    result=ai.chat(query,ds.companies)
    return result['reply'],result['reply'],result.get('page') or no_update,result.get('company_id') or current_company

@app.callback(Output('decision-result','children'),Input({'type':'choose-option','index':ALL},'n_clicks'),State({'type':'choose-option','index':ALL},'id'),prevent_initial_call=True)
def choose_option(clicks,ids):
    if not clicks or not any(x for x in clicks if x):raise PreventUpdate
    idx=max(range(len(clicks)),key=lambda i:clicks[i] or 0);code=ids[idx]['index']
    return html.Div([html.Div('DECISION RECORDED',className='decision-ok'),html.Div(f'已选择 {code}。演示版只记录前端决策；生产版应写入审批流、审计日志并要求二次确认。',className='insight-copy')],className='glass-panel')

if __name__=='__main__':
    app.run(host=APP_HOST,port=APP_PORT,debug=DEBUG)
