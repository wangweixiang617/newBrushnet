import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, dash_table
import dash_cytoscape as cyto
from services.analytics import add_scores, rank_table, composite_score, linear_forecast, METRIC_LABELS, DOMAIN_LABELS, SCORE_COLS
from ui.components import kpi_card, panel, graph_panel, status_badge

BG='rgba(0,0,0,0)'; GRID='rgba(113,221,255,.10)'; TEXT='#cdeefa'; MUTED='#6f96a8'
CYAN='#37e7ff'; BLUE='#5a78ff'; PURPLE='#aa6cff'; GREEN='#4af3a7'; ORANGE='#ffbb55'; RED='#ff5874'

def base_fig(fig,margin=None):
    fig.update_layout(template=None,paper_bgcolor=BG,plot_bgcolor=BG,font={'color':TEXT,'family':'Arial'},
        margin=margin or dict(l=35,r=18,t=28,b=35),hoverlabel={'bgcolor':'#081521','font_color':'#dff8ff','bordercolor':'#37e7ff'})
    fig.update_xaxes(gridcolor=GRID,zerolinecolor=GRID,tickfont={'color':MUTED})
    fig.update_yaxes(gridcolor=GRID,zerolinecolor=GRID,tickfont={'color':MUTED})
    return fig

def overview_page(ds,company_id,events):
    df=add_scores(ds.companies);c=ds.company(company_id);score=composite_score(c)
    open_events=[e for e in events if e.get('status')=='OPEN'];company_events=[e for e in open_events if e.get('company_id')==int(company_id)]
    fig_map=go.Figure()
    fig_map.add_trace(go.Scattergeo(lon=df.lon,lat=df.lat,text=df.company_name,customdata=np.stack([df.city,df.composite_score],axis=1),
        mode='markers+text',textposition='top center',marker={'size':12+df.composite_score/12,'color':df.composite_score,'colorscale':'Turbo','line':{'width':1,'color':'#bff8ff'},'cmin':45,'cmax':95,'showscale':False},
        textfont={'size':10,'color':'#bdebf7'},hovertemplate='%{text}<br>%{customdata[0]}<br>综合评分 %{customdata[1]:.1f}<extra></extra>'))
    fig_map.update_geos(scope='asia',projection_type='natural earth',showland=True,landcolor='#07131d',showocean=True,oceancolor='#030910',
        showcountries=True,countrycolor='#173849',showcoastlines=True,coastlinecolor='#19475a',center={'lat':34,'lon':105},projection_scale=3.15,
        lonaxis_range=[72,136],lataxis_range=[16,55],bgcolor=BG)
    fig_map.update_layout(paper_bgcolor=BG,margin=dict(l=0,r=0,t=0,b=0),height=360)

    hist=ds.history.groupby('month',as_index=False)[['revenue','cash_flow']].sum()
    fig_trend=go.Figure()
    fig_trend.add_trace(go.Scatter(x=hist.month,y=hist.revenue,mode='lines',name='集团收入',line={'color':CYAN,'width':3},fill='tozeroy',fillcolor='rgba(55,231,255,.06)'))
    fig_trend.add_trace(go.Scatter(x=hist.month,y=hist.cash_flow,mode='lines',name='现金流',line={'color':PURPLE,'width':2}))
    base_fig(fig_trend);fig_trend.update_layout(legend={'orientation':'h','y':1.05,'x':0})

    risk=df.sort_values('risk_score',ascending=False).head(6)
    fig_risk=go.Figure(go.Bar(x=risk.risk_score,y=risk.company_name,orientation='h',marker={'color':risk.risk_score,'colorscale':[[0,GREEN],[.55,ORANGE],[1,RED]],'cmin':40,'cmax':90},hovertemplate='%{y}<br>风险指数 %{x}<extra></extra>'))
    base_fig(fig_risk);fig_risk.update_layout(yaxis={'autorange':'reversed'})

    return html.Div([
        html.Div([
            kpi_card('集团营业收入',f"{df.revenue.sum():.1f}",'示例数据 / 亿元','cyan',' 亿'),
            kpi_card('集团平均利润率',f"{df.profit_margin.mean():.1f}",'同比监测','blue',' %'),
            kpi_card('当前企业评分',f"{score:.1f}",str(c.company_name),'purple'),
            kpi_card('AI 未闭环事件',str(len(open_events)),f'当前企业 {len(company_events)} 条','orange')
        ],className='kpi-grid'),
        html.Div([
            graph_panel('全国分公司数字分布',fig_map,extra_class='span-2'),
            panel('AI INSIGHT',html.Div([
                html.Div('GROUP INTELLIGENCE',className='eyebrow'),
                html.Div('集团数字孪生正在持续检测经营、财务、人力、项目、投资与风险指标。',className='insight-copy'),
                html.Div([status_badge('LIVE','normal'),status_badge(f'{len(open_events)} EVENTS','warning')],className='badge-row'),
                html.Hr(),
                html.Div('当前关注',className='mini-label'),
                html.Div(company_events[-1]['reason'] if company_events else '当前企业暂无新的高等级事件。',className='insight-focus')
            ]),extra_class='ai-insight')
        ],className='content-grid three-col'),
        html.Div([graph_panel('集团经营趋势',fig_trend),graph_panel('分公司风险雷达',fig_risk)],className='content-grid two-col')
    ],className='page-stack')

def ranking_page(ds):
    df=rank_table(ds.companies,'composite_score')
    fig=go.Figure(go.Bar(x=df.composite_score,y=df.company_name,orientation='h',marker={'color':df.composite_score,'colorscale':'Turbo'},text=df.composite_score,textposition='outside'))
    base_fig(fig);fig.update_layout(yaxis={'autorange':'reversed'},xaxis_title='综合评分')
    table=df[['rank','company_name','region','industry','revenue','profit_margin','growth','composite_score']].copy()
    table.columns=['排名','企业','区域','行业','营业收入','利润率','增长率','综合评分']
    return html.Div([
        html.Div([graph_panel('集团企业综合排行',fig,extra_class='span-2'),panel('排行说明',html.Div([
            html.Div('评分由经营、财务、人力、创新、投资、市场、合规、ESG等维度综合形成。',className='insight-copy'),
            html.Div('点击顶部企业选择器可切换当前分析对象。',className='hint')]))],className='content-grid three-col'),
        panel('企业排行明细',dash_table.DataTable(data=table.round(2).to_dict('records'),columns=[{'name':c,'id':c} for c in table.columns],
            page_size=8,style_as_list_view=True,style_header={'backgroundColor':'#091a26','color':'#7fe9ff','border':'0'},
            style_cell={'backgroundColor':'rgba(5,16,25,.72)','color':'#cdeefa','border':'0','padding':'12px','fontFamily':'Arial'},
            style_data_conditional=[{'if':{'row_index':'odd'},'backgroundColor':'rgba(12,34,48,.55)'}]))
    ],className='page-stack')

def portrait_page(ds,company_id):
    df=add_scores(ds.companies);current=df[df.company_id==int(company_id)].iloc[0]
    colors=[CYAN if int(x)==int(company_id) else 'rgba(102,166,190,.52)' for x in df.company_id]
    sizes=[28 if int(x)==int(company_id) else 14 for x in df.company_id]
    fig=go.Figure(go.Scatter(x=df.value_index,y=df.growth_index,mode='markers+text',text=df.company_name,textposition='top center',
        marker={'size':sizes,'color':colors,'line':{'width':1.5,'color':'#bdf7ff'}},customdata=np.stack([df.composite_score,df.industry],axis=1),
        hovertemplate='%{text}<br>价值 %{x:.1f}<br>成长 %{y:.1f}<br>综合 %{customdata[0]:.1f}<br>%{customdata[1]}<extra></extra>'))
    fig.add_vline(x=70,line_dash='dash',line_color='rgba(55,231,255,.28)');fig.add_hline(y=65,line_dash='dash',line_color='rgba(55,231,255,.28)')
    fig.add_annotation(x=85,y=88,text='核心企业',showarrow=False,font={'color':GREEN,'size':14});fig.add_annotation(x=55,y=88,text='潜力企业',showarrow=False,font={'color':CYAN,'size':14})
    fig.add_annotation(x=85,y=42,text='稳健企业',showarrow=False,font={'color':PURPLE,'size':14});fig.add_annotation(x=55,y=42,text='风险关注',showarrow=False,font={'color':ORANGE,'size':14})
    base_fig(fig);fig.update_xaxes(range=[35,105],title='企业价值指数');fig.update_yaxes(range=[25,105],title='成长指数')
    return html.Div([
        html.Div([graph_panel('企业画像四象限',fig,extra_class='span-2'),panel('当前企业画像',html.Div([
            html.Div(str(current.company_name),className='hero-company'),
            html.Div([kpi_card('价值指数',f"{current.value_index:.1f}",'','cyan'),kpi_card('成长指数',f"{current.growth_index:.1f}",'','purple')],className='mini-kpi-grid'),
            html.Div(f"行业：{current.industry} · 区域：{current.region}",className='hint'),
            html.Div('四象限用于集团层面的企业分层，可进一步接入业务规则或聚类模型。',className='insight-copy')
        ]))],className='content-grid three-col')
    ],className='page-stack')

def profile_page(ds,company_id):
    c=ds.company(company_id);hist=ds.company_history(company_id)
    labels=[DOMAIN_LABELS[x] for x in SCORE_COLS];values=[float(c[x]) for x in SCORE_COLS]
    fig_radar=go.Figure(go.Scatterpolar(r=values+[values[0]],theta=labels+[labels[0]],fill='toself',line={'color':CYAN,'width':2},fillcolor='rgba(55,231,255,.12)'))
    fig_radar.update_layout(polar={'bgcolor':BG,'radialaxis':{'range':[0,100],'gridcolor':GRID,'tickfont':{'color':MUTED}},'angularaxis':{'gridcolor':GRID,'tickfont':{'color':TEXT}}},paper_bgcolor=BG,font={'color':TEXT},margin=dict(l=40,r=40,t=20,b=25))
    fig_hist=go.Figure()
    fig_hist.add_trace(go.Scatter(x=hist.month,y=hist.profit_margin,name='利润率',line={'color':CYAN,'width':3}))
    fig_hist.add_trace(go.Scatter(x=hist.month,y=hist.employee_sentiment,name='员工状态指数',yaxis='y2',line={'color':PURPLE,'width':2}))
    base_fig(fig_hist);fig_hist.update_layout(yaxis={'title':'利润率 %','gridcolor':GRID},yaxis2={'title':'员工状态','overlaying':'y','side':'right','showgrid':False},legend={'orientation':'h','y':1.05})
    options=[{'label':METRIC_LABELS.get(c,c),'value':c} for c in ['profit_margin','cash_flow','employee_sentiment','employee_turnover','project_progress','operating_cost']]
    return html.Div([
        html.Div([graph_panel('企业能力雷达',fig_radar),graph_panel('历史趋势联动',fig_hist)],className='content-grid two-col'),
        panel('自定义指标分析',[html.Div([dcc.Dropdown(id='metric-dropdown',options=options,value='profit_margin',clearable=False,className='dark-dropdown'),
            html.Div('选择指标后，下方趋势与AI提示自动更新。',className='hint')],className='metric-toolbar'),dcc.Graph(id='metric-detail-graph',config={'displayModeBar':False},className='twin-graph')])
    ],className='page-stack')

def investment_page(ds,company_id):
    inv=ds.investments.copy();current=ds.company_investments(company_id)
    fig_sun=px.sunburst(inv,path=['sector','subsector','project'],values='amount',color='roi',color_continuous_scale='Turbo')
    fig_sun.update_traces(marker_line_color='#0b2432',marker_line_width=1);base_fig(fig_sun,dict(l=5,r=5,t=10,b=10));fig_sun.update_layout(coloraxis_showscale=False)
    agg=inv.groupby('company_id',as_index=False).agg(amount=('amount','sum'),npv=('npv','sum'),roi=('roi','mean')).merge(ds.companies[['company_id','company_name']],on='company_id').sort_values('npv')
    fig_rank=go.Figure(go.Bar(x=agg.npv,y=agg.company_name,orientation='h',marker={'color':agg.roi,'colorscale':'Turbo'},customdata=agg[['amount','roi']],hovertemplate='%{y}<br>NPV %{x:.1f}<br>投资 %{customdata[0]:.1f}<br>ROI %{customdata[1]:.1f}%<extra></extra>'))
    base_fig(fig_rank);fig_rank.update_xaxes(title='NPV（示例单位）')
    return html.Div([
        html.Div([graph_panel('集团投资结构 Sunburst',fig_sun),graph_panel('分公司投资价值排行',fig_rank)],className='content-grid two-col'),
        html.Div([kpi_card('当前企业投资额',f"{current.amount.sum():.1f}",'项目总投入','cyan'),kpi_card('平均 ROI',f"{current.roi.mean():.1f}",'当前企业','green',' %'),kpi_card('项目 NPV',f"{current.npv.sum():.1f}",'当前企业','purple')],className='kpi-grid three')
    ],className='page-stack')

def twin_page(ds,company_id):
    c=ds.company(company_id)
    domains=[('finance','财务 Twin',c.finance_score),('operation','经营 Twin',c.operation_score),('hr','人力 Twin',c.hr_score),('investment','投资 Twin',c.investment_score),('project','项目 Twin',100-min(c.project_delay_days*4,60)),('risk','风险 Twin',100-c.risk_score*0.65)]
    elements=[{'data':{'id':'company','label':str(c.company_name),'kind':'core'}},{'data':{'id':'ai','label':'AI CORE','kind':'ai'}},{'data':{'id':'decision','label':'DECISION ENGINE','kind':'decision'}}]
    for node,label,score in domains:
        state='danger' if score<60 else 'warning' if score<72 else 'normal'
        elements.append({'data':{'id':node,'label':label,'score':round(float(score),1),'state':state,'kind':'domain'}})
        elements.append({'data':{'id':f'e-company-{node}','source':'company','target':node,'active':1}})
        elements.append({'data':{'id':f'e-{node}-ai','source':node,'target':'ai','active':1}})
    elements.extend([{'data':{'id':'e-ai-decision','source':'ai','target':'decision','active':1}}])
    stylesheet=[
        {'selector':'node','style':{'label':'data(label)','color':'#d9f7ff','font-size':11,'text-valign':'center','text-halign':'center','background-color':'#102b39','border-width':1.5,'border-color':'#3fe8ff','width':90,'height':54,'text-wrap':'wrap'}},
        {'selector':'node[kind = "core"]','style':{'width':150,'height':72,'background-color':'#153c4d','border-width':3,'font-size':13}},
        {'selector':'node[kind = "ai"]','style':{'shape':'hexagon','width':110,'height':80,'background-color':'#32205b','border-color':'#b181ff','border-width':3}},
        {'selector':'node[kind = "decision"]','style':{'shape':'round-rectangle','width':145,'height':58,'background-color':'#133a2d','border-color':'#4af3a7','border-width':2}},
        {'selector':'node[state = "warning"]','style':{'border-color':'#ffbb55','background-color':'#3d2d18'}},
        {'selector':'node[state = "danger"]','style':{'border-color':'#ff5874','background-color':'#451925'}},
        {'selector':'edge','style':{'width':2,'line-color':'#35dfff','target-arrow-color':'#35dfff','target-arrow-shape':'triangle','curve-style':'bezier','line-style':'dashed','line-dash-pattern':[10,6],'opacity':.78}}
    ]
    cy=cyto.Cytoscape(id='twin-cytoscape',elements=elements,layout={'name':'breadthfirst','directed':True,'spacingFactor':1.35},stylesheet=stylesheet,style={'height':'560px','width':'100%'},minZoom=.55,maxZoom=1.6)
    return html.Div([
        dcc.Interval(id='twin-animation',interval=450,n_intervals=0),
        html.Div([panel('DIGITAL TWIN LIVE',cy,extra_class='span-2'),panel('Twin Status',html.Div([
            html.Div(str(c.company_name),className='hero-company'),html.Div('DATABASE → DOMAIN TWIN → AI CORE → DECISION',className='code-line'),
            html.Div([status_badge('LIVE','normal'),status_badge('AI MONITOR ON','normal')],className='badge-row'),
            html.Div('点击任意 Twin 节点可查看当前分值与监测状态。',className='insight-copy'),
            html.Div(id='twin-node-detail',className='node-detail')
        ]))],className='content-grid three-col')
    ],className='page-stack')

def monitor_page(ds,events):
    rules=ds.rules.copy();rules['enabled']=rules.enabled.astype(int).map({1:'ON',0:'OFF'});rules['speak']=rules.speak.astype(int).map({1:'YES',0:'NO'});rules['auto_decision']=rules.auto_decision.astype(int).map({1:'YES',0:'NO'})
    rules['metric']=rules.metric.map(lambda x:METRIC_LABELS.get(x,x))
    ev=pd.DataFrame(events[:30]) if events else pd.DataFrame(columns=['created_at','company_name','domain','title','severity','status'])
    cols=['created_at','company_name','domain','title','severity','status'];ev=ev[[c for c in cols if c in ev.columns]]
    return html.Div([
        html.Div([panel('AI MONITOR CONFIGURATION',html.Div([
            html.Div([dcc.Dropdown(id='rule-domain',options=[{'label':x,'value':x} for x in ['财务','经营','人力','员工状态','投资','项目','风险','市场','采购','合规','ESG']],value='财务',clearable=False,className='dark-dropdown'),
                dcc.Dropdown(id='rule-metric',options=[{'label':v,'value':k} for k,v in METRIC_LABELS.items()],value='profit_margin',clearable=False,className='dark-dropdown'),
                dcc.Dropdown(id='rule-op',options=[{'label':x,'value':x} for x in ['<','<=','>','>=']],value='<',clearable=False,className='dark-dropdown'),
                dcc.Input(id='rule-threshold',type='number',value=8,debounce=True,className='dark-input'),
                dcc.Dropdown(id='rule-severity',options=[{'label':x,'value':x} for x in ['NOTICE','WARNING','CRITICAL']],value='WARNING',clearable=False,className='dark-dropdown')],className='rule-grid'),
            html.Div([dcc.Checklist(id='rule-flags',options=[{'label':'自动语音播报','value':'speak'},{'label':'自动生成决策入口','value':'auto'}],value=['speak','auto'],inline=True,className='dark-check'),
                html.Button('新增监测项',id='add-rule-btn',className='primary-btn'),html.Button('立即执行AI检测',id='run-monitor-btn',className='ghost-btn'),html.Button('触发演示预警',id='demo-alert-btn',className='danger-btn')],className='action-row'),
            html.Div(id='monitor-action-msg',className='hint')
        ]),extra_class='span-2'),panel('监测域',html.Div([html.Div('FINANCE · OPERATION · HR · EMPLOYEE · INVESTMENT · PROJECT · RISK · COMPLIANCE · ESG',className='code-line'),html.Div('第一阶段使用规则与统计逻辑；AI服务通过统一REST接口预留，可替换为集团内部模型。',className='insight-copy')]))],className='content-grid three-col'),
        panel('当前监测规则',dash_table.DataTable(id='rules-table',data=rules.to_dict('records'),columns=[{'name':c,'id':c} for c in rules.columns],page_size=8,style_as_list_view=True,style_header={'backgroundColor':'#091a26','color':'#7fe9ff','border':'0'},style_cell={'backgroundColor':'rgba(5,16,25,.72)','color':'#cdeefa','border':'0','padding':'10px'})),
        panel('AI事件流',dash_table.DataTable(id='events-table',data=ev.to_dict('records'),columns=[{'name':c,'id':c} for c in ev.columns],page_size=8,style_as_list_view=True,style_header={'backgroundColor':'#091a26','color':'#7fe9ff','border':'0'},style_cell={'backgroundColor':'rgba(5,16,25,.72)','color':'#cdeefa','border':'0','padding':'10px'}))
    ],className='page-stack')

def decision_page(ds,company_id,event,diagnosis,options):
    c=ds.company(company_id);hist=ds.company_history(company_id)
    metric=(event or {}).get('metric','profit_margin');forecast=linear_forecast(hist,metric if metric in hist.columns else 'profit_margin',6)
    fig=go.Figure()
    metric_plot=metric if metric in hist.columns else 'profit_margin'
    fig.add_trace(go.Scatter(x=hist.month,y=hist[metric_plot],name='历史',line={'color':CYAN,'width':3}))
    if forecast:fig.add_trace(go.Scatter(x=[x['month'] for x in forecast],y=[x['value'] for x in forecast],name='预测',line={'color':ORANGE,'width':3,'dash':'dot'}))
    base_fig(fig);fig.update_layout(legend={'orientation':'h','y':1.05})
    cards=[]
    for o in options or []:
        tone='green' if o['risk']=='LOW' else 'orange' if o['risk']=='MEDIUM' else 'red'
        cards.append(html.Div([
            html.Div(o['code'],className='option-code'),html.Div(o['name'],className='option-title'),html.Div(o['summary'],className='option-summary'),
            html.Div([html.Div([html.Span('投入'),html.B(f"{o['investment']:.1f}")]),html.Div([html.Span('3年NPV'),html.B(f"{o['npv']:.1f}")]),html.Div([html.Span('成功概率'),html.B(f"{o['success_prob']:.1f}%")]),html.Div([html.Span('风险'),html.B(o['risk'])])],className='option-metrics'),
            html.Button('选择此方案',id={'type':'choose-option','index':o['code']},className='option-btn')
        ],className=f'option-card tone-{tone}'))
    diag=diagnosis or {'summary':'请选择AI事件或触发演示预警后进入分析。','causes':['历史趋势','集团横向比较','业务规则'],'confidence':0}
    return html.Div([
        html.Div([graph_panel(f"AI趋势诊断 · {METRIC_LABELS.get(metric,metric)}",fig,extra_class='span-2'),panel('AI DIAGNOSIS',html.Div([
            html.Div(str(c.company_name),className='hero-company'),html.Div(diag['summary'],className='insight-copy'),html.Div(f"Confidence {diag.get('confidence',0)*100:.0f}%",className='code-line'),
            html.Ul([html.Li(x) for x in diag.get('causes',[])],className='cause-list')]))],className='content-grid three-col'),
        panel('DECISION OPTIONS',html.Div(cards if cards else [html.Div('等待AI生成方案…',className='empty-state')],className='option-grid')),
        html.Div(id='decision-result',className='decision-result')
    ],className='page-stack')
