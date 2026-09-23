from dash import html, dcc

NAV_ITEMS=[
    ('overview','◈','集团总览'),('ranking','▥','企业排行'),('portrait','◉','企业画像'),('profile','⌁','全景指标'),
    ('investment','◇','投资分析'),('twin','✦','数字孪生'),('monitor','⚙','AI监测'),('decision','◆','AI决策')
]

def nav_button(page,icon,label):
    return html.Button([html.Span(icon,className='nav-icon'),html.Span(label)],id=f'nav-{page}',className='nav-item',n_clicks=0)

def kpi_card(title,value,sub='',tone='cyan',unit=''):
    return html.Div([
        html.Div(title,className='kpi-title'),
        html.Div([html.Span(value,className='kpi-value'),html.Span(unit,className='kpi-unit')],className='kpi-main'),
        html.Div(sub,className='kpi-sub')
    ],className=f'kpi-card tone-{tone}')

def panel(title,children,extra_class=''):
    return html.Div([
        html.Div([html.Span(title),html.Span('●',className='panel-dot')],className='panel-head'),
        html.Div(children,className='panel-body')
    ],className=f'glass-panel {extra_class}')

def graph_panel(title,fig,graph_id=None,extra_class=''):
    graph=dcc.Graph(id=graph_id,figure=fig,config={'displayModeBar':False,'responsive':True},className='twin-graph') if graph_id else dcc.Graph(figure=fig,config={'displayModeBar':False,'responsive':True},className='twin-graph')
    return panel(title,graph,extra_class)

def status_badge(text,tone='normal'):
    return html.Span(text,className=f'status-badge status-{tone.lower()}')
