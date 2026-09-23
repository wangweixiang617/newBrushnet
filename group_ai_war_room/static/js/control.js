(function(){
  const $=id=>document.getElementById(id);let boot=null,es=null,recognition=null;
  async function api(path,opts={}){const r=await fetch(path,{headers:{'Content-Type':'application/json'},...opts});const d=await r.json();if(!r.ok)throw new Error(d.error||r.status);return d}const post=(p,o)=>api(p,{method:'POST',body:JSON.stringify(o||{})});
  function log(msg){const d=document.createElement('div');d.className='log-line';d.textContent=new Date().toLocaleTimeString('zh-CN',{hour12:false})+'  '+msg;$('live-log').prepend(d);while($('live-log').children.length>60)$('live-log').lastChild.remove()}
  async function init(){new WarVisuals.AmbientParticles($('ambient-canvas'));boot=await api('/api/bootstrap');$('company-select').innerHTML=boot.companies.map(c=>`<option value="${c.company_id}" ${c.company_id===boot.company.company_id?'selected':''}>${c.company_name}</option>`).join('');renderStatus();renderRules();renderDecisions();attach();connect();log('控制台已连接')}
  function renderStatus(){const s=boot.summary;$('system-status').innerHTML=`<div class="big-kpi"><span>分公司</span><strong>${s.subsidiaries}</strong></div><div class="big-kpi"><span>风险对象</span><strong>${s.risk_count}</strong></div><div class="big-kpi"><span>集团营收</span><strong>${s.revenue}</strong></div><div class="big-kpi"><span>综合健康</span><strong>${s.score}</strong></div>`}
  function renderRules(){$('rules-body').innerHTML=boot.rules.map(r=>`<tr><td>${r.domain}</td><td>${r.metric}</td><td>${r.op} ${r.threshold}</td><td><span class="status-badge">${r.severity}</span></td><td>${+r.speak?'YES':'NO'}</td><td>${+r.auto_decision?'YES':'NO'}</td></tr>`).join('')}
  async function renderDecisions(){const ds=await api('/api/decisions?limit=20');$('decisions-body').innerHTML=ds.map(d=>`<tr><td>${String(d.created_at).replace('T',' ').slice(0,19)}</td><td>${boot.companies.find(c=>c.company_id===d.company_id)?.company_name||d.company_id}</td><td>${d.option_id} · ${d.option_name||''}</td><td>${d.operator||''}</td><td>${d.rationale||''}</td></tr>`).join('')||'<tr><td colspan="5">暂无决策记录</td></tr>'}
  function ctx(){return{company_id:+$('company-select').value,metric:$('metric-select').value,ambient:false}}
  function attach(){
    document.querySelectorAll('[data-scene]').forEach(b=>b.onclick=()=>{post('/api/scene',{...ctx(),scene:b.dataset.scene}).then(()=>log('切换大屏场景：'+b.dataset.scene))});
    $('company-select').onchange=()=>post('/api/scene',{...ctx(),scene:'overview'});$('metric-select').onchange=()=>post('/api/scene',{...ctx()});
    $('demo-alert').onclick=()=>post('/api/alerts/demo',{...ctx(),severity:'CRITICAL'}).then(x=>log('触发演示预警：'+x.company_name));
    $('start-diagnosis').onclick=()=>post('/api/scene',{...ctx(),scene:'decision',action:'diagnose'}).then(()=>log('启动 AI 诊断场景'));
    $('start-simulation').onclick=()=>post('/api/scene',{...ctx(),scene:'decision',action:'simulate'}).then(()=>log('启动方案仿真'));
    $('run-scan').onclick=()=>post('/api/monitor/scan',{force:true}).then(x=>log('扫描完成，新增事件 '+x.created.length+' 条'));
    $('command-send').onclick=sendCommand;$('command-text').onkeydown=e=>{if(e.key==='Enter')sendCommand()};$('command-voice').onclick=startVoice;
    $('add-rule').onclick=async()=>{const p={domain:$('rule-domain').value,metric:$('rule-metric').value,op:$('rule-op').value,threshold:+$('rule-threshold').value,severity:$('rule-severity').value,enabled:true,speak:true,auto_decision:true};await post('/api/monitor/rules',p);boot.rules=await api('/api/monitor/rules');renderRules();log('新增监测规则：'+p.metric)};
  }
  async function sendCommand(){const text=$('command-text').value.trim();if(!text)return;const d=await post('/api/ai/chat',{text});$('intent-result').innerHTML=`<strong>${d.reply}</strong>Scene: ${d.intent.scene} · Action: ${d.intent.action} · Metric: ${d.intent.metric}`;log('AI 指令：'+text)}
  function startVoice(){const SR=window.SpeechRecognition||window.webkitSpeechRecognition;if(!SR){alert('当前浏览器不支持语音识别，建议使用 Chrome / Edge');return}if(!recognition){recognition=new SR();recognition.lang='zh-CN';recognition.onresult=e=>{$('command-text').value=e.results[0][0].transcript;sendCommand()}}recognition.start()}
  function connect(){es=new EventSource('/api/events/stream');es.onmessage=e=>{const x=JSON.parse(e.data);if(x.type==='alert')log(`ALERT ${x.severity} · ${x.company_name} · ${x.metric}`);else if(x.type==='scene')log('大屏状态更新：'+x.payload.scene);else if(x.type==='decision_confirmed'){log('决策已确认：OPTION '+x.option?.id);renderDecisions()}else if(x.type==='rules_changed')log('监控规则已更新')};es.onerror=()=>log('SSE 正在重连...')}
  document.addEventListener('DOMContentLoaded',init);
})();
