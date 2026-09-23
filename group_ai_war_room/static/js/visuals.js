(function(){
  const NS='http://www.w3.org/2000/svg';
  const color={normal:'#42f5b6',warning:'#ff9a47',critical:'#ff4f6d',cyan:'#39e7ff',purple:'#a56dff'};
  const q=(sel,root=document)=>root.querySelector(sel),qa=(sel,root=document)=>Array.from(root.querySelectorAll(sel));
  function svgEl(name,attrs={},text=''){const e=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;return e}
  function safeStatus(s){return String(s||'NORMAL').toLowerCase()}

  class AmbientParticles{
    constructor(canvas){this.canvas=canvas;this.ctx=canvas.getContext('2d');this.p=[];this.last=0;this.resize();for(let i=0;i<75;i++)this.p.push(this.make(true));window.addEventListener('resize',()=>this.resize());requestAnimationFrame(t=>this.tick(t))}
    resize(){const d=Math.max(1,Math.min(2,devicePixelRatio||1));this.w=innerWidth;this.h=innerHeight;this.canvas.width=this.w*d;this.canvas.height=this.h*d;this.ctx.setTransform(d,0,0,d,0,0)}
    make(initial=false){return{x:Math.random()*this.w,y:initial?Math.random()*this.h:this.h+5,r:.4+Math.random()*1.2,v:.05+Math.random()*.18,a:.12+Math.random()*.35}}
    tick(t){const ctx=this.ctx;ctx.clearRect(0,0,this.w,this.h);ctx.fillStyle='#39e7ff';for(const p of this.p){p.y-=p.v;p.x+=Math.sin((t+p.y)*.0006)*.04;if(p.y<-5)Object.assign(p,this.make(false));ctx.globalAlpha=p.a;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;requestAnimationFrame(tt=>this.tick(tt))}
  }

  async function renderStrategicMap(svg,companies,selectedId){
    if(!svg)return;svg.innerHTML='';const defs=svgEl('defs');const filter=svgEl('filter',{id:'mapGlow',x:'-50%',y:'-50%',width:'200%',height:'200%'});filter.append(svgEl('feGaussianBlur',{stdDeviation:'2.5',result:'b'}),svgEl('feMerge'));const merge=filter.lastChild;merge.append(svgEl('feMergeNode',{in:'b'}),svgEl('feMergeNode',{in:'SourceGraphic'}));defs.append(filter);svg.append(defs);
    let geo=null;try{geo=await fetch('/static/data/china.geojson').then(r=>r.json())}catch(e){}
    const W=1000,H=570,pad=70;
    // Natural Earth China bounds; fixed so enterprise coordinates remain aligned.
    const minLon=73,maxLon=135,minLat=18,maxLat=54;const proj=(lon,lat)=>[pad+(lon-minLon)/(maxLon-minLon)*(W-2*pad),H-pad-(lat-minLat)/(maxLat-minLat)*(H-2*pad)];
    for(let i=0;i<9;i++){const y=55+i*55;svg.append(svgEl('line',{x1:50,y1:y,x2:950,y2:y,class:'map-gridline'}))}
    for(let i=0;i<12;i++){const x=55+i*78;svg.append(svgEl('line',{x1:x,y1:30,x2:x,y2:540,class:'map-gridline'}))}
    if(geo){for(const f of geo.features||[]){const mp=f.geometry.type==='MultiPolygon'?f.geometry.coordinates:[f.geometry.coordinates];for(const poly of mp){for(const ring of poly){let d='';ring.forEach((pt,i)=>{const[x,y]=proj(pt[0],pt[1]);d+=(i?'L':'M')+x.toFixed(1)+' '+y.toFixed(1)+' '});d+='Z';svg.append(svgEl('path',{d,class:'map-country'}))}}}}
    const hq=companies.find(c=>String(c.city).includes('北京'))||companies[0];const[hqx,hqy]=proj(+hq.lon,+hq.lat);
    companies.forEach(c=>{if(c.company_id===hq.company_id)return;const[x,y]=proj(+c.lon,+c.lat),cls=safeStatus(c.status);const mx=(hqx+x)/2,my=Math.min(hqy,y)-40-Math.abs(x-hqx)*.07;svg.append(svgEl('path',{d:`M${hqx} ${hqy} Q${mx} ${my} ${x} ${y}`,class:`map-link ${cls}`}))});
    companies.forEach(c=>{const[x,y]=proj(+c.lon,+c.lat),cls=safeStatus(c.status),g=svgEl('g',{'data-company':c.company_id,class:'map-company'});g.append(svgEl('circle',{cx:x,cy:y,r:c.company_id===selectedId?19:14,class:'map-pulse',stroke:color[cls]||color.cyan}));g.append(svgEl('circle',{cx:x,cy:y,r:c.company_id===selectedId?6:4.5,fill:color[cls]||color.cyan,class:'map-node-core'}));g.append(svgEl('text',{x:x+10,y:y-4,class:'map-node-label'},c.city));g.append(svgEl('text',{x:x+10,y:y+10,class:'map-node-sub'},`${Number(c.score).toFixed(1)} · ${c.status}`));g.addEventListener('click',()=>window.dispatchEvent(new CustomEvent('warroom-company-select',{detail:{company_id:c.company_id}})));svg.append(g)});
  }

  function renderQuadrant(svg,companies,selectedId){
    if(!svg)return;svg.innerHTML='';const W=900,H=640,p={l:72,r:40,t:44,b:60};for(let i=0;i<=8;i++){const x=p.l+(W-p.l-p.r)*i/8;svg.append(svgEl('line',{x1:x,y1:p.t,x2:x,y2:H-p.b,class:'q-grid'}))}for(let i=0;i<=6;i++){const y=p.t+(H-p.t-p.b)*i/6;svg.append(svgEl('line',{x1:p.l,y1:y,x2:W-p.r,y2:y,class:'q-grid'}))}const midX=(p.l+W-p.r)/2,midY=(p.t+H-p.b)/2;svg.append(svgEl('line',{x1:midX,y1:p.t,x2:midX,y2:H-p.b,class:'q-axis'}));svg.append(svgEl('line',{x1:p.l,y1:midY,x2:W-p.r,y2:midY,class:'q-axis'}));
    [['潜力企业',p.l+105,p.t+55],['核心企业',W-p.r-115,p.t+55],['风险企业',p.l+105,H-p.b-45],['稳健企业',W-p.r-115,H-p.b-45]].forEach(x=>svg.append(svgEl('text',{x:x[1],y:x[2],class:'q-label','text-anchor':'middle'},x[0])));
    const values=companies.map(c=>({c,x:+c.score,y:+c.growth})),minY=Math.min(...values.map(d=>d.y),-5),maxY=Math.max(...values.map(d=>d.y),25),minX=Math.min(...values.map(d=>d.x),50),maxX=Math.max(...values.map(d=>d.x),95);
    values.forEach(({c,x,y})=>{const px=p.l+(x-minX)/(maxX-minX||1)*(W-p.l-p.r),py=H-p.b-(y-minY)/(maxY-minY||1)*(H-p.t-p.b),sel=c.company_id===selectedId,g=svgEl('g',{class:'q-point'+(sel?' selected':''),'data-company':c.company_id});if(sel)g.append(svgEl('circle',{cx:px,cy:py,r:20,class:'q-pulse'}));g.append(svgEl('circle',{cx:px,cy:py,r:sel?7:4.2,fill:sel?color.cyan:(c.status==='CRITICAL'?color.critical:c.status==='WARNING'?color.warning:'#5e8ea1')}));if(sel)g.append(svgEl('text',{x:px+12,y:py-10,class:'map-node-label'},c.city));g.addEventListener('click',()=>window.dispatchEvent(new CustomEvent('warroom-company-select',{detail:{company_id:c.company_id}})));svg.append(g)});
  }

  function renderEnterpriseOrbit(svg,company){
    if(!svg||!company)return;svg.innerHTML='';const cx=500,cy=360;svg.append(svgEl('circle',{cx,cy,r:116,class:'orbit-core-ring r1'}));svg.append(svgEl('circle',{cx,cy,r:92,class:'orbit-core-ring r2'}));svg.append(svgEl('circle',{cx,cy,r:72,fill:'rgba(7,28,43,.92)',stroke:'rgba(57,231,255,.38)'}));
    const dims=[['财务','finance_score'],['经营','operation_score'],['创新','innovation_score'],['投资','investment_score'],['人力','hr_score'],['市场','market_score'],['合规','compliance_score'],['ESG','esg_score']];const R=260;
    dims.forEach((d,i)=>{const a=-Math.PI/2+i*Math.PI*2/dims.length,x=cx+Math.cos(a)*R,y=cy+Math.sin(a)*R,val=+company[d[1]],g=svgEl('g',{class:'orbit-node'});svg.append(svgEl('line',{x1:cx+Math.cos(a)*78,y1:cy+Math.sin(a)*78,x2:x-Math.cos(a)*42,y2:y-Math.sin(a)*42,class:'orbit-link'}));g.append(svgEl('circle',{cx:x,cy:y,r:49,class:'pulse'}));g.append(svgEl('circle',{cx:x,cy:y,r:37,class:'base'}));g.append(svgEl('text',{x,y:y-4,class:'value'},Math.round(val)));g.append(svgEl('text',{x,y:y+17,class:'name'},d[0]));svg.append(g)});
    // animated capability polygon behind nodes
    const pts=dims.map((d,i)=>{const a=-Math.PI/2+i*Math.PI*2/dims.length,r=95+(+company[d[1]]/100)*72;return[cx+Math.cos(a)*r,cy+Math.sin(a)*r]});svg.insertBefore(svgEl('polygon',{points:pts.map(p=>p.join(',')).join(' '),fill:'rgba(57,231,255,.035)',stroke:'rgba(57,231,255,.16)','stroke-dasharray':'5 8'}),svg.firstChild);
  }

  class FlowNetwork{
    constructor(svg){this.svg=svg;this.nodes=[];this.edges=[];this.particles=[];this.raf=null;this.t0=performance.now()}
    set(company){if(!this.svg||!company)return;cancelAnimationFrame(this.raf);this.svg.innerHTML='';this.particles=[];const W=1200,H=720;const nodes=[
      {id:'db',x:105,y:360,title:'DATA HUB',state:'DATABASE',score:100,status:'NORMAL'},
      {id:'finance',x:350,y:115,title:'FINANCE',state:'财务 TWIN',score:+company.finance_score,status:+company.profit_margin<8?'CRITICAL':+company.finance_score<70?'WARNING':'NORMAL'},
      {id:'operation',x:350,y:250,title:'OPERATION',state:'经营 TWIN',score:+company.operation_score,status:+company.operation_score<70?'WARNING':'NORMAL'},
      {id:'hr',x:350,y:385,title:'HR / SENTIMENT',state:'人力 TWIN',score:+company.hr_score,status:+company.employee_sentiment<65?'WARNING':'NORMAL'},
      {id:'investment',x:350,y:520,title:'INVESTMENT',state:'投资 TWIN',score:+company.investment_score,status:+company.investment_score<65?'WARNING':'NORMAL'},
      {id:'project',x:350,y:655,title:'PROJECT',state:'项目 TWIN',score:Math.max(0,100-+company.project_delay_days*3),status:+company.project_delay_days>10?'WARNING':'NORMAL'},
      {id:'risk',x:630,y:115,title:'RISK',state:'风险 TWIN',score:100-+company.risk_score,status:+company.risk_score>=80?'CRITICAL':+company.risk_score>=72?'WARNING':'NORMAL'},
      {id:'market',x:630,y:655,title:'MARKET',state:'市场 TWIN',score:+company.market_score,status:'NORMAL'},
      {id:'ai',x:785,y:360,title:'AI CORE',state:'ANALYZING',score:96,status:'NORMAL',core:true},
      {id:'decision',x:1060,y:360,title:'DECISION',state:'ENGINE',score:100,status:'NORMAL',core:true}
    ];const by=Object.fromEntries(nodes.map(n=>[n.id,n]));const edges=[['db','finance'],['db','operation'],['db','hr'],['db','investment'],['db','project'],['finance','ai'],['operation','ai'],['hr','ai'],['investment','ai'],['project','ai'],['risk','ai'],['market','ai'],['ai','decision']];
      const defs=svgEl('defs');const marker=svgEl('marker',{id:'arrow',viewBox:'0 0 10 10',refX:'8',refY:'5',markerWidth:'5',markerHeight:'5',orient:'auto-start-reverse'});marker.append(svgEl('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'rgba(57,231,255,.55)'}));defs.append(marker);this.svg.append(defs);
      edges.forEach((e,idx)=>{const a=by[e[0]],b=by[e[1]],d=`M${a.x} ${a.y} C${(a.x+b.x)/2} ${a.y},${(a.x+b.x)/2} ${b.y},${b.x} ${b.y}`,st=(a.status==='CRITICAL'||b.status==='CRITICAL')?'critical':(a.status==='WARNING'||b.status==='WARNING')?'warning':'';const base=svgEl('path',{d,class:'flow-edge-base'}),glow=svgEl('path',{d,class:`flow-edge-glow ${st}`,id:`flowpath-${idx}`,'marker-end':'url(#arrow)'});this.svg.append(base,glow);this.particles.push({path:glow,offset:Math.random(),speed:.000075+Math.random()*.00008,status:st})});
      nodes.forEach(n=>{const g=svgEl('g',{class:`flow-node ${safeStatus(n.status)}`});if(n.core){g.append(svgEl('circle',{cx:n.x,cy:n.y,r:58,class:'node-pulse'}));g.append(svgEl('circle',{cx:n.x,cy:n.y,r:46,class:'node-shell'}))}else{g.append(svgEl('rect',{x:n.x-72,y:n.y-34,width:144,height:68,rx:8,class:'node-shell'}));g.append(svgEl('rect',{x:n.x-77,y:n.y-39,width:154,height:78,rx:10,class:'node-pulse'}))}g.append(svgEl('text',{x:n.x,y:n.y-3,class:'title'},n.title));g.append(svgEl('text',{x:n.x,y:n.y+17,class:'state'},`${n.state} · ${Math.round(n.score)}`));this.svg.append(g)});this.t0=performance.now();this.tick(performance.now())}
    tick(now){if(!this.svg)return;this.particles.forEach((p,i)=>{p.offset=(p.offset+(now-this.t0)*p.speed)%1;let circle=this.svg.querySelector(`#particle-${i}`);if(!circle){circle=svgEl('circle',{id:`particle-${i}`,r:2.8,class:'flow-particle',fill:p.status==='critical'?color.critical:p.status==='warning'?color.warning:color.cyan});this.svg.append(circle)}try{const len=p.path.getTotalLength(),pt=p.path.getPointAtLength(len*p.offset);circle.setAttribute('cx',pt.x);circle.setAttribute('cy',pt.y)}catch(e){}});this.t0=now;this.raf=requestAnimationFrame(t=>this.tick(t))}
  }

  function renderCausal(svg,data){
    if(!svg||!data)return;svg.innerHTML='';const root={x:450,y:92,label:data.root};const causePos=[[190,300],[365,260],[550,260],[710,310],[450,430]];const causes=data.causes||[];causes.forEach((c,i)=>{const p=causePos[i%causePos.length];const path=svgEl('path',{d:`M${p[0]} ${p[1]-26} Q${(p[0]+root.x)/2} ${(p[1]+root.y)/2} ${root.x} ${root.y+31}`,class:'causal-link'});path.style.strokeDasharray='1000';path.style.strokeDashoffset='1000';svg.append(path);setTimeout(()=>{path.style.transition='stroke-dashoffset .7s ease';path.style.strokeDashoffset='0'},160+i*160)});function node(x,y,label,conf,rootNode=false,delay=0){const g=svgEl('g',{class:'causal-node '+(rootNode?'root':'')});g.style.opacity='0';g.style.transformOrigin=`${x}px ${y}px`;g.style.transform='scale(.7)';g.append(svgEl('rect',{x:x-93,y:y-30,width:186,height:60,rx:8,class:'shell'}));g.append(svgEl('text',{x,y:y-3},label));if(conf!=null)g.append(svgEl('text',{x,y:y+16,class:'conf'},`${Number(conf).toFixed(0)}% CONFIDENCE`));svg.append(g);setTimeout(()=>{g.style.transition='.35s ease';g.style.opacity='1';g.style.transform='scale(1)'},delay)}node(root.x,root.y,root.label,96,true,50);causes.forEach((c,i)=>node(causePos[i][0],causePos[i][1],c.label,c.confidence,false,300+i*180))
  }

  function setupMonitorWheel(root,domains){
    if(!root)return[];qa('.monitor-orbit-node',root).forEach(x=>x.remove());const nodes=[];const cx=root.clientWidth/2,cy=root.clientHeight/2,R=Math.min(root.clientWidth,root.clientHeight)*.37;domains.forEach((d,i)=>{const a=-Math.PI/2+i*Math.PI*2/domains.length,n=document.createElement('div');n.className='monitor-orbit-node '+safeStatus(d.status);n.style.left=(cx+Math.cos(a)*R)+'px';n.style.top=(cy+Math.sin(a)*R)+'px';n.innerHTML=`<b>${d.name}</b><span>${Math.round(d.score)}</span>`;root.append(n);nodes.push(n)});return nodes
  }

  window.WarVisuals={AmbientParticles,renderStrategicMap,renderQuadrant,renderEnterpriseOrbit,FlowNetwork,renderCausal,setupMonitorWheel,colors:color};
})();
