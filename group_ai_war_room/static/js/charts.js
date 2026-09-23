(function(){
  const C={cyan:'#39e7ff',blue:'#4f8cff',purple:'#a56dff',magenta:'#ff5cc8',green:'#42f5b6',orange:'#ff9a47',red:'#ff4f6d',grid:'rgba(90,210,240,.09)',text:'#7499a8',white:'#dffaff'};
  function prep(canvas){
    const r=canvas.getBoundingClientRect(),dpr=Math.max(1,Math.min(2,window.devicePixelRatio||1));
    const w=Math.max(10,r.width),h=Math.max(10,r.height);canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);return{ctx,w,h};
  }
  function grid(ctx,w,h,p={l:42,r:16,t:16,b:28}){
    ctx.strokeStyle=C.grid;ctx.lineWidth=1;ctx.font='9px Consolas';ctx.fillStyle=C.text;ctx.textAlign='right';ctx.textBaseline='middle';
    for(let i=0;i<5;i++){const y=p.t+(h-p.t-p.b)*i/4;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke()}
  }
  function ease(t){return 1-Math.pow(1-t,3)}
  function line(canvas,series,opts={}){
    if(!canvas||!series?.length)return;const start=performance.now(),duration=opts.duration||900;
    function frame(now){const {ctx,w,h}=prep(canvas),p={l:38,r:14,t:14,b:24};grid(ctx,w,h,p);const vals=series.flatMap(s=>s.values||[]);let min=Math.min(...vals),max=Math.max(...vals);if(!isFinite(min)||!isFinite(max))return;if(max===min){max+=1;min-=1}const pad=(max-min)*.12;min-=pad;max+=pad;
      const prog=ease(Math.min(1,(now-start)/duration));series.forEach((s,si)=>{const color=s.color||[C.cyan,C.purple,C.green,C.orange][si%4];ctx.strokeStyle=color;ctx.lineWidth=1.8;ctx.shadowColor=color;ctx.shadowBlur=7;ctx.beginPath();const n=s.values.length;const visible=(n-1)*prog;for(let i=0;i<n;i++){if(i>visible+1)break;const x=p.l+(w-p.l-p.r)*(i/(n-1||1)),v=s.values[i],y=p.t+(h-p.t-p.b)*(1-(v-min)/(max-min));if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}ctx.stroke();ctx.shadowBlur=0;
        if(prog>=1){ctx.fillStyle=color;s.values.forEach((v,i)=>{const x=p.l+(w-p.l-p.r)*(i/(n-1||1)),y=p.t+(h-p.t-p.b)*(1-(v-min)/(max-min));ctx.beginPath();ctx.arc(x,y,2.2,0,Math.PI*2);ctx.fill()})}
      });if(prog<1)requestAnimationFrame(frame)}requestAnimationFrame(frame)
  }
  function bars(canvas,items,opts={}){
    if(!canvas||!items?.length)return;const start=performance.now(),duration=opts.duration||800;
    function frame(now){const {ctx,w,h}=prep(canvas),p={l:28,r:12,t:12,b:34},prog=ease(Math.min(1,(now-start)/duration)),max=Math.max(...items.map(x=>x.value),1);grid(ctx,w,h,p);const bw=(w-p.l-p.r)/items.length*.56;
      items.forEach((it,i)=>{const slot=(w-p.l-p.r)/items.length,x=p.l+slot*i+(slot-bw)/2,hh=(h-p.t-p.b)*(it.value/max)*prog,y=h-p.b-hh;const g=ctx.createLinearGradient(0,y,0,h-p.b);g.addColorStop(0,it.color||C.cyan);g.addColorStop(1,'rgba(57,231,255,.12)');ctx.fillStyle=g;ctx.shadowColor=it.color||C.cyan;ctx.shadowBlur=8;ctx.fillRect(x,y,bw,hh);ctx.shadowBlur=0;ctx.fillStyle=C.text;ctx.font='8px sans-serif';ctx.textAlign='center';ctx.fillText(it.label,x+bw/2,h-16)})
      if(prog<1)requestAnimationFrame(frame)}requestAnimationFrame(frame)
  }
  function radar(canvas,axes,values,opts={}){
    if(!canvas||!axes?.length)return;const start=performance.now(),duration=opts.duration||1000;
    function frame(now){const {ctx,w,h}=prep(canvas),cx=w/2,cy=h/2+3,R=Math.min(w,h)*.34,n=axes.length,prog=ease(Math.min(1,(now-start)/duration));ctx.save();ctx.translate(cx,cy);
      for(let ring=1;ring<=4;ring++){ctx.beginPath();for(let i=0;i<n;i++){const a=-Math.PI/2+i*Math.PI*2/n,r=R*ring/4,x=Math.cos(a)*r,y=Math.sin(a)*r;i?ctx.lineTo(x,y):ctx.moveTo(x,y)}ctx.closePath();ctx.strokeStyle=C.grid;ctx.stroke()}
      axes.forEach((a,i)=>{const ang=-Math.PI/2+i*Math.PI*2/n;ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(Math.cos(ang)*R,Math.sin(ang)*R);ctx.strokeStyle=C.grid;ctx.stroke();ctx.fillStyle=C.text;ctx.font='9px sans-serif';ctx.textAlign=Math.cos(ang)>.2?'left':Math.cos(ang)<-.2?'right':'center';ctx.textBaseline=Math.sin(ang)>.2?'top':Math.sin(ang)<-.2?'bottom':'middle';ctx.fillText(a,Math.cos(ang)*(R+11),Math.sin(ang)*(R+11))});
      ctx.beginPath();values.forEach((v,i)=>{const ang=-Math.PI/2+i*Math.PI*2/n,r=R*Math.max(0,Math.min(100,v))/100*prog,x=Math.cos(ang)*r,y=Math.sin(ang)*r;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath();ctx.fillStyle='rgba(57,231,255,.10)';ctx.fill();ctx.strokeStyle=C.cyan;ctx.lineWidth=1.7;ctx.shadowColor=C.cyan;ctx.shadowBlur=8;ctx.stroke();ctx.shadowBlur=0;ctx.restore();if(prog<1)requestAnimationFrame(frame)}requestAnimationFrame(frame)
  }
  function sunburst(canvas,items){
    if(!canvas||!items?.length)return;const {ctx,w,h}=prep(canvas),cx=w/2,cy=h/2,R=Math.min(w,h)*.35,total=items.reduce((a,x)=>a+x.value,0),palette=[C.cyan,C.purple,C.magenta,C.blue,C.green,C.orange];let a=-Math.PI/2;items.forEach((it,i)=>{const da=Math.PI*2*it.value/total;ctx.beginPath();ctx.arc(cx,cy,R,a,a+da);ctx.arc(cx,cy,R*.58,a+da,a,true);ctx.closePath();ctx.fillStyle=it.color||palette[i%palette.length];ctx.globalAlpha=.72;ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle='rgba(2,8,18,.8)';ctx.lineWidth=2;ctx.stroke();a+=da});ctx.beginPath();ctx.arc(cx,cy,R*.39,0,Math.PI*2);ctx.fillStyle='rgba(4,17,29,.95)';ctx.fill();ctx.strokeStyle='rgba(57,231,255,.22)';ctx.stroke();ctx.fillStyle=C.white;ctx.font='700 24px Consolas';ctx.textAlign='center';ctx.fillText(total.toFixed(1),cx,cy-2);ctx.fillStyle=C.text;ctx.font='9px sans-serif';ctx.fillText('INVESTMENT / 亿元',cx,cy+17)
  }
  window.WarCharts={line,bars,radar,sunburst,colors:C,prep};
})();
