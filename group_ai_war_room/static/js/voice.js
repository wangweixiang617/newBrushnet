(function(){
  const $=id=>document.getElementById(id);
  let recognition=null,listening=false;
  function speak(text,force=false){
    if(!('speechSynthesis' in window)||!text)return;
    try{speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang='zh-CN';u.rate=.98;u.pitch=1;u.volume=1;u.onstart=()=>setState('SPEAKING',true);u.onend=()=>setState('READY',false);speechSynthesis.speak(u)}catch(e){}
  }
  function setState(state,active){
    const s=$('voice-status'),w=$('voice-wave'),b=$('voice-btn');if(s)s.textContent=state;if(w)w.classList.toggle('active',!!active);if(b)b.classList.toggle('listening',state==='LISTENING')
  }
  async function submit(text){
    text=(text||'').trim();if(!text)return;setState('THINKING',true);
    try{const r=await fetch('/api/ai/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});const data=await r.json();if(window.WarRoom)window.WarRoom.handleVoiceIntent(data);speak(data.reply||'指令已执行')}
    catch(e){speak('指令处理失败，请检查网络连接');}finally{setTimeout(()=>setState('READY',false),700)}
  }
  function startListening(){
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;if(!SR){speak('当前浏览器不支持语音识别，请使用文字输入');return}
    if(!recognition){recognition=new SR();recognition.lang='zh-CN';recognition.interimResults=true;recognition.continuous=false;recognition.onresult=e=>{let final='',interim='';for(let i=e.resultIndex;i<e.results.length;i++){const t=e.results[i][0].transcript;e.results[i].isFinal?final+=t:interim+=t}if($('voice-text'))$('voice-text').value=final||interim;if(final)submit(final)};recognition.onend=()=>{listening=false;setState('READY',false)};recognition.onerror=()=>{listening=false;setState('READY',false)}}
    if(listening){recognition.stop();return}listening=true;setState('LISTENING',true);recognition.start()
  }
  window.addEventListener('load',()=>{
    $('voice-btn')?.addEventListener('click',startListening);$('voice-send')?.addEventListener('click',()=>submit($('voice-text')?.value));$('voice-text')?.addEventListener('keydown',e=>{if(e.key==='Enter')submit(e.target.value)})
  });
  window.WarVoice={speak,submit,startListening,setState};
})();
