(function(){
  if(window.__groupTwinBound)return;window.__groupTwinBound=true;
  let lastTts='',lastAssistantTts='';
  function speak(text){
    if(!text||!('speechSynthesis' in window))return;
    window.speechSynthesis.cancel();
    const u=new SpeechSynthesisUtterance(text);u.lang='zh-CN';u.rate=1.02;u.pitch=1.0;window.speechSynthesis.speak(u);
  }
  function bindVoice(){
    const btn=document.getElementById('voice-btn');
    if(!btn||btn.dataset.bound)return;
    btn.dataset.bound='1';
    btn.addEventListener('click',()=>{
      const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
      if(!SR){speak('当前浏览器不支持语音识别，请使用最新版 Chrome 或 Edge。');return;}
      const rec=new SR();rec.lang='zh-CN';rec.interimResults=false;rec.maxAlternatives=1;
      btn.textContent='◉';btn.classList.add('listening');
      rec.onresult=(e)=>{
        const input=document.getElementById('assistant-query');
        if(!input)return;
        const text=e.results[0][0].transcript;const setter=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;
        setter.call(input,text);input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new Event('change',{bubbles:true}));
        setTimeout(()=>{const send=document.getElementById('assistant-send');if(send)send.click();},120);
      };
      rec.onerror=()=>speak('语音识别失败，请重试。');rec.onend=()=>btn.classList.remove('listening');rec.start();
    });
  }
  function pollTts(){
    const a=document.getElementById('tts-text');const b=document.getElementById('assistant-tts-text');
    if(a&&a.textContent&&a.textContent!==lastTts){lastTts=a.textContent;speak(lastTts);}
    if(b&&b.textContent&&b.textContent!==lastAssistantTts){lastAssistantTts=b.textContent;speak(lastAssistantTts);}
  }
  setInterval(()=>{bindVoice();pollTts();},700);
})();
