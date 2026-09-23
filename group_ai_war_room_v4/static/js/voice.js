(()=>{
  let recognition=null,active=false;
  function speak(text){if(!('speechSynthesis'in window)||!text)return;speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang='zh-CN';u.rate=.96;u.pitch=1;speechSynthesis.speak(u)}
  function start(onText,onState){const SR=window.SpeechRecognition||window.webkitSpeechRecognition;if(!SR){onState?.('unsupported');return false}if(active){recognition?.stop();return true}recognition=new SR();recognition.lang='zh-CN';recognition.interimResults=true;recognition.continuous=false;recognition.onstart=()=>{active=true;onState?.('listening')};recognition.onresult=e=>{let text='';for(let i=e.resultIndex;i<e.results.length;i++)text+=e.results[i][0].transcript;if(text)onText?.(text,e.results[e.results.length-1].isFinal)};recognition.onerror=()=>onState?.('error');recognition.onend=()=>{active=false;onState?.('ready')};recognition.start();return true}
  window.WarVoice={speak,start};
})();
