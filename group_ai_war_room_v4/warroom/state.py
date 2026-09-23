import threading
from config import DEFAULT_COMPANY_ID
from .bus import BUS
from .utils import now_iso

class SceneState:
    def __init__(self):
        self._lock=threading.Lock();self._state={'scene':'overview','company_id':DEFAULT_COMPANY_ID,'metric':'revenue','ambient':True,'action':None,'option':None,'event_id':None,'command_id':0,'updated_at':now_iso()}
    def get(self):
        with self._lock:return dict(self._state)
    def set(self,patch,publish=True):
        with self._lock:
            self._state.update({k:v for k,v in patch.items() if v is not None or k in {'action','option','event_id'}})
            self._state['updated_at']=now_iso();self._state['command_id']=self._state.get('command_id',0)+1
            out=dict(self._state)
        if publish:BUS.publish({'type':'scene','payload':out})
        return out
SCENE=SceneState()
