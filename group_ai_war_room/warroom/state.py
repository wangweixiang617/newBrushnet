import threading
from config import DEFAULT_COMPANY_ID
from .bus import BUS
from .utils import now_iso


class SceneState:
    def __init__(self):
        self.lock=threading.RLock();self.data={'scene':'overview','company_id':DEFAULT_COMPANY_ID,'metric':'profit_margin','ambient':True,'updated_at':now_iso(),'command_id':0}
    def get(self):
        with self.lock:return dict(self.data)
    def set(self,patch,publish=True):
        with self.lock:
            self.data.update(patch);self.data['updated_at']=now_iso();self.data['command_id']=int(self.data.get('command_id',0))+1;out=dict(self.data)
        if publish:BUS.publish({'type':'scene','payload':out})
        return out

SCENE=SceneState()
