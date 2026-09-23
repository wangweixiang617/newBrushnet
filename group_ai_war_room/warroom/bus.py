import queue, threading


class EventBus:
    def __init__(self):
        self._lock=threading.Lock()
        self._subscribers=[]
        self._seq=0

    def subscribe(self):
        q=queue.Queue(maxsize=100)
        with self._lock:self._subscribers.append(q)
        return q

    def unsubscribe(self,q):
        with self._lock:
            if q in self._subscribers:self._subscribers.remove(q)

    def publish(self,event):
        with self._lock:
            self._seq+=1
            payload={**event,'seq':self._seq}
            for q in list(self._subscribers):
                try:q.put_nowait(payload)
                except queue.Full:
                    try:q.get_nowait();q.put_nowait(payload)
                    except Exception:pass
        return payload

BUS=EventBus()
