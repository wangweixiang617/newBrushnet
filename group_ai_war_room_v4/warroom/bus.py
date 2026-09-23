import queue, threading

class EventBus:
    def __init__(self):
        self._lock=threading.Lock();self._subs=set()
    def subscribe(self):
        q=queue.Queue(maxsize=200)
        with self._lock:self._subs.add(q)
        return q
    def unsubscribe(self,q):
        with self._lock:self._subs.discard(q)
    def publish(self,event):
        with self._lock:subs=list(self._subs)
        for q in subs:
            try:q.put_nowait(event)
            except queue.Full:
                try:q.get_nowait();q.put_nowait(event)
                except Exception:pass

BUS=EventBus()
