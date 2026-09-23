import json, math
from datetime import datetime, timezone


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def clamp(v,lo,hi):
    return max(lo,min(hi,v))


def json_dumps(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(',',':'))


def safe_float(v,default=0.0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except (TypeError,ValueError):
        return default
