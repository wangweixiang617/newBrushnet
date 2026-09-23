import json, math
from datetime import datetime, timezone


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def json_dumps(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(',',':'),default=str)


def safe_float(value,default=0.0):
    try:return float(value)
    except (TypeError,ValueError):return default


def clamp(value,lo,hi):
    return max(lo,min(hi,value))


def percentile(values,value):
    clean=sorted(float(v) for v in values if v is not None)
    if not clean:return 0.0
    return round(100*sum(v<=value for v in clean)/len(clean),1)


def mean(values):
    clean=[float(v) for v in values if v is not None]
    return sum(clean)/len(clean) if clean else 0.0


def median(values):
    clean=sorted(float(v) for v in values if v is not None)
    n=len(clean)
    if not n:return 0.0
    m=n//2
    return clean[m] if n%2 else (clean[m-1]+clean[m])/2


def stdev(values):
    clean=[float(v) for v in values if v is not None]
    if len(clean)<2:return 0.0
    m=mean(clean)
    return math.sqrt(sum((v-m)**2 for v in clean)/(len(clean)-1))
