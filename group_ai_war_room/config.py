import os
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/'data'
STATIC_DIR=BASE_DIR/'static'
DB_PATH=Path(os.getenv('DB_PATH',str(DATA_DIR/'group_twin.db')))
HOST=os.getenv('HOST','0.0.0.0')
PORT=int(os.getenv('PORT','8050'))
MONITOR_INTERVAL=float(os.getenv('MONITOR_INTERVAL','15'))
AI_MODE=os.getenv('AI_MODE','mock').lower()
AI_ENDPOINT=os.getenv('AI_ENDPOINT','').strip()
AI_API_KEY=os.getenv('AI_API_KEY','').strip()
GROUP_NAME=os.getenv('GROUP_NAME','华辰控股集团')
DEFAULT_COMPANY_ID=int(os.getenv('DEFAULT_COMPANY_ID','1001'))
EVENT_COOLDOWN_SECONDS=int(os.getenv('EVENT_COOLDOWN_SECONDS','600'))
SSE_HEARTBEAT_SECONDS=int(os.getenv('SSE_HEARTBEAT_SECONDS','12'))
