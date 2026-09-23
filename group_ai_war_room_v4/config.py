from pathlib import Path
import os

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/'data'
STATIC_DIR=ROOT/'static'
DB_PATH=Path(os.getenv('WARROOM_DB',DATA_DIR/'group_twin.db'))
HOST=os.getenv('HOST','0.0.0.0')
PORT=int(os.getenv('PORT','8050'))
GROUP_NAME=os.getenv('GROUP_NAME','华辰控股集团')
DEFAULT_COMPANY_ID=int(os.getenv('DEFAULT_COMPANY_ID','1'))
AI_MODE=os.getenv('AI_MODE','mock').lower()
AI_ENDPOINT=os.getenv('AI_ENDPOINT','').rstrip('/')
AI_API_KEY=os.getenv('AI_API_KEY','')
SSE_HEARTBEAT_SECONDS=15
PROFILE_WEIGHTS={'basic':0.20,'operation':0.35,'technology':0.25,'supervision':0.20}
