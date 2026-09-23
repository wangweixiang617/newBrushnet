import os
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/'data'

APP_HOST=os.getenv('APP_HOST','0.0.0.0')
APP_PORT=int(os.getenv('APP_PORT','8050'))
DEBUG=os.getenv('DEBUG','0')=='1'
DATA_MODE=os.getenv('DATA_MODE','csv').lower()  # csv | database
DATABASE_URL=os.getenv('DATABASE_URL',f"sqlite:///{DATA_DIR/'demo.db'}")
AI_MODE=os.getenv('AI_MODE','mock').lower()      # mock | remote
AI_ENDPOINT=os.getenv('AI_ENDPOINT','')
ENABLE_BACKGROUND_MONITOR=os.getenv('ENABLE_BACKGROUND_MONITOR','1')=='1'
MONITOR_INTERVAL_SECONDS=int(os.getenv('MONITOR_INTERVAL_SECONDS','30'))
DEFAULT_COMPANY_ID=int(os.getenv('DEFAULT_COMPANY_ID','1001'))
SIMULATION_RUNS=int(os.getenv('SIMULATION_RUNS','3000'))
