#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from warroom.database import init_db

if __name__=='__main__':
    init_db(force='--force' in sys.argv)
    print('Database initialized.')
