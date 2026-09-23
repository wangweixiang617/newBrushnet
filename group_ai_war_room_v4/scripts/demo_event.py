#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from warroom.database import init_db
from warroom.monitor import ENGINE
if __name__=='__main__':
    init_db();print(ENGINE.demo_alert(1,'roe','CRITICAL'))
