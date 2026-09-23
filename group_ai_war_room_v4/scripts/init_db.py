#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from warroom.database import init_db

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reset',action='store_true',help='删除现有演示数据库后重建');args=ap.parse_args()
    p=init_db(reset=args.reset)
    print(f'数据库初始化完成: {p}')
