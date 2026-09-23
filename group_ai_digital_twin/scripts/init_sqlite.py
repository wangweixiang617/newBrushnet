from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from services.data_service import DataService

if __name__=='__main__':
    ds=DataService(mode='csv')
    ds.export_sqlite('sqlite:///data/demo.db')
    print('SQLite initialized: data/demo.db')
