@echo off
cd /d %~dp0
python scripts\init_db.py
python server.py --host 0.0.0.0 --port 8050
pause
