@echo off
cd /d %~dp0
python scripts\init_db.py
python server.py
pause
