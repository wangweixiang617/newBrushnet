#!/usr/bin/env python3
import json, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
print('1/4 Python syntax...')
files=[ROOT/'server.py',ROOT/'config.py',*sorted((ROOT/'warroom').glob('*.py')),*sorted((ROOT/'scripts').glob('*.py'))]
for f in files:compile(f.read_text(encoding='utf-8'),str(f),'exec')
print('   OK')
print('2/4 Core smoke test...')
r=subprocess.run([sys.executable,str(ROOT/'tests'/'smoke_test.py')],cwd=ROOT,capture_output=True,text=True,timeout=30)
print('   '+r.stdout.strip());assert r.returncode==0,r.stderr
print('3/4 Static files...')
for rel in ['static/index.html','static/control.html','static/css/style.css','static/js/warroom.js','static/js/visuals.js','static/js/charts.js','static/js/voice.js','static/data/china.geojson']:
    p=ROOT/rel;assert p.exists() and p.stat().st_size>100,rel
print('   OK')
print('4/4 Package ready.')
