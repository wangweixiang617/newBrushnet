#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 scripts/init_db.py
python3 server.py
