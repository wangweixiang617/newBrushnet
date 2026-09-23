#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/init_db.py
exec python3 server.py --host 0.0.0.0 --port "${PORT:-8050}"
