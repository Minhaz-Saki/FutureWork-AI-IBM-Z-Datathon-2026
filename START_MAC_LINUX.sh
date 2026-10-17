#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m pip install -r requirements-ui.txt
echo "Open http://localhost:8899 in your browser. Leave this terminal open."
exec python3 app.py --port 8899
