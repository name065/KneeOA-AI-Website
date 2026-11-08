#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    python3.11 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -r requirements.txt
echo "Buka http://127.0.0.1:5000 di browser Anda."
python app.py
