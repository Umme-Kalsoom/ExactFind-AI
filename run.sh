#!/usr/bin/env bash
set -e
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
python app.py "$@"
