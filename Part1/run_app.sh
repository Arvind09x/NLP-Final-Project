#!/bin/zsh
set -euo pipefail

for env_file in ../.env .env ../Part2/.env; do
  if [[ -f "$env_file" ]]; then
    set -a
    source "$env_file"
    set +a
  fi
done

export PYTHONPATH=src
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll
exec ./venv/bin/streamlit run app.py
