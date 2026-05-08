#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

export EVIDENCE_AGENT_MODEL="${EVIDENCE_AGENT_MODEL:-gpt-5.1}"
export PYTHONPATH=src
exec .venv/bin/uvicorn evidence_agent.app:app --host 127.0.0.1 --port 8000
