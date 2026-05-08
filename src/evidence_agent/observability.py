from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
APP_LOG = LOG_DIR / "app.log"
EVENT_LOG = LOG_DIR / "agent-events.jsonl"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if any(getattr(handler, "_evidence_agent_handler", False) for handler in root.handlers):
        return

    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler = logging.FileHandler(APP_LOG, encoding="utf-8")
    handler.setFormatter(formatter)
    handler._evidence_agent_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)


def logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def append_agent_event(run_id: str, event: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": utc_now(),
        "run_id": run_id,
        **event,
    }
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
