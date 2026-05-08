from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from evidence_agent.observability import append_agent_event, logger
from evidence_agent.types import EventRecord, Plan, RunStatus

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
LOG = logger(__name__)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value[:44] or "run"


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def safe_resolve(path_value: str | Path, allowed_roots: list[Path] | None = None) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    roots = [root.resolve() for root in (allowed_roots or [PROJECT_ROOT])]
    if not any(is_relative_to(resolved, root) for root in roots):
        raise ValueError(f"Path is outside the allowed workspace: {path_value}")
    return resolved


class RunStore:
    def __init__(self, base_dir: Path = RUNS_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks_guard = threading.Lock()
        self._run_locks: dict[str, threading.RLock] = {}

    def _lock_for(self, run_id: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._run_locks.get(run_id)
            if lock is None:
                lock = threading.RLock()
                self._run_locks[run_id] = lock
            return lock

    @contextmanager
    def locked_run(self, run_id: str):
        lock = self._lock_for(run_id)
        with lock:
            yield

    def create_run(self, goal: str, source_paths: list[str], parent_run_id: str | None = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}-{slugify(goal)}-{uuid4().hex[:6]}"
        with self.locked_run(run_id):
            run_dir = self.run_dir(run_id)
            (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
            (run_dir / "goal.md").write_text(goal + "\n", encoding="utf-8")
            self.write_json(run_id, "sources.json", {"source_paths": source_paths})
            self.write_json(run_id, "conversation.json", {"parent_run_id": parent_run_id})
            self.write_json(
                run_id,
                "state.json",
                {
                    "status": RunStatus.CREATED.value,
                    "goal": goal,
                    "done": [],
                    "facts": [],
                    "decisions": [],
                    "open_questions": [],
                    "next_action": "Planner will create a task graph.",
                    "refs": [],
                    "updated_at": utc_now(),
                },
            )
            (run_dir / "current_state.md").write_text(
                f"# Current State\n\nGoal: {goal}\n\nNext action: Planner will create a task graph.\n",
                encoding="utf-8",
            )
            (run_dir / "events.jsonl").touch()
            (run_dir / "index.jsonl").touch()
        LOG.info("created run run_id=%s source_count=%s", run_id, len(source_paths))
        return run_id

    def run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def read_json(self, run_id: str, name: str, default: Any = None) -> Any:
        with self.locked_run(run_id):
            path = self.run_dir(run_id) / name
            if not path.exists():
                return default
            return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, run_id: str, name: str, value: Any) -> None:
        with self.locked_run(run_id):
            path = self.run_dir(run_id) / name
            tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            tmp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(path)

    def write_plan(self, run_id: str, plan: Plan) -> None:
        self.write_json(run_id, "plan.json", plan.model_dump())

    def read_plan(self, run_id: str) -> Plan | None:
        raw = self.read_json(run_id, "plan.json")
        return Plan.model_validate(raw) if raw else None

    def next_event_id(self, run_id: str) -> str:
        with self.locked_run(run_id):
            path = self.run_dir(run_id) / "events.jsonl"
            count = 0
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    count = sum(1 for _ in handle)
            return f"evt_{count + 1:04d}"

    def append_event(
        self,
        run_id: str,
        *,
        type: str,
        title: str,
        summary: str,
        task_id: str | None = None,
        tool: str | None = None,
        status: str = "success",
        data: dict[str, Any] | None = None,
        artifact_ref: str | None = None,
    ) -> EventRecord:
        with self.locked_run(run_id):
            event = EventRecord(
                event_id=self.next_event_id(run_id),
                type=type,  # type: ignore[arg-type]
                title=title,
                summary=summary,
                task_id=task_id,
                tool=tool,
                status=status,  # type: ignore[arg-type]
                data=data or {},
                artifact_ref=artifact_ref,
                created_at=utc_now(),
            )
            with (self.run_dir(run_id) / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
            append_agent_event(run_id, event.model_dump())
            return event

    def read_events(self, run_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self.locked_run(run_id):
            path = self.run_dir(run_id) / "events.jsonl"
            if not path.exists():
                return []
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            return rows[-limit:] if limit else rows

    def write_artifact(self, run_id: str, name: str, content: str, suffix: str = ".txt") -> str:
        with self.locked_run(run_id):
            safe_name = slugify(name)
            artifact_name = f"{safe_name}-{uuid4().hex[:8]}{suffix}"
            path = self.run_dir(run_id) / "artifacts" / artifact_name
            path.write_text(content, encoding="utf-8")
            return f"artifacts/{artifact_name}"

    def append_index(self, run_id: str, record: dict[str, Any]) -> None:
        with self.locked_run(run_id):
            record.setdefault("created_at", utc_now())
            with (self.run_dir(run_id) / "index.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_index(self, run_id: str) -> list[dict[str, Any]]:
        with self.locked_run(run_id):
            path = self.run_dir(run_id) / "index.jsonl"
            if not path.exists():
                return []
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def read_artifact(self, run_id: str, ref: str, max_chars: int = 4000) -> str:
        with self.locked_run(run_id):
            artifact = ref.split("#", 1)[0]
            artifacts_root = (self.run_dir(run_id) / "artifacts").resolve()
            path = (self.run_dir(run_id) / artifact).resolve()
            if not is_relative_to(path, artifacts_root):
                raise ValueError(f"Invalid artifact ref: {ref}")
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[:max_chars]

    def update_state(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        with self.locked_run(run_id):
            state = self.read_json(run_id, "state.json", {})
            state.update(patch)
            state["updated_at"] = utc_now()
            self.write_json(run_id, "state.json", state)
            self._write_current_state_md(run_id, state)
            LOG.info("updated run state run_id=%s status=%s", run_id, state.get("status"))
            return state

    def _write_current_state_md(self, run_id: str, state: dict[str, Any]) -> None:
        lines = [
            "# Current State",
            "",
            f"Goal: {state.get('goal', '')}",
            f"Status: {state.get('status', '')}",
            "",
            "## Done",
            *[f"- {item}" for item in state.get("done", [])],
            "",
            "## Facts",
            *[f"- {item}" for item in state.get("facts", [])],
            "",
            "## Decisions",
            *[f"- {item}" for item in state.get("decisions", [])],
            "",
            "## Open Questions",
            *[f"- {item}" for item in state.get("open_questions", [])],
            "",
            f"## Next Action\n{state.get('next_action', '')}",
            "",
            "## Refs",
            *[f"- {item}" for item in state.get("refs", [])],
            "",
        ]
        (self.run_dir(run_id) / "current_state.md").write_text("\n".join(lines), encoding="utf-8")

    def snapshot(self, run_id: str) -> dict[str, Any]:
        with self.locked_run(run_id):
            run_dir = self.run_dir(run_id)
            final_path = run_dir / "final.md"
            return {
                "run_id": run_id,
                "state": self.read_json(run_id, "state.json", {}),
                "plan": self.read_json(run_id, "plan.json", None),
                "sources": self.read_json(run_id, "sources.json", {"source_paths": []}),
                "events": self.read_events(run_id),
                "index": self.read_index(run_id),
                "final": final_path.read_text(encoding="utf-8") if final_path.exists() else "",
            }

    def prior_run_payload(self, parent_run_id: str, final_max_chars: int = 20000) -> dict[str, Any]:
        parent_dir = self.run_dir(parent_run_id)
        if not parent_dir.exists():
            raise ValueError(f"Parent run not found: {parent_run_id}")
        parent_state = self.read_json(parent_run_id, "state.json", {})
        parent_plan = self.read_json(parent_run_id, "plan.json", None) or {}
        parent_final = parent_dir / "final.md"
        task_summaries = []
        for task in parent_plan.get("tasks", []):
            report = task.get("report") or {}
            task_summaries.append(
                {
                    "id": task.get("id"),
                    "title": task.get("title"),
                    "status": task.get("status"),
                    "summary": report.get("summary", ""),
                    "key_facts": report.get("key_facts", [])[-4:],
                    "refs": report.get("refs", [])[-6:],
                    "open_questions": report.get("open_questions", [])[-4:],
                }
            )
        return {
            "parent_run_id": parent_run_id,
            "goal": parent_state.get("goal", ""),
            "status": parent_state.get("status", ""),
            "done": parent_state.get("done", [])[-12:],
            "facts": parent_state.get("facts", [])[-12:],
            "refs": parent_state.get("refs", [])[-16:],
            "open_questions": parent_state.get("open_questions", [])[-8:],
            "tasks": task_summaries,
            "final_markdown_excerpt": parent_final.read_text(encoding="utf-8")[:final_max_chars] if parent_final.exists() else "",
        }

    def recent_runs(self, limit: int = 8, final_max_chars: int = 25000) -> list[dict[str, Any]]:
        run_dirs = [path for path in self.base_dir.iterdir() if path.is_dir()]
        run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        summaries: list[dict[str, Any]] = []
        for run_dir in run_dirs[:limit]:
            run_id = run_dir.name
            state = self.read_json(run_id, "state.json", {})
            plan = self.read_json(run_id, "plan.json", None) or {}
            final_path = run_dir / "final.md"
            index_path = run_dir / "index.jsonl"
            ref_count = 0
            if index_path.exists():
                ref_count = sum(1 for line in index_path.read_text(encoding="utf-8").splitlines() if line)
            summaries.append(
                {
                    "run_id": run_id,
                    "state": state,
                    "final": final_path.read_text(encoding="utf-8")[:final_max_chars] if final_path.exists() else "",
                    "task_count": len(plan.get("tasks", [])),
                    "ref_count": ref_count,
                }
            )
        return summaries

    def write_final(self, run_id: str, markdown: str) -> None:
        with self.locked_run(run_id):
            (self.run_dir(run_id) / "final.md").write_text(markdown, encoding="utf-8")
            LOG.info("wrote final brief run_id=%s chars=%s", run_id, len(markdown))
