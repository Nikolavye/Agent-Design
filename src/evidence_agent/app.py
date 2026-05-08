from __future__ import annotations

import asyncio
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from evidence_agent.context import configured_context_token_budget
from evidence_agent.loop import Orchestrator
from evidence_agent.observability import logger, setup_logging
from evidence_agent.state.run_store import PDF_DIR, PROJECT_ROOT, RunStore, is_relative_to, slugify
from evidence_agent.tools.fs_tools import list_sources
from evidence_agent.tools.registry import tool_manifest
from evidence_agent.types import (
    ApprovePlanRequest,
    ManualReplanRequest,
    PlanFeedbackRequest,
    PlanRequest,
    RunHistoryResponse,
    RunRequest,
    RunSnapshot,
    RunState,
    RunStatus,
)

DEFAULT_MODEL = "gpt-5.1"
DEFAULT_BACKGROUND_WORKERS = 4
DEFAULT_PROMPT = (
    "Read this paper: https://arxiv.org/abs/2601.06700. "
    "Summarize what the paper is doing, its main results and findings, "
    "and why it is valuable. Produce a structured evidence brief with "
    "citations for key factual claims."
)


def load_local_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        os.environ.setdefault("EVIDENCE_AGENT_MODEL", DEFAULT_MODEL)
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"OPENAI_API_KEY", "EVIDENCE_AGENT_MODEL"} and value:
            os.environ[key] = value
    os.environ.setdefault("EVIDENCE_AGENT_MODEL", DEFAULT_MODEL)


load_local_env()
setup_logging()
LOG = logger(__name__)

BACKGROUND_WORKERS = int(os.getenv("EVIDENCE_AGENT_WORKERS", str(DEFAULT_BACKGROUND_WORKERS)))


def create_background_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=BACKGROUND_WORKERS,
        thread_name_prefix="agent-run",
    )


background_executor = create_background_executor()
background_tasks_lock = threading.Lock()
pdf_upload_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global background_executor
    if getattr(background_executor, "_shutdown", False):
        background_executor = create_background_executor()
    with background_tasks_lock:
        app.state.background_tasks = set()
    try:
        yield
    finally:
        background_executor.shutdown(wait=True, cancel_futures=True)
        with background_tasks_lock:
            app.state.background_tasks = set()


app = FastAPI(title="Evidence Brief Agent", version="0.1.0", lifespan=lifespan)
store = RunStore()
orchestrator = Orchestrator(store)

STATIC_DIR = PROJECT_ROOT / "src" / "evidence_agent" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
PDF_UPLOAD = File(...)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    LOG.info(
        "http request method=%s path=%s status=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


async def _run_background(run_id: str, coro):
    try:
        LOG.info("background run started run_id=%s", run_id)
        await coro
        LOG.info("background run completed run_id=%s", run_id)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        LOG.exception("background run crashed run_id=%s error=%s", run_id, message)
        store.update_state(
            run_id,
            {
                "status": RunStatus.FAILED.value,
                "open_questions": [f"Background task crashed: {message}"],
                "next_action": "Inspect events.jsonl and restart the run after fixing the crash.",
            },
        )
        store.append_event(
            run_id,
            type="error",
            title="Background task crashed",
            summary=message,
            status="failed",
        )


def _run_background_thread(run_id: str, coro) -> None:
    asyncio.run(_run_background(run_id, coro))


def start_background_task(run_id: str, coro) -> Future:
    LOG.info("background run queued run_id=%s", run_id)
    future = background_executor.submit(_run_background_thread, run_id, coro)
    with background_tasks_lock:
        tasks = getattr(app.state, "background_tasks", None)
        if tasks is None:
            tasks = set()
            app.state.background_tasks = tasks
        tasks.add(future)

    def discard_task(done: Future) -> None:
        with background_tasks_lock:
            tasks = getattr(app.state, "background_tasks", None)
            if tasks is not None:
                tasks.discard(done)

    future.add_done_callback(discard_task)
    return future


def unique_pdf_upload_path(filename: str) -> Path:
    stem = slugify(Path(filename).stem)
    candidate = PDF_DIR / f"{stem}.pdf"
    while candidate.exists():
        candidate = PDF_DIR / f"{stem}-{uuid4().hex[:6]}.pdf"
    return candidate


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def config():
    return {
        "tools": tool_manifest(),
        "project_root": str(PROJECT_ROOT),
        "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("EVIDENCE_AGENT_MODEL", DEFAULT_MODEL),
        "background_workers": BACKGROUND_WORKERS,
        "context_token_budget": configured_context_token_budget(),
        "context_window": "model_default" if configured_context_token_budget() is None else "custom_budget",
        "sample_goal": DEFAULT_PROMPT,
    }


@app.get("/api/sources")
def sources():
    return {"sources": [source.model_dump() for source in list_sources()]}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = PDF_UPLOAD):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported in this demo.")
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    with pdf_upload_lock:
        destination = unique_pdf_upload_path(file.filename)
        with destination.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    LOG.info("uploaded pdf name=%s path=%s", file.filename, destination.relative_to(PROJECT_ROOT))
    return {"path": str(destination.relative_to(PROJECT_ROOT)), "name": destination.name}


@app.post("/api/runs")
async def create_run(request: RunRequest):
    run_id = store.create_run(request.goal, request.source_paths, parent_run_id=request.parent_run_id)
    LOG.info(
        "api create run run_id=%s mode=direct source_count=%s parent_run_id=%s",
        run_id,
        len(request.source_paths),
        request.parent_run_id,
    )
    start_background_task(
        run_id,
        orchestrator.run(
            run_id,
            request.goal,
            request.source_paths,
            request.max_parallel,
            request.step_delay_ms,
        )
    )
    return {"run_id": run_id}


@app.post("/api/plans")
async def create_plan(request: PlanRequest):
    run_id = store.create_run(request.goal, request.source_paths, parent_run_id=request.parent_run_id)
    LOG.info(
        "api create plan run_id=%s source_count=%s parent_run_id=%s",
        run_id,
        len(request.source_paths),
        request.parent_run_id,
    )
    start_background_task(
        run_id,
        orchestrator.plan_for_review(
            run_id,
            request.goal,
            request.source_paths,
            step_delay_ms=request.step_delay_ms,
        )
    )
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/plan-feedback")
async def revise_plan(run_id: str, request: PlanFeedbackRequest):
    run_dir = store.run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    with store.locked_run(run_id):
        state = RunState.model_validate(store.read_json(run_id, "state.json", {}))
        if not state.can_accept_feedback():
            raise HTTPException(status_code=409, detail="Plan feedback is only accepted while awaiting approval.")
        source_payload = store.read_json(run_id, "sources.json", {"source_paths": []})
        store.update_state(run_id, {"status": RunStatus.PLANNING.value, "next_action": "Planner is revising the plan."})
    start_background_task(
        run_id,
        orchestrator.plan_for_review(
            run_id,
            state.goal,
            source_payload.get("source_paths", []),
            step_delay_ms=request.step_delay_ms,
            feedback=request.feedback,
        )
    )
    LOG.info("api revise plan run_id=%s", run_id)
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/manual-replan")
async def manual_replan(run_id: str, request: ManualReplanRequest):
    run_dir = store.run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    with store.locked_run(run_id):
        state = RunState.model_validate(store.read_json(run_id, "state.json", {}))
        if not state.can_recover_with_feedback():
            raise HTTPException(status_code=409, detail="Manual recovery is only accepted after blocked or failed runs.")
        store.update_state(
            run_id,
            {
                "status": RunStatus.PLANNING.value,
                "next_action": "Planner is revising the run from human recovery feedback.",
            },
        )
    start_background_task(
        run_id,
        orchestrator.plan_after_human_interrupt(
            run_id,
            request.feedback,
            step_delay_ms=request.step_delay_ms,
        )
    )
    LOG.info("api manual replan run_id=%s", run_id)
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/approve")
async def approve_plan(run_id: str, request: ApprovePlanRequest):
    run_dir = store.run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    with store.locked_run(run_id):
        state = RunState.model_validate(store.read_json(run_id, "state.json", {}))
        if not state.can_approve():
            raise HTTPException(status_code=409, detail="Run is not waiting for plan approval.")
        store.update_state(run_id, {"status": RunStatus.RUNNING.value, "next_action": "Approved plan; execution is starting."})
    start_background_task(
        run_id,
        orchestrator.execute_existing_plan(
            run_id,
            max_parallel=request.max_parallel,
            step_delay_ms=request.step_delay_ms,
        )
    )
    LOG.info("api approve plan run_id=%s max_parallel=%s", run_id, request.max_parallel or "dynamic")
    return {"run_id": run_id}


@app.get("/api/runs/recent", response_model=RunHistoryResponse)
def recent_runs(limit: int = 8) -> RunHistoryResponse:
    bounded_limit = max(1, min(limit, 20))
    return RunHistoryResponse.model_validate({"runs": store.recent_runs(limit=bounded_limit)})


@app.get("/api/runs/{run_id}", response_model=RunSnapshot)
def get_run(run_id: str) -> RunSnapshot:
    run_dir = store.run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return RunSnapshot.model_validate(store.snapshot(run_id))


@app.get("/api/runs/{run_id}/artifact")
def get_artifact(run_id: str, ref: str):
    run_dir = store.run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    artifacts_root = (run_dir / "artifacts").resolve()
    path = (run_dir / ref.split("#", 1)[0]).resolve()
    if not is_relative_to(path, artifacts_root):
        raise HTTPException(status_code=400, detail="Invalid artifact ref")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)
