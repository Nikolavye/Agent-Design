from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from evidence_agent import app as app_module
from evidence_agent.loop import Orchestrator
from evidence_agent.state.run_store import RunStore
from evidence_agent.types import RunStatus


@pytest.fixture()
def isolated_client(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVIDENCE_AGENT_MODEL", raising=False)
    test_store = RunStore(base_dir=tmp_path / "runs")
    monkeypatch.setattr(app_module, "store", test_store)
    monkeypatch.setattr(app_module, "orchestrator", Orchestrator(test_store))
    app_module.app.state.background_tasks = set()
    with TestClient(app_module.app) as client:
        yield client, test_store


def _wait_for_terminal_state(client: TestClient, run_id: str, timeout_s: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout_s
    snapshot = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        response.raise_for_status()
        snapshot = response.json()
        if snapshot["state"].get("status") in {RunStatus.DONE.value, RunStatus.BLOCKED.value, RunStatus.FAILED.value}:
            return snapshot
        time.sleep(0.05)
    return snapshot


def test_config_exposes_runtime_contract(isolated_client) -> None:
    client, _store = isolated_client

    response = client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    tool_names = {tool["name"] for tool in payload["tools"]}
    assert "pdf_download" in tool_names
    assert "fs_read" in tool_names
    assert "llm_configured" in payload
    assert payload["model"] == "gpt-5.1"
    assert payload["context_token_budget"] is None
    assert payload["context_window"] == "model_default"


def test_no_key_plan_fails_with_actionable_state(isolated_client, monkeypatch) -> None:
    client, _store = isolated_client
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.post(
        "/api/plans",
        json={"goal": "Write a short evidence brief from the selected sources.", "source_paths": [], "step_delay_ms": 0},
    )
    assert response.status_code == 200

    snapshot = _wait_for_terminal_state(client, response.json()["run_id"])
    assert snapshot["state"]["status"] == RunStatus.FAILED.value
    assert "GPT-5 runtime" in " ".join(snapshot["state"]["open_questions"])


def test_plan_feedback_requires_awaiting_approval(isolated_client, monkeypatch) -> None:
    client, store = isolated_client
    run_id = store.create_run("Write a final brief.", [])

    response = client.post(f"/api/runs/{run_id}/plan-feedback", json={"feedback": "Make it shorter.", "step_delay_ms": 0})
    assert response.status_code == 409

    store.update_state(run_id, {"status": RunStatus.AWAITING_APPROVAL.value})
    started = []

    def fake_start_background_task(run_id: str, coro):
        started.append(run_id)
        coro.close()
        return None

    monkeypatch.setattr(app_module, "start_background_task", fake_start_background_task)
    accepted = client.post(f"/api/runs/{run_id}/plan-feedback", json={"feedback": "Make it shorter.", "step_delay_ms": 0})
    rejected = client.post(f"/api/runs/{run_id}/plan-feedback", json={"feedback": "Again.", "step_delay_ms": 0})

    assert accepted.status_code == 200
    assert rejected.status_code == 409
    assert started == [run_id]
    assert store.read_json(run_id, "state.json")["status"] == RunStatus.PLANNING.value


def test_manual_replan_requires_terminal_blocked_or_failed(isolated_client, monkeypatch) -> None:
    client, store = isolated_client
    run_id = store.create_run("Write a final brief.", [])

    response = client.post(
        f"/api/runs/{run_id}/manual-replan",
        json={"feedback": "Retry with a narrower retrieval task.", "step_delay_ms": 0},
    )
    assert response.status_code == 409

    store.update_state(run_id, {"status": RunStatus.BLOCKED.value})
    started = []

    def fake_start_background_task(run_id: str, coro):
        started.append(run_id)
        coro.close()
        return None

    monkeypatch.setattr(app_module, "start_background_task", fake_start_background_task)
    accepted = client.post(
        f"/api/runs/{run_id}/manual-replan",
        json={"feedback": "Retry with memory_get instead of fs_read.", "step_delay_ms": 0},
    )
    rejected = client.post(
        f"/api/runs/{run_id}/manual-replan",
        json={"feedback": "Try again.", "step_delay_ms": 0},
    )

    assert accepted.status_code == 200
    assert rejected.status_code == 409
    assert started == [run_id]
    assert store.read_json(run_id, "state.json")["status"] == RunStatus.PLANNING.value


def test_approve_requires_awaiting_approval(isolated_client, monkeypatch) -> None:
    client, store = isolated_client
    run_id = store.create_run("Write a final brief.", [])

    response = client.post(f"/api/runs/{run_id}/approve", json={"max_parallel": 2, "step_delay_ms": 0})
    assert response.status_code == 409

    store.update_state(run_id, {"status": RunStatus.AWAITING_APPROVAL.value})
    started = []

    def fake_start_background_task(run_id: str, coro):
        started.append(run_id)
        coro.close()
        return None

    monkeypatch.setattr(app_module, "start_background_task", fake_start_background_task)
    accepted = client.post(f"/api/runs/{run_id}/approve", json={"max_parallel": 2, "step_delay_ms": 0})
    rejected = client.post(f"/api/runs/{run_id}/approve", json={"max_parallel": 2, "step_delay_ms": 0})

    assert accepted.status_code == 200
    assert rejected.status_code == 409
    assert started == [run_id]
    assert store.read_json(run_id, "state.json")["status"] == RunStatus.RUNNING.value


def test_concurrent_approve_only_starts_one_executor(isolated_client, monkeypatch) -> None:
    client, store = isolated_client
    run_id = store.create_run("Write a final brief.", [])
    store.update_state(run_id, {"status": RunStatus.AWAITING_APPROVAL.value})
    started = []

    def fake_start_background_task(run_id: str, coro):
        started.append(run_id)
        coro.close()
        return None

    monkeypatch.setattr(app_module, "start_background_task", fake_start_background_task)

    def approve_once() -> int:
        response = client.post(f"/api/runs/{run_id}/approve", json={"max_parallel": 2, "step_delay_ms": 0})
        return response.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda _idx: approve_once(), range(2)))

    assert sorted(statuses) == [200, 409]
    assert started == [run_id]


def test_recent_runs_lists_persisted_history(isolated_client) -> None:
    client, store = isolated_client
    run_id = store.create_run("Write a final brief.", [])
    store.update_state(run_id, {"status": RunStatus.DONE.value})
    store.write_json(run_id, "plan.json", {"tasks": [{"id": "t1"}, {"id": "t2"}]})
    store.append_index(run_id, {"kind": "pdf_chunk", "source": "paper.pdf", "ref": "artifacts/chunk.txt"})
    store.write_final(run_id, "Final answer [ref: artifacts/chunk.txt]")

    response = client.get("/api/runs/recent", params={"limit": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["runs"][0]["run_id"] == run_id
    assert payload["runs"][0]["state"]["status"] == RunStatus.DONE.value
    assert payload["runs"][0]["task_count"] == 2
    assert payload["runs"][0]["ref_count"] == 1
    assert "Final answer" in payload["runs"][0]["final"]


def test_artifact_path_traversal_is_rejected(isolated_client) -> None:
    client, store = isolated_client
    run_id = store.create_run("Write a final brief.", [])

    response = client.get(f"/api/runs/{run_id}/artifact", params={"ref": "../state.json"})

    assert response.status_code == 400
    with pytest.raises(ValueError):
        store.read_artifact(run_id, "../state.json")
