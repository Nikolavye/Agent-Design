from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskStatus = Literal["pending", "running", "done", "blocked", "failed"]
RunStatusValue = Literal[
    "created",
    "planning",
    "awaiting_approval",
    "running",
    "replanning",
    "done",
    "blocked",
    "failed",
]
EventType = Literal[
    "planner",
    "orchestrator",
    "executor",
    "tool_call",
    "tool_result",
    "state",
    "final",
    "error",
]


class RunStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    REPLANNING = "replanning"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"


class RunState(BaseModel):
    status: RunStatus = RunStatus.CREATED
    goal: str = ""
    done: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_action: str = ""
    refs: list[str] = Field(default_factory=list)
    planning_feedback: list[str] = Field(default_factory=list)
    replan_count: int = 0
    updated_at: str = ""

    def can_accept_feedback(self) -> bool:
        return self.status == RunStatus.AWAITING_APPROVAL

    def can_approve(self) -> bool:
        return self.status == RunStatus.AWAITING_APPROVAL

    def can_recover_with_feedback(self) -> bool:
        return self.status in {RunStatus.BLOCKED, RunStatus.FAILED}


class SourceFile(BaseModel):
    name: str
    path: str
    kind: Literal["pdf", "text", "other"]
    size_bytes: int


class AgentTask(BaseModel):
    id: str
    title: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    parallelizable: bool = False
    produces_final: bool = False
    requires: list[str] = Field(default_factory=list)
    acceptance_criteria: str = ""
    status: TaskStatus = "pending"
    assigned_to: str | None = None
    report: dict[str, Any] | None = None


class Plan(BaseModel):
    goal: str
    version: int = 1
    tasks: list[AgentTask]
    assumptions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    goal: str = Field(..., min_length=5)
    source_paths: list[str] = Field(default_factory=list)
    parent_run_id: str | None = None
    max_parallel: int | None = Field(default=None, ge=1)
    step_delay_ms: int = Field(default=700, ge=0, le=3000)


class PlanRequest(BaseModel):
    goal: str = Field(..., min_length=5)
    source_paths: list[str] = Field(default_factory=list)
    parent_run_id: str | None = None
    step_delay_ms: int = Field(default=700, ge=0, le=3000)


class PlanFeedbackRequest(BaseModel):
    feedback: str = Field(..., min_length=2)
    step_delay_ms: int = Field(default=700, ge=0, le=3000)


class ManualReplanRequest(BaseModel):
    feedback: str = Field(..., min_length=2)
    step_delay_ms: int = Field(default=700, ge=0, le=3000)


class ApprovePlanRequest(BaseModel):
    max_parallel: int | None = Field(default=None, ge=1)
    step_delay_ms: int = Field(default=700, ge=0, le=3000)


class ToolResult(BaseModel):
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    artifact_ref: str | None = None


class EventRecord(BaseModel):
    event_id: str
    type: EventType
    title: str
    summary: str
    task_id: str | None = None
    tool: str | None = None
    status: Literal["pending", "running", "success", "failed"] = "success"
    data: dict[str, Any] = Field(default_factory=dict)
    artifact_ref: str | None = None
    created_at: str


class TaskReport(BaseModel):
    task_id: str
    status: TaskStatus
    summary: str
    key_facts: list[dict[str, str]] = Field(default_factory=list)
    refs: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recommended_next_tasks: list[str] = Field(default_factory=list)


class RunSnapshot(BaseModel):
    run_id: str
    state: dict[str, Any]
    plan: dict[str, Any] | None = None
    sources: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    index: list[dict[str, Any]] = Field(default_factory=list)
    final: str = ""


class RunHistoryItem(BaseModel):
    run_id: str
    state: dict[str, Any]
    final: str = ""
    task_count: int = 0
    ref_count: int = 0


class RunHistoryResponse(BaseModel):
    runs: list[RunHistoryItem] = Field(default_factory=list)
