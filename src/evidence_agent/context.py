from __future__ import annotations

import copy
import json
import os
from typing import Any

DEFAULT_CONTEXT_TOKEN_BUDGET = None
CHARS_PER_TOKEN = 4


def configured_context_token_budget() -> int | None:
    raw_value = os.getenv("EVIDENCE_AGENT_CONTEXT_TOKENS", "").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value > 0 else None

CONTEXT_SUMMARIZER_SYSTEM = """You are a context summarizer inside a planner-executor agent.
Return exactly one JSON object. No markdown and no prose outside JSON.

Your job is to compress prior context into a contextual summary for the next
agent decision. Preserve operationally important information; do not solve the
user's research task.

Required schema:
{
  "summary": "short summary of the prior context relevant to the next decision",
  "must_keep": ["facts, constraints, ids, dependencies, failures, or user requests that must not be lost"],
  "task_statuses": [{"id": "t1", "status": "done|pending|failed|blocked", "note": "..."}],
  "evidence_refs": ["artifacts/..."],
  "open_questions": ["..."]
}

Rules:
- Preserve task ids, dependency boundaries, failed/blocked reasons, and user feedback.
- Preserve artifact refs exactly when present.
- Do not invent refs, results, task ids, or claims.
- Prefer concise summaries over raw text.
"""


CONVERSATION_SUMMARIZER_SYSTEM = """You are compressing the previous run of a research agent so the next user question can be answered as a follow-up.
Return exactly one JSON object. No markdown and no prose outside JSON.

Required schema:
{
  "summary": "concise summary of the previous user question and answer",
  "must_keep": ["facts, constraints, user preferences, or conclusions that matter for follow-up questions"],
  "task_statuses": [{"id": "t1", "status": "done|pending|failed|blocked", "note": "..."}],
  "evidence_refs": ["artifacts/..."],
  "open_questions": ["..."]
}

Rules:
- Preserve what the user asked, what the prior run concluded, and any requested output style.
- Preserve artifact refs exactly when present.
- Keep the summary useful for follow-up questions such as "make it shorter", "explain the method", or "what about limitations".
- Do not invent refs, facts, task ids, or claims.
"""


def estimate_tokens(value: Any) -> int:
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, len(serialized) // CHARS_PER_TOKEN)


def summarize_prior_run_for_followup(
    current_goal: str,
    prior_run: dict[str, Any],
    llm: Any | None = None,
) -> dict[str, Any]:
    payload = {
        "current_user_goal": current_goal,
        "previous_run": prior_run,
    }
    if _llm_enabled(llm):
        summary = llm.json(
            CONVERSATION_SUMMARIZER_SYSTEM,
            json.dumps(payload, ensure_ascii=False, default=str),
            default_response={},
        )
        if isinstance(summary, dict) and summary.get("summary"):
            normalized = _normalize_contextual_summary(summary)
            normalized["source"] = "llm_contextual_summary"
            normalized["parent_run_id"] = prior_run.get("parent_run_id")
            normalized["previous_goal"] = str(prior_run.get("goal", ""))[:1200]
            return normalized

    fallback = {
        "summary": str(prior_run.get("final_markdown_excerpt") or prior_run.get("goal", ""))[:1600],
        "must_keep": [
            str(item)[:400]
            for item in [
                f"Previous user goal: {prior_run.get('goal', '')}",
                *prior_run.get("facts", [])[-6:],
            ]
            if item
        ][:12],
        "task_statuses": [
            {
                "id": task.get("id"),
                "status": task.get("status"),
                "note": str(task.get("summary", "") or task.get("title", ""))[:240],
            }
            for task in prior_run.get("tasks", [])[-12:]
            if isinstance(task, dict)
        ],
        "evidence_refs": [str(item) for item in prior_run.get("refs", [])[-16:]],
        "open_questions": [str(item)[:400] for item in prior_run.get("open_questions", [])[-8:]],
        "source": "deterministic_fallback",
        "parent_run_id": prior_run.get("parent_run_id"),
        "previous_goal": str(prior_run.get("goal", ""))[:1200],
    }
    return fallback


def compact_executor_context(
    context: dict[str, Any],
    budget_tokens: int | None = DEFAULT_CONTEXT_TOKEN_BUDGET,
    llm: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compacted = copy.deepcopy(context)
    before = estimate_tokens(compacted)
    operations: list[str] = []
    contextual_summary_used = False

    if budget_tokens is None:
        compacted["context_budget"] = {
            "mode": "model_default",
            "budget_tokens": None,
            "estimated_tokens_before": before,
            "estimated_tokens_after": before,
            "compacted": False,
            "overflow": False,
            "contextual_summary_used": False,
            "operations": [],
        }
        return compacted, compacted["context_budget"]

    if before > budget_tokens and _llm_enabled(llm):
        summary = _summarize_context_with_llm("executor", compacted, budget_tokens, llm)
        if summary:
            _apply_executor_summary(compacted, summary)
            operations.append("llm contextual summary generated for executor context")
            contextual_summary_used = True

    if before > budget_tokens:
        _trim_current_state(compacted, facts=4, refs=4)
        operations.append("current_state facts/refs trimmed to 4")

    if estimate_tokens(compacted) > budget_tokens:
        compacted["indexed_sources"] = compacted.get("indexed_sources", [])[-10:]
        operations.append("indexed_sources trimmed to 10")

    if estimate_tokens(compacted) > budget_tokens:
        compacted["recent_observations"] = [
            _compact_observation(observation, text_limit=220, list_limit=3)
            for observation in compacted.get("recent_observations", [])[-4:]
        ]
        operations.append("recent_observations trimmed to 4 with smaller previews")

    if estimate_tokens(compacted) > budget_tokens:
        compacted["known_public_urls"] = compacted.get("known_public_urls", [])[:3]
        operations.append("known_public_urls trimmed to 3")

    if estimate_tokens(compacted) > budget_tokens:
        _trim_current_state(compacted, facts=2, refs=2)
        compacted["indexed_sources"] = compacted.get("indexed_sources", [])[-5:]
        compacted["recent_observations"] = [
            _compact_observation(observation, text_limit=140, list_limit=2)
            for observation in compacted.get("recent_observations", [])[-2:]
        ]
        operations.append("context reduced to minimal rolling window")

    after = estimate_tokens(compacted)
    compacted["context_budget"] = {
        "budget_tokens": budget_tokens,
        "estimated_tokens_before": before,
        "estimated_tokens_after": after,
        "compacted": bool(operations),
        "overflow": after > budget_tokens,
        "contextual_summary_used": contextual_summary_used,
        "operations": operations,
    }
    return compacted, compacted["context_budget"]


def compact_planner_context(
    context: dict[str, Any],
    budget_tokens: int | None = DEFAULT_CONTEXT_TOKEN_BUDGET,
    llm: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compacted = copy.deepcopy(context)
    before = estimate_tokens(compacted)
    operations: list[str] = []
    contextual_summary_used = False

    if budget_tokens is None:
        compacted["context_budget"] = {
            "mode": "model_default",
            "budget_tokens": None,
            "estimated_tokens_before": before,
            "estimated_tokens_after": before,
            "compacted": False,
            "overflow": False,
            "contextual_summary_used": False,
            "operations": [],
        }
        return compacted, compacted["context_budget"]

    if before > budget_tokens and _llm_enabled(llm):
        summary = _summarize_context_with_llm("planner", compacted, budget_tokens, llm)
        if summary:
            _apply_planner_summary(compacted, summary)
            operations.append("llm contextual summary generated for planner context")
            contextual_summary_used = True

    if before > budget_tokens:
        planning_mode = compacted.get("planning_mode", {})
        if isinstance(planning_mode, dict):
            planning_mode["feedback_history"] = [
                str(item)[:800]
                for item in planning_mode.get("feedback_history", [])[-6:]
            ]
            if planning_mode.get("previous_plan"):
                planning_mode["previous_plan"] = _compact_plan(planning_mode["previous_plan"], report_limit=3)
            if planning_mode.get("replan_context"):
                planning_mode["replan_context"] = _compact_replan_context(planning_mode["replan_context"], report_limit=3)
        operations.append("planning_mode compacted to task-level summaries")

    if estimate_tokens(compacted) > budget_tokens:
        compacted["recent_observations"] = [
            _compact_observation(observation, text_limit=220, list_limit=3)
            for observation in compacted.get("recent_observations", [])[-3:]
        ]
        operations.append("planner observations trimmed to 3 with smaller previews")

    if estimate_tokens(compacted) > budget_tokens:
        planning_mode = compacted.get("planning_mode", {})
        if isinstance(planning_mode, dict):
            if planning_mode.get("previous_plan"):
                planning_mode["previous_plan"] = _compact_plan(planning_mode["previous_plan"], report_limit=1)
            if planning_mode.get("replan_context"):
                planning_mode["replan_context"] = _compact_replan_context(planning_mode["replan_context"], report_limit=1)
            planning_mode["feedback_history"] = [
                str(item)[:300]
                for item in planning_mode.get("feedback_history", [])[-3:]
            ]
        compacted["recent_observations"] = [
            _compact_observation(observation, text_limit=120, list_limit=2)
            for observation in compacted.get("recent_observations", [])[-2:]
        ]
        operations.append("planner context reduced to minimal task graph window")

    if estimate_tokens(compacted) > budget_tokens:
        planning_mode = compacted.get("planning_mode", {})
        if isinstance(planning_mode, dict):
            if planning_mode.get("previous_plan"):
                planning_mode["previous_plan"] = _compact_plan_skeleton(planning_mode["previous_plan"])
            if planning_mode.get("replan_context"):
                planning_mode["replan_context"] = _compact_replan_context_skeleton(planning_mode["replan_context"])
            planning_mode["feedback_history"] = [
                str(item)[:120]
                for item in planning_mode.get("feedback_history", [])[-2:]
            ]
        compacted["recent_observations"] = [
            _compact_observation(observation, text_limit=80, list_limit=1)
            for observation in compacted.get("recent_observations", [])[-1:]
        ]
        operations.append("planner context reduced to graph skeleton")

    after = estimate_tokens(compacted)
    compacted["context_budget"] = {
        "budget_tokens": budget_tokens,
        "estimated_tokens_before": before,
        "estimated_tokens_after": after,
        "compacted": bool(operations),
        "overflow": after > budget_tokens,
        "contextual_summary_used": contextual_summary_used,
        "operations": operations,
    }
    return compacted, compacted["context_budget"]


def _llm_enabled(llm: Any | None) -> bool:
    return bool(llm is not None and getattr(llm, "enabled", False))


def _summarize_context_with_llm(node: str, context: dict[str, Any], budget_tokens: int, llm: Any) -> dict[str, Any] | None:
    payload = {
        "node": node,
        "budget_tokens": budget_tokens,
        "summary_goal": (
            "Create a compact contextual summary for the next planner decision."
            if node == "planner"
            else "Create a compact contextual summary for the next executor ReAct decision."
        ),
        "context": _summary_input(node, context),
    }
    summary = llm.json(
        CONTEXT_SUMMARIZER_SYSTEM,
        json.dumps(payload, ensure_ascii=False, default=str),
        default_response={},
    )
    return summary if isinstance(summary, dict) and summary.get("summary") else None


def _summary_input(node: str, context: dict[str, Any]) -> dict[str, Any]:
    if node == "planner":
        planning_mode = context.get("planning_mode", {})
        if not isinstance(planning_mode, dict):
            planning_mode = {}
        return {
            "goal": context.get("goal", ""),
            "conversation_context": context.get("conversation_context"),
            "selected_sources": context.get("selected_sources", []),
            "available_tools": context.get("available_tools", []),
            "feedback_history": [str(item)[:1200] for item in planning_mode.get("feedback_history", [])[-10:]],
            "previous_plan": _compact_plan(planning_mode.get("previous_plan", {}), report_limit=2)
            if planning_mode.get("previous_plan")
            else None,
            "replan_context": _compact_replan_context(planning_mode.get("replan_context", {}), report_limit=2)
            if planning_mode.get("replan_context")
            else None,
            "recent_observations": [
                _compact_observation(observation, text_limit=500, list_limit=5)
                for observation in context.get("recent_observations", [])[-5:]
            ],
        }
    return {
        "goal": context.get("goal", ""),
        "conversation_context": context.get("conversation_context"),
        "task": _compact_task(context.get("task", {}), report_limit=2),
        "selected_sources": context.get("selected_sources", []),
        "current_state": context.get("current_state", {}),
        "indexed_sources": context.get("indexed_sources", [])[-20:],
        "recent_observations": [
            _compact_observation(observation, text_limit=600, list_limit=5)
            for observation in context.get("recent_observations", [])[-6:]
        ],
    }


def _apply_planner_summary(context: dict[str, Any], summary: dict[str, Any]) -> None:
    planning_mode = context.get("planning_mode", {})
    if not isinstance(planning_mode, dict):
        planning_mode = {}
        context["planning_mode"] = planning_mode
    planning_mode["contextual_summary"] = _normalize_contextual_summary(summary)
    if planning_mode.get("previous_plan"):
        planning_mode["previous_plan"] = _compact_plan_skeleton(planning_mode["previous_plan"])
    if planning_mode.get("replan_context"):
        planning_mode["replan_context"] = _compact_replan_context_skeleton(planning_mode["replan_context"])
    planning_mode["feedback_history"] = [str(item)[:180] for item in planning_mode.get("feedback_history", [])[-2:]]
    context["recent_observations"] = [
        _compact_observation(observation, text_limit=120, list_limit=2)
        for observation in context.get("recent_observations", [])[-2:]
    ]


def _apply_executor_summary(context: dict[str, Any], summary: dict[str, Any]) -> None:
    context["contextual_summary"] = _normalize_contextual_summary(summary)
    _trim_current_state(context, facts=2, refs=2)
    context["indexed_sources"] = context.get("indexed_sources", [])[-5:]
    context["recent_observations"] = [
        _compact_observation(observation, text_limit=160, list_limit=2)
        for observation in context.get("recent_observations", [])[-2:]
    ]
    context["known_public_urls"] = context.get("known_public_urls", [])[:3]


def _normalize_contextual_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": str(summary.get("summary", ""))[:1600],
        "must_keep": [str(item)[:400] for item in _as_list(summary.get("must_keep"))[:12]],
        "task_statuses": [
            _compact_mapping(item, text_limit=240, list_limit=4) if isinstance(item, dict) else {"note": str(item)[:240]}
            for item in _as_list(summary.get("task_statuses"))[:12]
        ],
        "evidence_refs": [str(item) for item in _as_list(summary.get("evidence_refs"))[:16]],
        "open_questions": [str(item)[:400] for item in _as_list(summary.get("open_questions"))[:8]],
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _trim_current_state(context: dict[str, Any], *, facts: int, refs: int) -> None:
    state = context.get("current_state")
    if not isinstance(state, dict):
        return
    state["facts"] = state.get("facts", [])[-facts:]
    state["refs"] = state.get("refs", [])[-refs:]


def _compact_observation(observation: dict[str, Any], *, text_limit: int, list_limit: int) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in observation.items():
        if key == "data_preview" and isinstance(value, dict):
            compacted[key] = _compact_preview(value, text_limit=text_limit, list_limit=list_limit)
        elif isinstance(value, str):
            compacted[key] = value[:text_limit]
        elif isinstance(value, list):
            compacted[key] = value[:list_limit]
        else:
            compacted[key] = value
    return compacted


def _compact_plan(plan: dict[str, Any], *, report_limit: int) -> dict[str, Any]:
    tasks = plan.get("tasks", []) if isinstance(plan, dict) else []
    return {
        "goal": plan.get("goal", "") if isinstance(plan, dict) else "",
        "version": plan.get("version", 1) if isinstance(plan, dict) else 1,
        "assumptions": (plan.get("assumptions", []) if isinstance(plan, dict) else [])[-4:],
        "success_criteria": (plan.get("success_criteria", []) if isinstance(plan, dict) else [])[-4:],
        "tasks": [_compact_task(task, report_limit=report_limit) for task in tasks],
    }


def _compact_plan_skeleton(plan: dict[str, Any]) -> dict[str, Any]:
    tasks = plan.get("tasks", []) if isinstance(plan, dict) else []
    return {
        "goal": plan.get("goal", "") if isinstance(plan, dict) else "",
        "version": plan.get("version", 1) if isinstance(plan, dict) else 1,
        "tasks": [_compact_task_skeleton(task) for task in tasks],
    }


def _compact_replan_context(context: dict[str, Any], *, report_limit: int) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    current_state = context.get("current_state", {})
    return {
        "mode": context.get("mode"),
        "reason": _compact_mapping(context.get("reason", {}), text_limit=500, list_limit=4),
        "current_plan": _compact_plan(context.get("current_plan", {}), report_limit=report_limit),
        "completed_tasks": [_compact_task(task, report_limit=report_limit) for task in context.get("completed_tasks", [])],
        "blocked_or_failed_tasks": [
            _compact_task(task, report_limit=report_limit)
            for task in context.get("blocked_or_failed_tasks", [])
        ],
        "pending_tasks": [_compact_task(task, report_limit=report_limit) for task in context.get("pending_tasks", [])],
        "latest_reports": [_compact_report(report, report_limit=report_limit) for report in context.get("latest_reports", [])],
        "current_state": {
            "done": current_state.get("done", [])[-8:] if isinstance(current_state, dict) else [],
            "facts": current_state.get("facts", [])[-4:] if isinstance(current_state, dict) else [],
            "refs": current_state.get("refs", [])[-4:] if isinstance(current_state, dict) else [],
            "open_questions": current_state.get("open_questions", [])[-4:] if isinstance(current_state, dict) else [],
            "next_action": current_state.get("next_action", "") if isinstance(current_state, dict) else "",
        },
        "rules": context.get("rules", [])[-8:],
    }


def _compact_replan_context_skeleton(context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    current_state = context.get("current_state", {})
    return {
        "mode": context.get("mode"),
        "reason": _compact_mapping(context.get("reason", {}), text_limit=180, list_limit=2),
        "current_plan": _compact_plan_skeleton(context.get("current_plan", {})),
        "completed_tasks": [_compact_task_skeleton(task) for task in context.get("completed_tasks", [])],
        "blocked_or_failed_tasks": [
            _compact_task(task, report_limit=1)
            for task in context.get("blocked_or_failed_tasks", [])
        ],
        "pending_tasks": [_compact_task_skeleton(task) for task in context.get("pending_tasks", [])],
        "latest_reports": [_compact_report(report, report_limit=1) for report in context.get("latest_reports", [])[-2:]],
        "current_state": {
            "done": current_state.get("done", [])[-6:] if isinstance(current_state, dict) else [],
            "facts": current_state.get("facts", [])[-2:] if isinstance(current_state, dict) else [],
            "refs": current_state.get("refs", [])[-2:] if isinstance(current_state, dict) else [],
            "open_questions": current_state.get("open_questions", [])[-2:] if isinstance(current_state, dict) else [],
            "next_action": current_state.get("next_action", "") if isinstance(current_state, dict) else "",
        },
        "rules": context.get("rules", [])[-4:],
    }


def _compact_task(task: dict[str, Any], *, report_limit: int) -> dict[str, Any]:
    if not isinstance(task, dict):
        return {}
    compacted = {
        "id": task.get("id"),
        "title": str(task.get("title", ""))[:160],
        "description": str(task.get("description", ""))[:400],
        "depends_on": task.get("depends_on", []),
        "parallelizable": task.get("parallelizable", False),
        "produces_final": task.get("produces_final", False),
        "requires": task.get("requires", []),
        "acceptance_criteria": str(task.get("acceptance_criteria", ""))[:300],
        "status": task.get("status"),
    }
    if task.get("report"):
        compacted["report"] = _compact_report(task["report"], report_limit=report_limit)
    return compacted


def _compact_task_skeleton(task: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(task, dict):
        return {}
    return {
        "id": task.get("id"),
        "title": str(task.get("title", ""))[:100],
        "depends_on": task.get("depends_on", []),
        "parallelizable": task.get("parallelizable", False),
        "produces_final": task.get("produces_final", False),
        "requires": task.get("requires", []),
        "status": task.get("status"),
    }


def _compact_report(report: dict[str, Any], *, report_limit: int) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    key_facts = report.get("key_facts", [])[-report_limit:] if report_limit > 0 else []
    refs = report.get("refs", [])[-report_limit:] if report_limit > 0 else []
    open_questions = report.get("open_questions", [])[-report_limit:] if report_limit > 0 else []
    recommended_next_tasks = report.get("recommended_next_tasks", [])[-report_limit:] if report_limit > 0 else []
    return {
        "task_id": report.get("task_id"),
        "status": report.get("status"),
        "summary": str(report.get("summary", ""))[:500],
        "key_facts": [
            {
                "claim": str(fact.get("claim", ""))[:260],
                "ref": fact.get("ref", ""),
            }
            for fact in key_facts
            if isinstance(fact, dict)
        ],
        "refs": refs,
        "open_questions": [str(item)[:260] for item in open_questions],
        "recommended_next_tasks": [
            str(item)[:200]
            for item in recommended_next_tasks
        ],
    }


def _compact_mapping(value: dict[str, Any], *, text_limit: int, list_limit: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compacted: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, str):
            compacted[key] = item[:text_limit]
        elif isinstance(item, list):
            compacted[key] = item[-list_limit:]
        elif isinstance(item, dict):
            compacted[key] = _compact_mapping(item, text_limit=text_limit, list_limit=list_limit)
        else:
            compacted[key] = item
    return compacted


def _compact_preview(preview: dict[str, Any], *, text_limit: int, list_limit: int) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in preview.items():
        if isinstance(value, str):
            compacted[key] = value[:text_limit]
        elif isinstance(value, list):
            compacted[key] = value[:list_limit]
        else:
            compacted[key] = value
    return compacted
