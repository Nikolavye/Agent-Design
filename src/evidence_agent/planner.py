from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from evidence_agent.context import DEFAULT_CONTEXT_TOKEN_BUDGET, compact_planner_context
from evidence_agent.llm import LLMClient
from evidence_agent.tools.fs_tools import fs_list, fs_read
from evidence_agent.tools.registry import tool_manifest
from evidence_agent.types import AgentTask, Plan, ToolResult

Emit = Callable[..., Awaitable[None]]


PLANNER_REACT_SYSTEM = """You are the Planner node in a planner-executor research agent.
Your job is to create or revise a compact executable task DAG. You run as a visible ReAct agent,
but you do not execute the user's research task.

Response contract:
- Return exactly one JSON object. No markdown, no prose outside JSON, no arrays as the root.
- The top-level object must contain `thought_summary` and `action`.
- `thought_summary` must be a short operational summary, not hidden chain-of-thought.
- `action` must be exactly one of: `call_tool`, `complete_plan`, `block_plan`.
- Do not emit extra top-level keys beyond the selected schema.

Allowed response schemas:

1. Read-only tool call:
{
  "thought_summary": "why this read-only check is useful before planning",
  "action": "call_tool",
  "tool": "fs_list | fs_read",
  "args": {}
}

2. Complete plan:
{
  "thought_summary": "why the plan is ready",
  "action": "complete_plan",
  "plan": {
    "goal": "...",
    "assumptions": ["..."],
    "success_criteria": ["..."],
    "tasks": [
      {
        "id": "t1",
        "title": "...",
        "description": "...",
        "depends_on": [],
        "parallelizable": false,
        "produces_final": false,
        "requires": ["fs_list"],
        "acceptance_criteria": "..."
      }
    ]
  }
}

Task schema rules:
- `id`: stable short id such as `t1`, `t2`, `t3a`; ids must be unique.
- `title`: imperative phrase under 80 characters.
- `description`: executable instruction for one executor task.
- `depends_on`: ids that must be done first.
- `parallelizable`: true only when safe to run beside other ready tasks.
- `produces_final`: true for exactly one final synthesis task, false for every other task.
- `requires`: list of tool names the executor is likely to need.
- `acceptance_criteria`: observable completion condition.

3. Block planning:
{
  "thought_summary": "what is missing",
  "action": "block_plan",
  "open_questions": ["..."]
}

Planning policy:
- You may inspect local source inventory with `fs_list`; use `fs_read` only for small text docs, not full PDFs.
- Do not summarize papers, evaluate findings, or draft final content in the planner.
- If `conversation_context` is present, treat the current goal as a follow-up in the same chat session.
- Use `conversation_context` to resolve pronouns or short follow-ups such as "make it shorter", "explain this", or "what about limitations".
- Do not rerun prior work when `conversation_context` already contains enough prior evidence for the follow-up; instead plan a concise synthesis task that uses the prior context.
- In Plan Mode, only propose or revise the plan; execution starts only after user approval.
- If user feedback is provided, preserve useful existing tasks but revise titles, dependencies, tools, and acceptance criteria to match the feedback.
- If execution-time replan context is provided, preserve completed work and revise only blocked, failed, or pending work.
- In execution-time replan, do not remove done tasks; add new tasks only when they unblock the goal.
- If human interrupt recovery context is provided, follow the user's recovery instruction, preserve done work, and produce a revised plan for approval.
- Produce 4-7 tasks.
- Use dependencies to form a DAG.
- Mark independent source-processing tasks as parallelizable.
- Use available tool names in `requires`.
- If no local source is selected but the goal names a paper or URL, plan a web/PDF acquisition task using `browser_search`, `browser_fetch`, and `pdf_download` before `pdf_extract`.
- Include exactly one final synthesis task that depends on evidence retrieval.
- Set `produces_final: true` only on that final synthesis task; all other tasks must use false.
- Keep task descriptions executable by a ReAct executor.
- Prefer tasks that create verifiable refs before synthesis; final writing should depend on retrieval/memory tasks.
- If the goal is impossible without user input and tools cannot acquire the missing source, use `block_plan`.

Tool contracts:
- fs_list: {"root": "data", "glob": "**/*", "max_results": 50}
- fs_read: {"path": "...", "start_line": 1, "max_chars": 3000}

Example of a good executable DAG:
{
  "thought_summary": "Plan ready: acquire source, index it, run independent evidence retrieval tasks in parallel, then synthesize.",
  "action": "complete_plan",
  "plan": {
    "goal": "Create a cited evidence brief for a paper URL.",
    "assumptions": ["The paper URL exposes a downloadable PDF."],
    "success_criteria": ["PDF is indexed.", "Evidence refs support the final brief."],
    "tasks": [
      {
        "id": "t1",
        "title": "Download paper PDF",
        "description": "Download the paper PDF into data/pdfs.",
        "depends_on": [],
        "parallelizable": false,
        "produces_final": false,
        "requires": ["pdf_download"],
        "acceptance_criteria": "A local data/pdfs path is available."
      },
      {
        "id": "t2",
        "title": "Extract and index PDF",
        "description": "Extract the local PDF into searchable chunks.",
        "depends_on": ["t1"],
        "parallelizable": false,
        "produces_final": false,
        "requires": ["pdf_extract"],
        "acceptance_criteria": "PDF chunks are indexed with artifact refs."
      },
      {
        "id": "t3",
        "title": "Retrieve problem and motivation evidence",
        "description": "Search indexed chunks for the problem, motivation, and task framing.",
        "depends_on": ["t2"],
        "parallelizable": true,
        "produces_final": false,
        "requires": ["rag_search", "memory_get"],
        "acceptance_criteria": "Key problem/motivation facts have refs."
      },
      {
        "id": "t4",
        "title": "Retrieve method and result evidence",
        "description": "Search indexed chunks for methods, experiments, and main findings.",
        "depends_on": ["t2"],
        "parallelizable": true,
        "produces_final": false,
        "requires": ["rag_search", "memory_get"],
        "acceptance_criteria": "Method and result facts have refs."
      },
      {
        "id": "t5",
        "title": "Write final evidence brief",
        "description": "Synthesize retrieved evidence into a cited final brief.",
        "depends_on": ["t3", "t4"],
        "parallelizable": false,
        "produces_final": true,
        "requires": ["memory_search", "memory_get"],
        "acceptance_criteria": "Final markdown exists and cites artifact refs."
      }
    ]
  }
}

Now decide your next planning action and emit one JSON object.
"""


class Planner:
    def __init__(
        self,
        llm: LLMClient,
        step_delay_ms: int = 700,
        max_steps: int = 6,
        context_token_budget: int | None = DEFAULT_CONTEXT_TOKEN_BUDGET,
    ):
        self.llm = llm
        self.step_delay_ms = step_delay_ms
        self.max_steps = max_steps
        self.context_token_budget = context_token_budget

    async def pause(self) -> None:
        if self.step_delay_ms:
            await asyncio.sleep(self.step_delay_ms / 1000)

    async def create_plan(self, goal: str, source_paths: list[str], emit: Emit) -> Plan:
        return await self.create_plan_with_feedback(goal, source_paths, emit)

    async def create_plan_with_feedback(
        self,
        goal: str,
        source_paths: list[str],
        emit: Emit,
        *,
        feedback_history: list[str] | None = None,
        previous_plan: Plan | None = None,
        replan_context: dict[str, Any] | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> Plan:
        if not self.llm.enabled:
            raise RuntimeError("OPENAI_API_KEY is required for real agent mode.")

        observations: list[dict[str, Any]] = []
        for step in range(1, self.max_steps + 1):
            await emit(
                type="planner",
                title=f"Planner observe (step {step})",
                summary=self._observation_summary(goal, source_paths, observations),
                status="running",
                data={"observations": observations[-5:]},
            )
            await self.pause()

            action = self._next_action(
                goal,
                source_paths,
                observations,
                feedback_history or [],
                previous_plan,
                replan_context,
                conversation_context,
            )
            context_budget = action.pop("_context_budget", None)
            if context_budget and context_budget.get("compacted"):
                await emit(
                    type="state",
                    title="Planner context budget applied",
                    summary=(
                        "Planner context compacted "
                        f"{context_budget['estimated_tokens_before']} -> {context_budget['estimated_tokens_after']} "
                        f"tokens under {context_budget['budget_tokens']}."
                    ),
                    status="success",
                    data=context_budget,
                )
                await self.pause()
            await emit(
                type="planner",
                title=f"Planner ReAct decision (step {step})",
                summary=action.get("thought_summary", "Planner selected the next planning action."),
                status="running",
                data={"action": action},
            )
            await self.pause()

            action_type = action.get("action")
            if action_type == "call_tool":
                result = await self._call_tool(emit, action.get("tool"), action.get("args") or {})
                observations.append(
                    {
                        "step": step,
                        "tool": action.get("tool"),
                        "ok": result.ok,
                        "summary": result.summary,
                        "data_preview": self._preview_data(result.data),
                    }
                )
                continue

            if action_type == "complete_plan":
                external_done_ids = set()
                if replan_context:
                    external_done_ids = {
                        str(task.get("id"))
                        for task in replan_context.get("completed_tasks", [])
                        if task.get("id")
                    }
                plan = self._parse_plan(goal, action.get("plan") or {}, allowed_external_dependencies=external_done_ids)
                await emit(
                    type="planner",
                    title="Planner completed task DAG",
                    summary=f"Planner created {len(plan.tasks)} executable tasks.",
                    status="success",
                    data=plan.model_dump(),
                )
                await self.pause()
                return plan

            if action_type == "block_plan":
                questions = action.get("open_questions") or ["Planner blocked without a clear question."]
                raise RuntimeError(f"Planner blocked: {'; '.join(questions)}")

            raise RuntimeError(f"Planner returned an invalid action: {action}")

        raise RuntimeError(f"Planner reached max_steps={self.max_steps} without completing a plan.")

    def _next_action(
        self,
        goal: str,
        source_paths: list[str],
        observations: list[dict[str, Any]],
        feedback_history: list[str],
        previous_plan: Plan | None,
        replan_context: dict[str, Any] | None,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = {
            "goal": goal,
            "conversation_context": conversation_context,
            "selected_sources": source_paths,
            "available_tools": tool_manifest(),
            "planning_mode": {
                "requires_user_confirmation_before_execution": True,
                "feedback_history": feedback_history,
                "previous_plan": previous_plan.model_dump() if previous_plan else None,
                "replan_context": replan_context,
                "instruction": "If feedback exists, revise the previous plan to reflect it. Do not execute tasks.",
            },
            "recent_observations": observations[-5:],
        }
        context, context_budget = compact_planner_context(context, self.context_token_budget, llm=self.llm)
        if context_budget.get("overflow"):
            return {
                "thought_summary": (
                    "Planner context stayed above budget after compaction, so planning is blocked "
                    "instead of calling the model with an oversized prompt."
                ),
                "action": "block_plan",
                "_context_budget": context_budget,
                "open_questions": [
                    "Planner context exceeded the configured budget after compaction. Narrow the goal, reduce feedback/history, or raise EVIDENCE_AGENT_CONTEXT_TOKENS."
                ],
            }
        action = self.llm.json(PLANNER_REACT_SYSTEM, json.dumps(context, ensure_ascii=False, default=str), default_response={})
        if not action and self.llm.last_error:
            return {
                "thought_summary": "Planner could not receive a valid LLM response.",
                "action": "block_plan",
                "_context_budget": context_budget,
                "open_questions": [f"LLM call failed: {self.llm.last_error}"],
            }
        if action:
            action["_context_budget"] = context_budget
        return action

    async def _call_tool(self, emit: Emit, tool_name: str, args: dict[str, Any]) -> ToolResult:
        await emit(
            type="tool_call",
            title=f"Planner calls {tool_name}",
            summary="Planner performs a read-only context check before finalizing the task DAG.",
            tool=tool_name,
            status="running",
            data={"args": args},
        )
        await self.pause()
        try:
            result = self._dispatch_tool(tool_name, args)
        except Exception as exc:
            result = ToolResult(ok=False, summary=f"{tool_name} failed: {exc}", data={"args": args})
        await emit(
            type="tool_result" if result.ok else "error",
            title=f"Planner {tool_name}: {'ok' if result.ok else 'failed'}",
            summary=result.summary,
            tool=tool_name,
            status="success" if result.ok else "failed",
            data=result.data,
        )
        await self.pause()
        return result

    @staticmethod
    def _dispatch_tool(tool_name: str, args: dict[str, Any]) -> ToolResult:
        if tool_name == "fs_list":
            return fs_list(
                root=str(args.get("root", "data")),
                glob=str(args.get("glob", "**/*")),
                max_results=int(args.get("max_results", 50)),
            )
        if tool_name == "fs_read":
            return fs_read(
                path=str(args["path"]),
                start_line=int(args.get("start_line", 1)),
                max_chars=int(args.get("max_chars", 3000)),
            )
        raise ValueError(f"Planner cannot call tool: {tool_name}")

    @staticmethod
    def _parse_plan(
        goal: str,
        raw_plan: dict[str, Any],
        allowed_external_dependencies: set[str] | None = None,
    ) -> Plan:
        tasks = [AgentTask.model_validate(task) for task in raw_plan.get("tasks", [])]
        if not tasks:
            raise ValueError("Planner completed without tasks.")
        task_ids = [task.id for task in tasks]
        duplicate_ids = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
        if duplicate_ids:
            raise ValueError(f"Planner produced duplicate task ids: {', '.join(duplicate_ids)}")

        known_ids = set(task_ids) | (allowed_external_dependencies or set())
        for task in tasks:
            unknown_dependencies = [dependency for dependency in task.depends_on if dependency not in known_ids]
            if task.id in task.depends_on:
                raise ValueError(f"Task {task.id} cannot depend on itself.")
            if unknown_dependencies:
                raise ValueError(f"Task {task.id} depends on unknown tasks: {', '.join(unknown_dependencies)}")
        Planner._validate_acyclic(tasks, allowed_external_dependencies or set())

        final_tasks = [task for task in tasks if task.produces_final]
        if not final_tasks:
            markers = ("final", "synthesis", "synthesize", "brief", "report")
            inferred = [
                task
                for task in tasks
                if any(
                    marker in f"{task.title} {task.description} {task.acceptance_criteria}".lower()
                    for marker in markers
                )
            ]
            if inferred:
                inferred[-1].produces_final = True
                final_tasks = [inferred[-1]]
        if len(final_tasks) != 1:
            raise ValueError("Planner must produce exactly one final synthesis task with produces_final=true.")
        return Plan(
            goal=raw_plan.get("goal", goal),
            assumptions=raw_plan.get("assumptions", []),
            success_criteria=raw_plan.get("success_criteria", []),
            tasks=tasks,
        )

    @staticmethod
    def _validate_acyclic(tasks: list[AgentTask], external_dependencies: set[str] | None = None) -> None:
        by_id = {task.id: task for task in tasks}
        external_dependencies = external_dependencies or set()
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in external_dependencies:
                return
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError(f"Planner produced a cyclic dependency involving task {task_id}.")
            visiting.add(task_id)
            for dependency in by_id[task_id].depends_on:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task in tasks:
            visit(task.id)

    @staticmethod
    def _observation_summary(goal: str, source_paths: list[str], observations: list[dict[str, Any]]) -> str:
        if not observations:
            return f"Goal received. Selected sources: {len(source_paths)}. No planner observations yet."
        last = observations[-1]
        return f"Goal: {goal[:120]}. Last planner observation: {last.get('tool')} -> {last.get('summary')}"

    @staticmethod
    def _preview_data(data: dict[str, Any]) -> dict[str, Any]:
        preview: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                preview[key] = value[:500]
            elif isinstance(value, list):
                preview[key] = value[:8]
            else:
                preview[key] = value
        return preview


def ready_tasks(plan: Plan) -> list[AgentTask]:
    done = {task.id for task in plan.tasks if task.status == "done"}
    return [
        task
        for task in plan.tasks
        if task.status == "pending" and all(dependency in done for dependency in task.depends_on)
    ]
