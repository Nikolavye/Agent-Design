from __future__ import annotations

import asyncio
from typing import Any

from evidence_agent.context import summarize_prior_run_for_followup
from evidence_agent.executor import Executor
from evidence_agent.llm import LLMClient
from evidence_agent.planner import Planner, ready_tasks
from evidence_agent.state.run_store import RunStore
from evidence_agent.types import AgentTask, Plan, RunStatus, TaskReport


class Orchestrator:
    def __init__(self, store: RunStore):
        self.store = store

    async def emit(self, run_id: str, **kwargs: Any) -> None:
        self.store.append_event(run_id, **kwargs)

    async def run(
        self,
        run_id: str,
        goal: str,
        source_paths: list[str],
        max_parallel: int | None = None,
        step_delay_ms: int = 700,
    ) -> None:
        await self.plan_for_review(run_id, goal, source_paths, step_delay_ms=step_delay_ms)
        state = self.store.read_json(run_id, "state.json", {})
        if state.get("status") == RunStatus.AWAITING_APPROVAL.value:
            await self.execute_existing_plan(run_id, max_parallel=max_parallel, step_delay_ms=step_delay_ms)

    async def plan_for_review(
        self,
        run_id: str,
        goal: str,
        source_paths: list[str],
        step_delay_ms: int = 700,
        feedback: str | None = None,
    ) -> None:
        llm = LLMClient(enabled=True)
        planner = Planner(llm, step_delay_ms=step_delay_ms)
        conversation_context = await self._load_or_build_conversation_context(run_id, goal, llm, step_delay_ms)
        state = self.store.read_json(run_id, "state.json", {})
        feedback_history = list(state.get("planning_feedback", []))
        if feedback:
            feedback_history.append(feedback)
            await self.emit(
                run_id,
                type="planner",
                title="User feedback received",
                summary=feedback,
                status="success",
                data={"feedback_history": feedback_history},
            )
        previous_plan = self.store.read_plan(run_id)
        self.store.update_state(
            run_id,
            {
                "status": RunStatus.PLANNING.value,
                "planning_feedback": feedback_history,
                "next_action": "Planner is preparing a plan for user review.",
            },
        )
        try:
            planner_kwargs: dict[str, Any] = {
                "feedback_history": feedback_history,
                "previous_plan": previous_plan,
            }
            if conversation_context:
                planner_kwargs["conversation_context"] = conversation_context
            plan = await planner.create_plan_with_feedback(
                goal,
                source_paths,
                lambda **kwargs: self.emit(run_id, **kwargs),
                **planner_kwargs,
            )
        except Exception as exc:
            message = str(exc)
            if "OPENAI_API_KEY" in message:
                open_questions = ["GPT-5 runtime is not ready for this local session."]
                next_action = "Start a new run after the local GPT-5 runtime is ready."
            else:
                open_questions = [f"Resolve planner failure: {message}"]
                next_action = "Inspect planner trace, fix the LLM/tool issue, then start a new run."
            self.store.update_state(
                run_id,
                {
                    "status": RunStatus.FAILED.value,
                    "open_questions": open_questions,
                    "next_action": next_action,
                },
            )
            await self.emit(
                run_id,
                type="error",
                title="Planner failed",
                summary=message,
                status="failed",
            )
            return
        plan.version = (previous_plan.version + 1) if previous_plan else 1
        self.store.write_plan(run_id, plan)
        self.store.update_state(
            run_id,
            {
                "status": RunStatus.AWAITING_APPROVAL.value,
                "decisions": [
                    "Use Planner as orchestrator and Executor as ReAct-style task worker.",
                    "Plan Mode requires user approval before execution.",
                ],
                "planning_feedback": feedback_history,
                "next_action": "Review the plan, then approve it or send feedback for a revised plan.",
            },
        )
        await self.emit(
            run_id,
            type="planner",
            title="Plan ready for review",
            summary=f"Planner proposed {len(plan.tasks)} executable tasks. Waiting for user approval.",
            data=plan.model_dump(),
        )
        await asyncio.sleep(step_delay_ms / 1000)

    async def plan_after_human_interrupt(
        self,
        run_id: str,
        feedback: str,
        step_delay_ms: int = 700,
    ) -> None:
        llm = LLMClient(enabled=True)
        planner = Planner(llm, step_delay_ms=step_delay_ms)
        state = self.store.read_json(run_id, "state.json", {})
        previous_plan = self.store.read_plan(run_id)
        if not previous_plan:
            self.store.update_state(
                run_id,
                {
                    "status": RunStatus.FAILED.value,
                    "open_questions": ["Cannot recover this run because no prior plan exists."],
                    "next_action": "Start a new Plan Mode run.",
                },
            )
            await self.emit(
                run_id,
                type="error",
                title="Human recovery failed",
                summary="Cannot recover this run because no prior plan exists.",
                status="failed",
            )
            return

        source_payload = self.store.read_json(run_id, "sources.json", {"source_paths": []})
        source_paths = source_payload.get("source_paths", [])
        goal = state.get("goal") or previous_plan.goal
        feedback_history = list(state.get("planning_feedback", []))
        feedback_history.append(feedback)

        self.store.update_state(
            run_id,
            {
                "status": RunStatus.PLANNING.value,
                "planning_feedback": feedback_history,
                "next_action": "Planner is revising the blocked run from human recovery feedback.",
            },
        )
        await self.emit(
            run_id,
            type="planner",
            title="Human recovery feedback received",
            summary=feedback,
            status="success",
            data={"feedback_history": feedback_history},
        )
        await asyncio.sleep(step_delay_ms / 1000)

        replan_context = self._build_human_interrupt_context(previous_plan, state, feedback)
        try:
            planner_kwargs: dict[str, Any] = {
                "feedback_history": feedback_history,
                "previous_plan": previous_plan,
                "replan_context": replan_context,
            }
            conversation_context = self.store.read_json(run_id, "conversation_context.json", None)
            if conversation_context:
                planner_kwargs["conversation_context"] = conversation_context
            raw_plan = await planner.create_plan_with_feedback(
                goal,
                source_paths,
                lambda **kwargs: self.emit(run_id, **kwargs),
                **planner_kwargs,
            )
        except Exception as exc:
            message = f"Human recovery planning failed: {exc}"
            self.store.update_state(
                run_id,
                {
                    "status": RunStatus.FAILED.value,
                    "open_questions": [message],
                    "next_action": "Revise the recovery instruction or start a new run.",
                },
            )
            await self.emit(run_id, type="error", title="Human recovery planning failed", summary=message, status="failed")
            return

        replanned = self._merge_replanned_plan(previous_plan, raw_plan)
        self.store.write_plan(run_id, replanned)
        self.store.update_state(
            run_id,
            {
                "status": RunStatus.AWAITING_APPROVAL.value,
                "planning_feedback": feedback_history,
                "replan_count": 0,
                "open_questions": [],
                "next_action": "Review the recovery plan, then approve it to resume execution.",
            },
        )
        await self.emit(
            run_id,
            type="planner",
            title="Recovery plan ready for review",
            summary=f"Planner proposed plan v{replanned.version} after human recovery feedback. Done tasks were preserved.",
            status="success",
            data=replanned.model_dump(),
        )
        await asyncio.sleep(step_delay_ms / 1000)

    async def execute_existing_plan(self, run_id: str, max_parallel: int | None = None, step_delay_ms: int = 700) -> None:
        llm = LLMClient(enabled=True)
        executor = Executor(self.store, llm, step_delay_ms=step_delay_ms)
        planner = Planner(llm, step_delay_ms=step_delay_ms)
        plan = self.store.read_plan(run_id)
        if not plan:
            self.store.update_state(
                run_id,
                {
                    "status": RunStatus.FAILED.value,
                    "open_questions": ["Create and approve a plan before execution."],
                    "next_action": "Start Plan Mode first.",
                },
            )
            await self.emit(run_id, type="error", title="No approved plan", summary="Execution requires an existing plan.", status="failed")
            return

        source_payload = self.store.read_json(run_id, "sources.json", {"source_paths": []})
        source_paths = source_payload.get("source_paths", [])
        goal = self.store.read_json(run_id, "state.json", {}).get("goal", plan.goal)
        self.store.update_state(
            run_id,
            {
                "status": RunStatus.RUNNING.value,
                "next_action": "User approved the plan. Dispatch ready tasks.",
            },
        )
        await self.emit(
            run_id,
            type="orchestrator",
            title="User approved plan",
            summary=(
                f"Orchestrator received {len(plan.tasks)} tasks; parallel budget is dynamic."
                if max_parallel is None
                else f"Orchestrator received {len(plan.tasks)} tasks; parallel budget is {max_parallel}."
            ),
            status="success",
            data=plan.model_dump(),
        )
        await asyncio.sleep(step_delay_ms / 1000)

        while True:
            plan = self.store.read_plan(run_id) or plan
            if all(task.status in {"done", "blocked", "failed"} for task in plan.tasks):
                break
            ready = ready_tasks(plan)
            if not ready:
                await self.emit(run_id, type="error", title="No ready tasks", summary="Task graph is blocked.", status="failed")
                break
            parallel = self._select_dispatch_batch(ready, max_parallel)
            for task in parallel:
                task.status = "running"
                task.assigned_to = f"executor-{task.id}"
            self.store.write_plan(run_id, plan)
            if len(parallel) > 1:
                await self.emit(
                    run_id,
                    type="orchestrator",
                    title="Spawn parallel executors",
                    summary=f"Running {len(parallel)} independent tasks in parallel: {', '.join(task.id for task in parallel)}.",
                    data={"task_ids": [task.id for task in parallel]},
                )
                await asyncio.sleep(step_delay_ms / 1000)

            reports = await asyncio.gather(
                *[
                    executor.execute(
                        run_id,
                        task,
                        goal,
                        source_paths,
                        lambda **kwargs: self.emit(run_id, **kwargs),
                    )
                    for task in parallel
                ]
            )
            report_by_id = {report.task_id: report for report in reports}
            plan = self.store.read_plan(run_id) or plan
            done_items = []
            facts = []
            open_questions = []
            refs = []
            for task in plan.tasks:
                if task.id in report_by_id:
                    report = report_by_id[task.id]
                    task.status = report.status
                    task.report = report.model_dump()
                if task.status == "done":
                    done_items.append(f"{task.id}: {task.title}")
                if task.report:
                    facts.extend(fact.get("claim", "") for fact in task.report.get("key_facts", []) if fact.get("claim"))
                    open_questions.extend(task.report.get("open_questions", []))
                    refs.extend(task.report.get("refs", []))
            self.store.write_plan(run_id, plan)
            next_ready = ready_tasks(plan)
            self.store.update_state(
                run_id,
                {
                    "status": RunStatus.RUNNING.value,
                    "done": done_items,
                    "facts": facts[-12:],
                    "open_questions": open_questions[-8:],
                    "refs": refs[-12:],
                    "next_action": f"Next ready tasks: {', '.join(task.id for task in next_ready)}" if next_ready else "Finalize run.",
                },
            )
            await self.emit(run_id, type="state", title="Current state updated", summary="Orchestrator compacted task reports into current_state.md.")
            await asyncio.sleep(step_delay_ms / 1000)

            replan_reason = self._should_replan(run_id, reports, self.store.read_json(run_id, "state.json", {}))
            if replan_reason:
                replanned = await self._replan_after_batch(
                    run_id,
                    planner,
                    goal,
                    source_paths,
                    plan,
                    reports,
                    replan_reason,
                    step_delay_ms,
                )
                if replanned:
                    plan = replanned
                    continue
                return

        plan = self.store.read_plan(run_id) or plan
        final_status = self._final_status(run_id, plan)
        final_state = self.store.read_json(run_id, "state.json", {})
        updates = {"status": final_status, "next_action": "Run complete."}
        if final_status == RunStatus.BLOCKED.value:
            updates["next_action"] = "Run blocked. Add recovery feedback to revise the remaining plan."
            if not final_state.get("open_questions"):
                updates["open_questions"] = [
                    "Automatic execution stopped before a final brief was produced. Human recovery feedback is required."
                ]
        self.store.update_state(run_id, updates)
        await self.emit(run_id, type="orchestrator", title="Run complete", summary=f"Final status: {final_status}.")

    async def _replan_after_batch(
        self,
        run_id: str,
        planner: Planner,
        goal: str,
        source_paths: list[str],
        current_plan: Plan,
        reports: list[TaskReport],
        reason: dict[str, Any],
        step_delay_ms: int,
    ) -> Plan | None:
        state = self.store.read_json(run_id, "state.json", {})
        replan_count = int(state.get("replan_count", 0)) + 1
        self.store.update_state(
            run_id,
            {
                "status": RunStatus.REPLANNING.value,
                "replan_count": replan_count,
                "next_action": "Planner is revising the remaining task graph after executor feedback.",
            },
        )
        await self.emit(
            run_id,
            type="orchestrator",
            title="Replan triggered",
            summary=reason["summary"],
            status="running",
            data=reason,
        )
        await asyncio.sleep(step_delay_ms / 1000)

        replan_context = self._build_replan_context(current_plan, reports, state, reason)
        try:
            planner_kwargs: dict[str, Any] = {
                "previous_plan": current_plan,
                "replan_context": replan_context,
            }
            conversation_context = self.store.read_json(run_id, "conversation_context.json", None)
            if conversation_context:
                planner_kwargs["conversation_context"] = conversation_context
            raw_plan = await planner.create_plan_with_feedback(
                goal,
                source_paths,
                lambda **kwargs: self.emit(run_id, **kwargs),
                **planner_kwargs,
            )
        except Exception as exc:
            message = f"Execution-time replan failed: {exc}"
            self.store.update_state(
                run_id,
                {
                    "status": RunStatus.FAILED.value,
                    "open_questions": [message],
                    "next_action": "Inspect the replan trace and fix the planner issue.",
                },
            )
            await self.emit(run_id, type="error", title="Replan failed", summary=message, status="failed")
            return None

        replanned = self._merge_replanned_plan(current_plan, raw_plan)
        self.store.write_plan(run_id, replanned)
        self.store.update_state(
            run_id,
            {
                "status": RunStatus.RUNNING.value,
                "next_action": f"Execution resumed from plan v{replanned.version}.",
            },
        )
        await self.emit(
            run_id,
            type="planner",
            title="Planner revised execution plan",
            summary=f"Execution resumed with plan v{replanned.version}; done tasks were preserved.",
            status="success",
            data=replanned.model_dump(),
        )
        await asyncio.sleep(step_delay_ms / 1000)
        return replanned

    @staticmethod
    def _select_dispatch_batch(ready: list[AgentTask], max_parallel: int | None = None) -> list[AgentTask]:
        if max_parallel is not None and max_parallel <= 1:
            return ready[:1]
        parallelizable = [task for task in ready if task.parallelizable]
        if len(parallelizable) >= 2:
            return parallelizable if max_parallel is None else parallelizable[:max_parallel]
        return ready[:1]

    def _should_replan(self, run_id: str, reports: list[TaskReport], state: dict[str, Any]) -> dict[str, Any] | None:
        if (self.store.run_dir(run_id) / "final.md").exists():
            return None
        if int(state.get("replan_count", 0)) >= 2:
            return None

        failed_or_blocked = [report for report in reports if report.status in {"failed", "blocked"}]
        if failed_or_blocked:
            task_ids = ", ".join(report.task_id for report in failed_or_blocked)
            return {
                "kind": "blocked_or_failed_tasks",
                "summary": f"Replanning because task(s) {task_ids} ended blocked or failed.",
                "task_ids": [report.task_id for report in failed_or_blocked],
                "reports": [report.model_dump() for report in failed_or_blocked],
            }
        return None

    def _final_status(self, run_id: str, plan: Plan) -> str:
        final_exists = (self.store.run_dir(run_id) / "final.md").exists()
        final_task_done = any(task.produces_final and task.status == "done" for task in plan.tasks)
        return RunStatus.DONE.value if final_exists and final_task_done else RunStatus.BLOCKED.value

    async def _load_or_build_conversation_context(
        self,
        run_id: str,
        goal: str,
        llm: LLMClient,
        step_delay_ms: int,
    ) -> dict[str, Any] | None:
        existing = self.store.read_json(run_id, "conversation_context.json", None)
        if existing:
            return existing
        conversation = self.store.read_json(run_id, "conversation.json", {}) or {}
        parent_run_id = conversation.get("parent_run_id")
        if not parent_run_id:
            return None
        try:
            prior_payload = self.store.prior_run_payload(str(parent_run_id))
        except Exception as exc:
            await self.emit(
                run_id,
                type="error",
                title="Previous context unavailable",
                summary=str(exc),
                status="failed",
            )
            return None
        summary = await asyncio.to_thread(summarize_prior_run_for_followup, goal, prior_payload, llm)
        self.store.write_json(run_id, "conversation_context.json", summary)
        self.store.update_state(
            run_id,
            {
                "next_action": "Planner will use the previous run as follow-up context.",
            },
        )
        await self.emit(
            run_id,
            type="state",
            title="Previous run context attached",
            summary=f"Loaded compressed context from previous run {parent_run_id}.",
            status="success",
            data=summary,
        )
        await asyncio.sleep(step_delay_ms / 1000)
        return summary

    @staticmethod
    def _build_replan_context(
        plan: Plan,
        reports: list[TaskReport],
        state: dict[str, Any],
        reason: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "mode": "execution_replan",
            "reason": reason,
            "current_plan": plan.model_dump(),
            "completed_tasks": [task.model_dump() for task in plan.tasks if task.status == "done"],
            "blocked_or_failed_tasks": [
                task.model_dump()
                for task in plan.tasks
                if task.status in {"blocked", "failed"}
            ],
            "latest_reports": [report.model_dump() for report in reports],
            "current_state": {
                "done": state.get("done", []),
                "facts": state.get("facts", []),
                "refs": state.get("refs", []),
                "open_questions": state.get("open_questions", []),
                "next_action": state.get("next_action", ""),
            },
            "rules": [
                "Preserve done tasks exactly.",
                "Do not rerun done tasks.",
                "Replace blocked, failed, or pending tasks only when needed.",
                "Keep exactly one produces_final task.",
            ],
        }

    @staticmethod
    def _build_human_interrupt_context(plan: Plan, state: dict[str, Any], feedback: str) -> dict[str, Any]:
        return {
            "mode": "human_interrupt_recovery",
            "reason": {
                "kind": "human_recovery_feedback",
                "summary": "User supplied recovery instructions after the run reached a terminal blocked/failed state.",
                "feedback": feedback,
            },
            "current_plan": plan.model_dump(),
            "completed_tasks": [task.model_dump() for task in plan.tasks if task.status == "done"],
            "blocked_or_failed_tasks": [
                task.model_dump()
                for task in plan.tasks
                if task.status in {"blocked", "failed"}
            ],
            "pending_tasks": [task.model_dump() for task in plan.tasks if task.status == "pending"],
            "latest_reports": [task.report for task in plan.tasks if task.report],
            "current_state": {
                "done": state.get("done", []),
                "facts": state.get("facts", []),
                "refs": state.get("refs", []),
                "open_questions": state.get("open_questions", []),
                "next_action": state.get("next_action", ""),
            },
            "rules": [
                "Follow the user's recovery feedback.",
                "Preserve done tasks exactly.",
                "Do not rerun done tasks.",
                "Replace blocked, failed, or pending tasks only when needed.",
                "Keep exactly one produces_final task.",
                "Return the revised plan for user approval; do not resume execution directly.",
            ],
        }

    @staticmethod
    def _merge_replanned_plan(previous: Plan, replanned: Plan) -> Plan:
        done_by_id = {task.id: task for task in previous.tasks if task.status == "done"}
        merged_tasks: list[AgentTask] = []
        seen: set[str] = set()

        for task in replanned.tasks:
            if task.id in done_by_id:
                merged_tasks.append(done_by_id[task.id])
            else:
                task.status = "pending"
                task.assigned_to = None
                task.report = None
                merged_tasks.append(task)
            seen.add(task.id)

        for task in previous.tasks:
            if task.id in done_by_id and task.id not in seen:
                merged_tasks.insert(0, task)
                seen.add(task.id)

        return Plan(
            goal=replanned.goal or previous.goal,
            version=previous.version + 1,
            assumptions=replanned.assumptions,
            success_criteria=replanned.success_criteria,
            tasks=merged_tasks,
        )
