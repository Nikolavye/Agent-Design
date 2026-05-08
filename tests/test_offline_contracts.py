from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from evidence_agent.context import compact_executor_context, compact_planner_context, estimate_tokens
from evidence_agent.executor import Executor
from evidence_agent.llm import LLMClient
from evidence_agent.loop import Orchestrator
from evidence_agent.planner import Planner
from evidence_agent.state.run_store import RunStore
from evidence_agent.tools import pdf_tools
from evidence_agent.tools.pdf_tools import normalize_pdf_url
from evidence_agent.types import AgentTask, Plan, TaskReport


class FakeContextSummaryLLM:
    enabled = True
    last_error = None

    def __init__(self) -> None:
        self.calls = 0

    def json(self, system, user, default_response=None):
        self.calls += 1
        return {
            "summary": "Prior context says t1 is done, t2 failed, preserve artifact refs and revise only the failed work.",
            "must_keep": ["Preserve done task t1.", "Do not lose user feedback."],
            "task_statuses": [{"id": "t1", "status": "done", "note": "source indexed"}],
            "evidence_refs": ["artifacts/ref-1.txt"],
            "open_questions": ["Need a safer retrieval path."],
        }


class OfflineAgentContracts(unittest.IsolatedAsyncioTestCase):
    def test_planner_context_uses_llm_contextual_summary_when_over_budget(self) -> None:
        llm = FakeContextSummaryLLM()
        context = {
            "goal": "Write a final brief.",
            "selected_sources": [],
            "available_tools": [{"name": "memory_get", "description": "Load artifact."}],
            "planning_mode": {
                "requires_user_confirmation_before_execution": True,
                "feedback_history": ["feedback " + "f" * 1000 for _ in range(5)],
                "previous_plan": {
                    "goal": "Write a final brief.",
                    "version": 2,
                    "tasks": [
                        {
                            "id": "t1",
                            "title": "Done task",
                            "description": "Already completed. " + "x" * 1500,
                            "status": "done",
                            "report": {"summary": "done " + "y" * 1500, "refs": ["artifacts/ref-1.txt"]},
                        }
                    ],
                },
                "replan_context": None,
            },
            "recent_observations": [{"summary": "observed " + "z" * 1200}],
        }

        compacted, stats = compact_planner_context(context, budget_tokens=800, llm=llm)

        self.assertEqual(llm.calls, 1)
        self.assertTrue(stats["contextual_summary_used"])
        self.assertIn("llm contextual summary generated for planner context", stats["operations"])
        self.assertEqual(
            compacted["planning_mode"]["contextual_summary"]["summary"],
            "Prior context says t1 is done, t2 failed, preserve artifact refs and revise only the failed work.",
        )

    def test_executor_context_uses_llm_contextual_summary_when_over_budget(self) -> None:
        llm = FakeContextSummaryLLM()
        context = {
            "goal": "Write a final brief.",
            "task": {"id": "t2", "title": "Read evidence", "description": "Use refs. " + "x" * 1500},
            "selected_sources": [],
            "current_state": {
                "done": ["t1"],
                "facts": ["fact " + "y" * 800 for _ in range(8)],
                "refs": [f"artifacts/ref-{idx}.txt" for idx in range(8)],
                "next_action": "Continue.",
            },
            "indexed_sources": [f"source-{idx}" for idx in range(30)],
            "recent_observations": [{"summary": "observation " + "z" * 1000} for _ in range(6)],
            "known_public_urls": [f"https://example.com/{idx}" for idx in range(6)],
        }

        compacted, stats = compact_executor_context(context, budget_tokens=1000, llm=llm)

        self.assertEqual(llm.calls, 1)
        self.assertTrue(stats["contextual_summary_used"])
        self.assertIn("llm contextual summary generated for executor context", stats["operations"])
        self.assertEqual(compacted["contextual_summary"]["evidence_refs"], ["artifacts/ref-1.txt"])

    def test_planner_context_compacts_plan_and_replan_state(self) -> None:
        large_report = {
            "task_id": "t1",
            "status": "failed",
            "summary": "failed " + "x" * 2000,
            "key_facts": [{"claim": "claim " + "y" * 500, "ref": "artifacts/a.txt"} for _ in range(10)],
            "refs": [f"artifacts/ref-{idx}.txt" for idx in range(20)],
            "open_questions": ["question " + "z" * 300 for _ in range(10)],
        }
        tasks = [
            {
                "id": f"t{idx}",
                "title": "Task " + "a" * 200,
                "description": "Do work. " + "b" * 1200,
                "depends_on": [],
                "requires": ["memory_search", "memory_get"],
                "status": "failed" if idx == 1 else "done",
                "report": large_report,
            }
            for idx in range(1, 8)
        ]
        context = {
            "goal": "Write a final brief.",
            "selected_sources": [],
            "available_tools": [{"name": "memory_get", "description": "Load artifact."}],
            "planning_mode": {
                "requires_user_confirmation_before_execution": True,
                "feedback_history": ["feedback " + "f" * 1200 for _ in range(8)],
                "previous_plan": {"goal": "Write a final brief.", "version": 3, "tasks": tasks},
                "replan_context": {
                    "mode": "execution_replan",
                    "reason": {"summary": "failed " + "r" * 1200},
                    "current_plan": {"goal": "Write a final brief.", "version": 3, "tasks": tasks},
                    "completed_tasks": tasks[:3],
                    "blocked_or_failed_tasks": tasks[3:],
                    "latest_reports": [large_report for _ in range(6)],
                    "current_state": {
                        "done": [f"t{idx}" for idx in range(10)],
                        "facts": ["fact " + "c" * 300 for _ in range(10)],
                        "refs": [f"artifacts/ref-{idx}.txt" for idx in range(10)],
                        "open_questions": ["open " + "d" * 300 for _ in range(10)],
                        "next_action": "Recover.",
                    },
                },
            },
            "recent_observations": [
                {"tool": "fs_list", "summary": "listed " + "o" * 900, "data_preview": {"items": list(range(20))}}
                for _ in range(8)
            ],
        }

        compacted, stats = compact_planner_context(context, budget_tokens=5000)

        self.assertTrue(stats["compacted"])
        self.assertFalse(stats["overflow"])
        self.assertLess(estimate_tokens(compacted), estimate_tokens(context))
        self.assertLessEqual(len(compacted["recent_observations"]), 2)
        self.assertLessEqual(len(compacted["planning_mode"]["feedback_history"]), 3)

    def test_planner_blocks_before_llm_when_context_overflows(self) -> None:
        planner = Planner(LLMClient(enabled=False), step_delay_ms=0, context_token_budget=100)
        previous_plan = Plan(
            goal="Write a final brief.",
            tasks=[
                AgentTask(
                    id="t1",
                    title="Oversized planning task",
                    description="This non-droppable planning task is intentionally huge. " + "x" * 5000,
                    produces_final=True,
                )
            ],
        )

        action = planner._next_action(
            "Write a final brief.",
            [],
            [],
            [],
            previous_plan,
            None,
        )

        self.assertEqual(action["action"], "block_plan")
        self.assertTrue(action["_context_budget"]["overflow"])
        self.assertIn("Planner context exceeded", action["open_questions"][0])

    def test_executor_context_compacts_under_small_budget(self) -> None:
        context = {
            "goal": "Write a cited brief.",
            "task": {"id": "t1", "title": "Retrieve evidence", "description": "Search and read refs."},
            "selected_sources": ["data/pdfs/paper.pdf"],
            "current_state": {
                "done": ["t0"],
                "facts": [f"fact {idx} " + "x" * 300 for idx in range(12)],
                "refs": [f"artifacts/ref-{idx}.txt" for idx in range(12)],
                "next_action": "Continue.",
            },
            "indexed_sources": [f"source-{idx}.pdf" for idx in range(40)],
            "recent_observations": [
                {
                    "step": idx,
                    "tool": "memory_get",
                    "summary": "read " + "y" * 500,
                    "data_preview": {"text": "z" * 1200, "items": list(range(10))},
                }
                for idx in range(10)
            ],
            "known_public_urls": [f"https://example.com/{idx}" for idx in range(10)],
        }

        compacted, stats = compact_executor_context(context, budget_tokens=900)

        self.assertTrue(stats["compacted"])
        self.assertLess(estimate_tokens(compacted), estimate_tokens(context))
        self.assertLessEqual(len(compacted["recent_observations"]), 2)
        self.assertLessEqual(len(compacted["current_state"]["facts"]), 2)
        self.assertFalse(stats["overflow"])

    def test_executor_context_reports_overflow_after_hard_boundary(self) -> None:
        context = {
            "goal": "Write a cited brief.",
            "task": {
                "id": "t1",
                "title": "Oversized task",
                "description": "This non-droppable task description is intentionally huge. " + "x" * 5000,
            },
            "selected_sources": [],
            "current_state": {"done": [], "facts": [], "refs": [], "next_action": ""},
            "indexed_sources": [],
            "recent_observations": [],
            "known_public_urls": [],
        }

        _compacted, stats = compact_executor_context(context, budget_tokens=100)

        self.assertTrue(stats["overflow"])
        self.assertGreater(stats["estimated_tokens_after"], stats["budget_tokens"])

    def test_executor_blocks_before_llm_when_context_overflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            run_id = store.create_run("Write a final brief.", [])
            executor = Executor(store, LLMClient(enabled=False), step_delay_ms=0, context_token_budget=100)
            task = AgentTask(
                id="t1",
                title="Oversized task",
                description="This non-droppable task description is intentionally huge. " + "x" * 5000,
            )

            action = executor._next_action(run_id, task, "Write a final brief.", [], [])

        self.assertEqual(action["action"], "block_task")
        self.assertTrue(action["_context_budget"]["overflow"])
        self.assertIn("Context budget exceeded", action["completion"]["summary"])

    def test_pdf_download_normalizes_arxiv_abs_url(self) -> None:
        self.assertEqual(
            normalize_pdf_url("https://arxiv.org/abs/2601.01090"),
            "https://arxiv.org/pdf/2601.01090.pdf",
        )

    def test_pdf_download_writes_bounded_local_pdf(self) -> None:
        class FakeResponse:
            url = "https://arxiv.org/pdf/2601.01090.pdf"
            headers = {"content-type": "application/pdf"}
            content = b"%PDF-1.4\nfake test pdf\n%%EOF"

            def raise_for_status(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch.object(pdf_tools, "PROJECT_ROOT", root),
                patch.object(pdf_tools, "PDF_DIR", root / "pdfs"),
                patch("evidence_agent.tools.pdf_tools.httpx.get", return_value=FakeResponse()) as mocked_get,
            ):
                result = pdf_tools.pdf_download("https://arxiv.org/abs/2601.01090", filename="toxicity-paper.pdf")

        self.assertTrue(result.ok)
        self.assertEqual(mocked_get.call_args.args[0], "https://arxiv.org/pdf/2601.01090.pdf")
        self.assertEqual(result.data["path"], "pdfs/toxicity-paper.pdf")

    def test_planner_validates_and_infers_final_task(self) -> None:
        plan = Planner._parse_plan(
            "Create an evidence brief.",
            {
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Index PDFs",
                        "description": "Extract source documents.",
                        "depends_on": [],
                        "parallelizable": True,
                        "requires": ["pdf_extract"],
                        "acceptance_criteria": "PDF chunks are indexed.",
                    },
                    {
                        "id": "t2",
                        "title": "Final synthesis brief",
                        "description": "Write the cited final report.",
                        "depends_on": ["t1"],
                        "parallelizable": False,
                        "requires": ["memory_search", "memory_get"],
                        "acceptance_criteria": "Final markdown brief is produced.",
                    },
                ],
            },
        )

        self.assertEqual(len([task for task in plan.tasks if task.produces_final]), 1)
        self.assertTrue(plan.tasks[1].produces_final)

    def test_planner_rejects_dependency_cycles(self) -> None:
        with self.assertRaisesRegex(ValueError, "cyclic dependency"):
            Planner._parse_plan(
                "Create an evidence brief.",
                {
                    "tasks": [
                        {
                            "id": "t1",
                            "title": "Collect evidence",
                            "description": "Collect sources.",
                            "depends_on": ["t2"],
                        },
                        {
                            "id": "t2",
                            "title": "Final synthesis brief",
                            "description": "Write final report.",
                            "depends_on": ["t1"],
                            "produces_final": True,
                        },
                    ],
                },
            )

    def test_replan_allows_dependencies_on_completed_tasks(self) -> None:
        plan = Planner._parse_plan(
            "Create an evidence brief.",
            {
                "tasks": [
                    {
                        "id": "t2r",
                        "title": "Retry evidence retrieval",
                        "description": "Use the already indexed PDF.",
                        "depends_on": ["t1"],
                    },
                    {
                        "id": "t3r",
                        "title": "Final synthesis brief",
                        "description": "Write the final brief.",
                        "depends_on": ["t2r"],
                        "produces_final": True,
                    },
                ],
            },
            allowed_external_dependencies={"t1"},
        )

        self.assertEqual(plan.tasks[0].depends_on, ["t1"])

    def test_orchestrator_only_parallelizes_parallelizable_tasks(self) -> None:
        serial = AgentTask(id="t1", title="Serial", description="Must run alone.")
        parallel_a = AgentTask(id="t2", title="Parallel A", description="Can run with peers.", parallelizable=True)
        parallel_b = AgentTask(id="t3", title="Parallel B", description="Can run with peers.", parallelizable=True)
        parallel_c = AgentTask(id="t4", title="Parallel C", description="Can run with peers.", parallelizable=True)

        first_batch = Orchestrator._select_dispatch_batch([serial, parallel_a, parallel_b], max_parallel=None)
        second_batch = Orchestrator._select_dispatch_batch([parallel_a, parallel_b, parallel_c], max_parallel=None)
        capped_batch = Orchestrator._select_dispatch_batch([parallel_a, parallel_b, parallel_c], max_parallel=2)

        self.assertEqual([task.id for task in first_batch], ["t2", "t3"])
        self.assertEqual([task.id for task in second_batch], ["t2", "t3", "t4"])
        self.assertEqual([task.id for task in capped_batch], ["t2", "t3"])

    def test_execution_replan_trigger_respects_limits_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            run_id = store.create_run("Write a final brief.", [])
            orchestrator = Orchestrator(store)

            reason = orchestrator._should_replan(
                run_id,
                [TaskReport(task_id="t1", status="failed", summary="Search failed.")],
                {"replan_count": 0},
            )
            limited = orchestrator._should_replan(
                run_id,
                [TaskReport(task_id="t1", status="failed", summary="Search failed.")],
                {"replan_count": 2},
            )
            store.write_final(run_id, "# Final\n")
            after_final = orchestrator._should_replan(
                run_id,
                [TaskReport(task_id="t1", status="failed", summary="Search failed.")],
                {"replan_count": 0},
            )

        self.assertEqual(reason["kind"], "blocked_or_failed_tasks")
        self.assertIsNone(limited)
        self.assertIsNone(after_final)

    def test_final_status_requires_done_final_task_and_final_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            run_id = store.create_run("Write a final brief.", [])
            orchestrator = Orchestrator(store)
            blocked_final = Plan(
                goal="Write a final brief.",
                tasks=[
                    AgentTask(id="t1", title="Evidence", description="Done.", status="done"),
                    AgentTask(id="t2", title="Final", description="Blocked.", produces_final=True, status="blocked"),
                ],
            )
            store.write_final(run_id, "# Premature final\n")
            self.assertEqual(orchestrator._final_status(run_id, blocked_final), "blocked")

            blocked_final.tasks[1].status = "done"
            self.assertEqual(orchestrator._final_status(run_id, blocked_final), "done")

    def test_run_store_serializes_concurrent_events_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            run_id = store.create_run("Write a final brief.", [])

            def append_one(idx: int) -> str:
                event = store.append_event(
                    run_id,
                    type="executor",
                    title=f"event {idx}",
                    summary="concurrent append",
                )
                return event.event_id

            with ThreadPoolExecutor(max_workers=12) as pool:
                event_ids = list(pool.map(append_one, range(60)))

            events = store.read_events(run_id)

        self.assertEqual(len(events), 60)
        self.assertEqual(len(set(event_ids)), 60)
        self.assertEqual([event["event_id"] for event in events], [f"evt_{idx:04d}" for idx in range(1, 61)])

    def test_llm_json_parser_accepts_first_object_with_trailing_data(self) -> None:
        parsed = LLMClient._parse_json_object('{"action": "complete_task"}\n{"extra": true}')

        self.assertEqual(parsed, {"action": "complete_task"})

    def test_merge_replanned_plan_preserves_done_tasks(self) -> None:
        done_task = AgentTask(
            id="t1",
            title="Completed source inventory",
            description="Already done.",
            status="done",
            report={"summary": "done"},
        )
        previous = Plan(
            goal="Write a final brief.",
            version=3,
            tasks=[
                done_task,
                AgentTask(id="t2", title="Failed web search", description="Blocked.", status="failed"),
            ],
        )
        replanned = Plan(
            goal="Write a final brief.",
            tasks=[
                AgentTask(id="t1", title="Modified inventory", description="Should not overwrite done task."),
                AgentTask(id="t3", title="Retry acquisition", description="Use pdf_download."),
                AgentTask(id="t4", title="Final synthesis brief", description="Write final.", produces_final=True),
            ],
        )

        merged = Orchestrator._merge_replanned_plan(previous, replanned)

        self.assertEqual(merged.version, 4)
        self.assertEqual(merged.tasks[0].title, "Completed source inventory")
        self.assertEqual(merged.tasks[0].status, "done")
        self.assertEqual(merged.tasks[1].status, "pending")
        self.assertEqual(len([task for task in merged.tasks if task.produces_final]), 1)

    async def test_plan_mode_waits_for_user_approval_and_revises(self) -> None:
        async def fake_create_plan(
            planner,
            goal,
            source_paths,
            emit,
            *,
            feedback_history=None,
            previous_plan=None,
            replan_context=None,
        ):
            title = "Final synthesis brief"
            if feedback_history:
                title = f"Final synthesis brief revised for {len(feedback_history)} feedback item"
            await emit(type="planner", title="Fake planner completed", summary=title)
            return Plan(
                goal=goal,
                tasks=[
                    AgentTask(
                        id="t1",
                        title=title,
                        description="Write final brief.",
                        produces_final=True,
                        acceptance_criteria="Final markdown exists.",
                    )
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            orchestrator = Orchestrator(store)
            run_id = store.create_run("Write a final brief.", [])
            with patch("evidence_agent.loop.Planner.create_plan_with_feedback", new=fake_create_plan):
                await orchestrator.plan_for_review(run_id, "Write a final brief.", [], step_delay_ms=0)
                first_state = store.read_json(run_id, "state.json", {})
                first_plan = store.read_plan(run_id)
                await orchestrator.plan_for_review(
                    run_id,
                    "Write a final brief.",
                    [],
                    step_delay_ms=0,
                    feedback="Make the plan shorter.",
                )
                second_state = store.read_json(run_id, "state.json", {})
                second_plan = store.read_plan(run_id)
                events = store.read_events(run_id)

        self.assertEqual(first_state["status"], "awaiting_approval")
        self.assertEqual(first_plan.version, 1)
        self.assertEqual(second_state["status"], "awaiting_approval")
        self.assertEqual(second_state["planning_feedback"], ["Make the plan shorter."])
        self.assertEqual(second_plan.version, 2)
        self.assertFalse(any(event["type"] == "executor" for event in events))

    async def test_human_interrupt_replan_returns_to_approval(self) -> None:
        captured = {}

        async def fake_create_plan(
            planner,
            goal,
            source_paths,
            emit,
            *,
            feedback_history=None,
            previous_plan=None,
            replan_context=None,
        ):
            captured["feedback_history"] = feedback_history
            captured["previous_plan_version"] = previous_plan.version
            captured["replan_context"] = replan_context
            await emit(type="planner", title="Fake recovery planner completed", summary="Recovery plan.")
            return Plan(
                goal=goal,
                tasks=[
                    AgentTask(
                        id="t1",
                        title="Completed source acquisition",
                        description="Already done.",
                    ),
                    AgentTask(
                        id="t2r",
                        title="Retry evidence retrieval",
                        description="Use memory_get refs instead of invalid fs_read artifact paths.",
                        depends_on=["t1"],
                    ),
                    AgentTask(
                        id="t3r",
                        title="Final synthesis brief",
                        description="Write final brief from recovered evidence.",
                        depends_on=["t2r"],
                        produces_final=True,
                    ),
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            orchestrator = Orchestrator(store)
            run_id = store.create_run("Write a final brief.", [])
            previous = Plan(
                goal="Write a final brief.",
                version=5,
                tasks=[
                    AgentTask(
                        id="t1",
                        title="Completed source acquisition",
                        description="Already done.",
                        status="done",
                        report={"summary": "PDF downloaded and indexed."},
                    ),
                    AgentTask(
                        id="t2",
                        title="Failed retrieval",
                        description="Used the wrong artifact reader.",
                        status="failed",
                        report={"summary": "fs_read failed on artifact path."},
                    ),
                    AgentTask(
                        id="t3",
                        title="Final synthesis brief",
                        description="Write final.",
                        produces_final=True,
                        status="pending",
                    ),
                ],
            )
            store.write_plan(run_id, previous)
            store.update_state(
                run_id,
                {
                    "status": "blocked",
                    "planning_feedback": ["Earlier feedback."],
                    "replan_count": 2,
                    "open_questions": ["t2 failed repeatedly."],
                },
            )

            with patch("evidence_agent.loop.Planner.create_plan_with_feedback", new=fake_create_plan):
                await orchestrator.plan_after_human_interrupt(
                    run_id,
                    "Retry retrieval with memory_get, then continue.",
                    step_delay_ms=0,
                )

            state = store.read_json(run_id, "state.json", {})
            plan = store.read_plan(run_id)
            events = store.read_events(run_id)

        self.assertEqual(state["status"], "awaiting_approval")
        self.assertEqual(state["replan_count"], 0)
        self.assertEqual(
            state["planning_feedback"],
            ["Earlier feedback.", "Retry retrieval with memory_get, then continue."],
        )
        self.assertEqual(plan.version, 6)
        self.assertEqual(plan.tasks[0].id, "t1")
        self.assertEqual(plan.tasks[0].status, "done")
        self.assertEqual(plan.tasks[1].id, "t2r")
        self.assertEqual(captured["previous_plan_version"], 5)
        self.assertEqual(captured["feedback_history"], state["planning_feedback"])
        self.assertEqual(captured["replan_context"]["mode"], "human_interrupt_recovery")
        self.assertTrue(any(event["title"] == "Recovery plan ready for review" for event in events))

    async def test_final_task_requires_final_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            run_id = store.create_run("Write a final brief.", [])
            executor = Executor(store, LLMClient(enabled=False), step_delay_ms=0)
            task = AgentTask(
                id="t_final",
                title="Final synthesis",
                description="Write final brief.",
                produces_final=True,
                acceptance_criteria="Final markdown brief exists.",
            )

            async def emit(**kwargs):
                store.append_event(run_id, **kwargs)

            report = await executor._complete_task(
                run_id,
                task,
                emit,
                {"completion": {"summary": "Done without markdown."}},
            )

            self.assertEqual(report.status, "failed")
            self.assertFalse((store.run_dir(run_id) / "final.md").exists())

    async def test_executor_rejects_invalid_artifact_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            run_id = store.create_run("Write a final brief.", [])
            executor = Executor(store, LLMClient(enabled=False), step_delay_ms=0)
            task = AgentTask(id="t1", title="Retrieve evidence", description="Collect cited facts.")

            async def emit(**kwargs):
                store.append_event(run_id, **kwargs)

            report = await executor._complete_task(
                run_id,
                task,
                emit,
                {
                    "completion": {
                        "summary": "Collected evidence.",
                        "key_facts": [{"claim": "Unsupported claim.", "ref": "external paper"}],
                        "refs": ["artifacts/missing.txt"],
                    }
                },
            )

            self.assertEqual(report.status, "failed")
            self.assertTrue(any(event["title"] == "Artifact ref contract failed" for event in store.read_events(run_id)))

    async def test_executor_accepts_valid_refs_from_markdown_and_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            run_id = store.create_run("Write a final brief.", [])
            ref = store.write_artifact(run_id, "valid-ref", "source text")
            executor = Executor(store, LLMClient(enabled=False), step_delay_ms=0)
            task = AgentTask(
                id="t_final",
                title="Final synthesis",
                description="Write final brief.",
                produces_final=True,
            )

            async def emit(**kwargs):
                store.append_event(run_id, **kwargs)

            report = await executor._complete_task(
                run_id,
                task,
                emit,
                {
                    "completion": {
                        "summary": "Wrote final.",
                        "key_facts": [{"claim": "Supported claim.", "ref": ref}],
                        "refs": [ref],
                        "final_markdown": f"## Brief\nSupported claim. [ref: {ref}]\n",
                    }
                },
            )

            self.assertEqual(report.status, "done")
            self.assertTrue((store.run_dir(run_id) / "final.md").exists())

    async def test_non_final_task_ignores_premature_final_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            run_id = store.create_run("Write a final brief.", [])
            executor = Executor(store, LLMClient(enabled=False), step_delay_ms=0)
            task = AgentTask(id="t1", title="Retrieve evidence", description="Collect evidence.", produces_final=False)

            async def emit(**kwargs):
                store.append_event(run_id, **kwargs)

            report = await executor._complete_task(
                run_id,
                task,
                emit,
                {
                    "completion": {
                        "summary": "Collected evidence.",
                        "final_markdown": "# Premature final",
                    }
                },
            )

            self.assertEqual(report.status, "done")
            self.assertFalse((store.run_dir(run_id) / "final.md").exists())
            self.assertTrue(any(event["title"] == "Ignored premature final markdown" for event in store.read_events(run_id)))

    async def test_executor_normalizes_structured_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            run_id = store.create_run("Write a final brief.", [])
            executor = Executor(store, LLMClient(enabled=False), step_delay_ms=0)
            task = AgentTask(id="t1", title="Index PDF", description="Extract source text.")

            async def emit(**kwargs):
                store.append_event(run_id, **kwargs)

            report = await executor._complete_task(
                run_id,
                task,
                emit,
                {
                    "completion": {
                        "summary": "Indexed source.",
                        "recommended_next_tasks": [
                            {"id": "t2", "title": "Retrieve evidence", "description": "Search indexed chunks."}
                        ],
                    }
                },
            )

            self.assertEqual(report.status, "done")
            self.assertEqual(report.recommended_next_tasks, ["Retrieve evidence"])

    async def test_executor_normalizes_bare_tool_arguments(self) -> None:
        task = AgentTask(id="t1", title="Retrieve evidence", description="Search chunks.", requires=["rag_search"])

        action = Executor._normalize_action({"query": "toxicity findings", "max_results": 6}, task, ["paper.pdf"])

        self.assertEqual(action["action"], "call_tool")
        self.assertEqual(action["tool"], "rag_search")
        self.assertEqual(action["args"]["query"], "toxicity findings")

    async def test_executor_normalizes_bare_fs_list_arguments(self) -> None:
        task = AgentTask(id="t1", title="Inspect files", description="List available files.", requires=["fs_list"])

        action = Executor._normalize_action({"root": "data", "glob": "**/*.pdf", "max_results": 20}, task, [])

        self.assertEqual(action["action"], "call_tool")
        self.assertEqual(action["tool"], "fs_list")
        self.assertEqual(action["args"]["root"], "data")

    async def test_executor_normalizes_artifact_paths_to_memory_get(self) -> None:
        task = AgentTask(id="t1", title="Read evidence", description="Read artifact refs.", requires=["memory_get"])

        bare = Executor._normalize_action({"path": "artifacts/chunk.txt", "max_chars": 900}, task, ["paper.pdf"])
        tool_call = Executor._normalize_action(
            {
                "thought_summary": "Read chunk.",
                "action": "call_tool",
                "tool": "fs_read",
                "args": {"path": "artifacts/chunk.txt", "max_chars": 900},
            },
            task,
            ["paper.pdf"],
        )

        self.assertEqual(bare["tool"], "memory_get")
        self.assertEqual(bare["args"]["ref"], "artifacts/chunk.txt")
        self.assertEqual(tool_call["tool"], "memory_get")
        self.assertEqual(tool_call["args"]["ref"], "artifacts/chunk.txt")


if __name__ == "__main__":
    unittest.main()
