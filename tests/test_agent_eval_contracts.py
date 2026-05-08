from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evidence_agent.loop import Orchestrator
from evidence_agent.state.run_store import RunStore
from evidence_agent.types import AgentTask, Plan, RunStatus, TaskReport

REF_PATTERN = re.compile(r"\[ref:\s*([^\]]+)\]", re.IGNORECASE)


def event_titles(events: list[dict]) -> list[str]:
    return [str(event.get("title", "")) for event in events]


def cited_refs(markdown: str) -> list[str]:
    refs: list[str] = []
    for match in REF_PATTERN.findall(markdown):
        refs.extend(ref.strip() for ref in match.split(";") if ref.strip())
    return refs


class AgentEvalContracts(unittest.IsolatedAsyncioTestCase):
    async def test_simulated_arxiv_run_satisfies_outcome_trace_and_citation_contracts(self) -> None:
        """A no-API eval that checks the same properties expected from the live arXiv demo."""

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            orchestrator = Orchestrator(store)
            goal = "Read https://arxiv.org/abs/2601.06700 and produce a cited evidence brief."
            run_id = store.create_run(goal, [])
            plan = Plan(
                goal=goal,
                tasks=[
                    AgentTask(
                        id="t1",
                        title="Download paper PDF",
                        description="Acquire the paper PDF.",
                        requires=["pdf_download"],
                    ),
                    AgentTask(
                        id="t2",
                        title="Extract paper PDF",
                        description="Extract and index the PDF.",
                        depends_on=["t1"],
                        requires=["pdf_extract"],
                    ),
                    AgentTask(
                        id="t3",
                        title="Retrieve method evidence",
                        description="Find method evidence.",
                        depends_on=["t2"],
                        parallelizable=True,
                        requires=["rag_search", "memory_get"],
                    ),
                    AgentTask(
                        id="t4",
                        title="Retrieve findings evidence",
                        description="Find results evidence.",
                        depends_on=["t2"],
                        parallelizable=True,
                        requires=["rag_search", "memory_get"],
                    ),
                    AgentTask(
                        id="t5",
                        title="Write final evidence brief",
                        description="Synthesize a cited brief.",
                        depends_on=["t3", "t4"],
                        produces_final=True,
                        requires=["memory_search", "memory_get"],
                    ),
                ],
            )
            store.write_plan(run_id, plan)

            async def fake_execute(executor, run_id, task, goal, source_paths, emit):
                await emit(type="executor", title=f"Executor starts {task.id}", summary=task.title, task_id=task.id)
                await emit(
                    type="executor",
                    title=f"ReAct decision ({task.id}, step 1)",
                    summary="Executor selected the next action.",
                    task_id=task.id,
                    data={"action": "call_tool"},
                )
                if task.id == "t1":
                    await emit(
                        type="tool_call",
                        title="Call pdf_download",
                        summary="Download arXiv PDF.",
                        task_id=task.id,
                        tool="pdf_download",
                    )
                    await emit(
                        type="tool_result",
                        title="pdf_download: ok",
                        summary="Downloaded PDF.",
                        task_id=task.id,
                        tool="pdf_download",
                    )
                    return TaskReport(task_id=task.id, status="done", summary="PDF downloaded.")
                if task.id == "t2":
                    await emit(
                        type="tool_call",
                        title="Call pdf_extract",
                        summary="Extract PDF.",
                        task_id=task.id,
                        tool="pdf_extract",
                    )
                    await emit(
                        type="tool_result",
                        title="pdf_extract: ok",
                        summary="Indexed chunks.",
                        task_id=task.id,
                        tool="pdf_extract",
                    )
                    return TaskReport(task_id=task.id, status="done", summary="PDF extracted.")
                if task.id in {"t3", "t4"}:
                    await emit(
                        type="tool_call",
                        title="Call rag_search",
                        summary="Search indexed chunks.",
                        task_id=task.id,
                        tool="rag_search",
                    )
                    ref = store.write_artifact(run_id, f"{task.id}-evidence", f"Evidence for {task.title}.")
                    await emit(
                        type="tool_result",
                        title="rag_search: ok",
                        summary="Found relevant evidence.",
                        task_id=task.id,
                        tool="rag_search",
                        artifact_ref=ref,
                    )
                    await emit(
                        type="tool_call",
                        title="Call memory_get",
                        summary="Read bounded artifact.",
                        task_id=task.id,
                        tool="memory_get",
                    )
                    await emit(
                        type="tool_result",
                        title="memory_get: ok",
                        summary="Loaded artifact snippet.",
                        task_id=task.id,
                        tool="memory_get",
                    )
                    return TaskReport(
                        task_id=task.id,
                        status="done",
                        summary=f"Collected evidence for {task.title}.",
                        key_facts=[{"claim": f"{task.title} is supported.", "ref": ref}],
                        refs=[ref],
                    )

                state_refs = store.read_json(run_id, "state.json", {}).get("refs", [])
                method_ref = next(ref for ref in state_refs if "t3" in ref)
                findings_ref = next(ref for ref in state_refs if "t4" in ref)
                final = (
                    "## Evidence Brief\n\n"
                    f"The paper has method evidence. [ref: {method_ref}]\n\n"
                    f"The paper has findings evidence. [ref: {findings_ref}]\n"
                )
                store.write_final(run_id, final)
                await emit(
                    type="final",
                    title="Final brief written",
                    summary="Final markdown written.",
                    task_id=task.id,
                )
                return TaskReport(
                    task_id=task.id,
                    status="done",
                    summary="Final brief written.",
                    refs=[method_ref, findings_ref],
                )

            with patch("evidence_agent.loop.Executor.execute", new=fake_execute):
                await orchestrator.execute_existing_plan(run_id, step_delay_ms=0)

            snapshot = store.snapshot(run_id)
            events = snapshot["events"]
            titles = event_titles(events)
            final = snapshot["final"]
            refs = cited_refs(final)
            missing_refs = [ref for ref in refs if not (Path(tmpdir) / run_id / ref).exists()]

            self.assertEqual(snapshot["state"]["status"], RunStatus.DONE.value)
            self.assertTrue(final)
            self.assertEqual(len([task for task in snapshot["plan"]["tasks"] if task["produces_final"]]), 1)
            self.assertIn("User approved plan", titles)
            self.assertIn("Spawn parallel executors", titles)
            self.assertIn("Call pdf_download", titles)
            self.assertIn("pdf_download: ok", titles)
            self.assertIn("Call pdf_extract", titles)
            self.assertIn("pdf_extract: ok", titles)
            self.assertIn("Call rag_search", titles)
            self.assertIn("Call memory_get", titles)
            self.assertIn("Final brief written", titles)
            self.assertIn("Run complete", titles)
            self.assertLess(titles.index("User approved plan"), titles.index("Executor starts t1"))
            self.assertLess(titles.index("Call pdf_download"), titles.index("Call pdf_extract"))
            self.assertNotIn("[Evidence:", final)
            self.assertGreaterEqual(len(refs), 2)
            self.assertEqual(missing_refs, [])
            for ref in refs:
                self.assertTrue(ref.startswith("artifacts/"))

    async def test_blocked_acquisition_does_not_create_fake_final_when_replan_budget_is_exhausted(self) -> None:
        """Bad-source behavior should be a blocked run, not a hallucinated final answer."""

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_dir=Path(tmpdir))
            orchestrator = Orchestrator(store)
            run_id = store.create_run("Read a bad paper URL and write a brief.", [])
            store.update_state(run_id, {"replan_count": 2})
            store.write_plan(
                run_id,
                Plan(
                    goal="Read a bad paper URL and write a brief.",
                    tasks=[
                        AgentTask(
                            id="t1",
                            title="Download source",
                            description="Download bad URL.",
                            requires=["pdf_download"],
                        ),
                        AgentTask(
                            id="t2",
                            title="Final synthesis brief",
                            description="Write final brief.",
                            depends_on=["t1"],
                            produces_final=True,
                        ),
                    ],
                ),
            )

            async def fake_blocked_download(executor, run_id, task, goal, source_paths, emit):
                await emit(type="executor", title=f"Executor starts {task.id}", summary=task.title, task_id=task.id)
                await emit(
                    type="tool_call",
                    title="Call pdf_download",
                    summary="Try bad URL.",
                    task_id=task.id,
                    tool="pdf_download",
                )
                await emit(
                    type="error",
                    title="pdf_download: failed",
                    summary="Remote URL did not return a PDF.",
                    task_id=task.id,
                    tool="pdf_download",
                    status="failed",
                )
                return TaskReport(
                    task_id=task.id,
                    status="blocked",
                    summary="Could not acquire a valid PDF.",
                    open_questions=["Provide a valid PDF URL or upload the paper."],
                )

            with patch("evidence_agent.loop.Executor.execute", new=fake_blocked_download):
                await orchestrator.execute_existing_plan(run_id, step_delay_ms=0)

            snapshot = store.snapshot(run_id)
            titles = event_titles(snapshot["events"])

        self.assertEqual(snapshot["state"]["status"], RunStatus.BLOCKED.value)
        self.assertFalse(snapshot["final"])
        self.assertIn("pdf_download: failed", titles)
        self.assertIn("No ready tasks", titles)
        self.assertIn("Run complete", titles)
        self.assertIn("Run blocked", snapshot["state"]["next_action"])
        self.assertIn("Provide a valid PDF URL or upload the paper.", snapshot["state"]["open_questions"])
