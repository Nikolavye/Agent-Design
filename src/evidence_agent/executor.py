from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from evidence_agent.context import DEFAULT_CONTEXT_TOKEN_BUDGET, compact_executor_context
from evidence_agent.llm import LLMClient
from evidence_agent.state.run_store import RunStore
from evidence_agent.tools.browser_tools import browser_fetch, browser_search
from evidence_agent.tools.fs_tools import fs_list, fs_read
from evidence_agent.tools.memory_tools import memory_get, memory_search
from evidence_agent.tools.pdf_tools import extract_pdf_to_index, pdf_download, rag_search
from evidence_agent.types import AgentTask, TaskReport, ToolResult

Emit = Callable[..., Awaitable[None]]
REF_PATTERN = re.compile(r"\[ref:\s*([^\]]+)\]", re.IGNORECASE)


REACT_SYSTEM = """You are the Executor node in a planner-executor research agent.
You execute exactly one task at a time by running a visible ReAct loop:

1. Observe the task, available sources, current state, and previous observations.
2. Decide the next action.
3. Call exactly one tool, or complete/block the task.
4. Use the tool observation in the next step.

Response contract:
- Return exactly one JSON object. No markdown, no prose outside JSON, no arrays as the root.
- The top-level object must contain `thought_summary` and `action`.
- `thought_summary` must be a short operational summary, not hidden chain-of-thought.
- `action` must be exactly one of: `call_tool`, `complete_task`, `block_task`.
- Call at most one tool per response.
- Do not emit extra top-level keys beyond the selected schema.

Allowed response schemas:

1. Tool call:
{
  "thought_summary": "why this tool is the next useful action",
  "action": "call_tool",
  "tool": "fs_list | fs_read | pdf_download | pdf_extract | rag_search | browser_search | browser_fetch | memory_search | memory_get",
  "args": {}
}

2. Complete task:
{
  "thought_summary": "why the task is complete",
  "action": "complete_task",
  "completion": {
    "summary": "...",
    "key_facts": [{"claim": "...", "ref": "..."}],
    "refs": ["..."],
    "open_questions": [],
    "recommended_next_tasks": [],
    "final_markdown": "only include for final synthesis tasks"
  }
}

3. Block task:
{
  "thought_summary": "what is missing",
  "action": "block_task",
  "completion": {
    "summary": "...",
    "open_questions": ["..."]
  }
}

Tool argument contracts:
- fs_list: {"root": "data", "glob": "**/*", "max_results": 50}
- fs_read: {"path": "...", "start_line": 1, "max_chars": 4000}
- pdf_download: {"url": "https://.../paper.pdf", "filename": "optional-name.pdf", "max_mb": 25}
- pdf_extract: {"path": "data/pdfs/file.pdf"}
- rag_search: {"query": "...", "max_results": 6}
- browser_search: {"query": "...", "max_results": 5}
- browser_fetch: {"url": "...", "max_chars": 9000}
- memory_search: {"query": "...", "max_results": 8}
- memory_get: {"ref": "artifacts/...", "max_chars": 3500}

Execution policy by phase:
Session follow-up:
- If `conversation_context` is present, treat the current task as part of the same chat session.
- Use prior summary, must_keep, open_questions, and evidence_refs to answer follow-up requests without unnecessarily repeating source acquisition.
- If the follow-up asks to shorten, reframe, explain, or extend the previous answer, use `conversation_context` first and retrieve extra evidence only when the task acceptance criteria requires it.

Phase 1 — Acquire source:
- If the task needs a remote paper page or PDF URL, call `pdf_download`, then use returned local `path` with `pdf_extract`.
- If the task needs already selected local PDFs indexed, call `pdf_extract` for each selected PDF that still needs extraction.
- If the task needs web context, use `browser_search` before `browser_fetch`.

Phase 2 — Retrieve evidence:
- Prefer `rag_search` or `memory_search` before `memory_get`.
- Artifact refs such as `artifacts/name.txt` are not filesystem paths; use `memory_get`, not `fs_read`.
- Use `memory_get` only after search/tool observations expose relevant refs.

Phase 3 — Complete:
- Before `complete_task`, `thought_summary` must restate the acceptance criteria and say how it was satisfied.
- For non-final tasks, do not include `final_markdown`; complete with summary, key_facts, refs, and optional recommended_next_tasks.
- If `task.produces_final` is true, the task is not complete until `completion.final_markdown` contains the final brief.

Phase 4 — Failure handling:
- If a tool fails, adjust args once; do not repeat the same failing tool call with identical args.
- Usually block when two searches in the same query family return no useful refs.
- Usually block when a tool budget is exhausted and collected refs are below the task acceptance threshold.
- If evidence is partial but acceptance criteria are satisfied, complete with scoped facts instead of continuing tool calls.
- Stop when the acceptance criteria is satisfied; do not keep calling tools unnecessarily.

Evidence and citation contract:
- `key_facts` must contain only claims supported by retrieved snippets or tool observations.
- Every `key_facts[].ref` must exactly match a real artifact ref returned by tools, such as `artifacts/name.txt`.
- `completion.refs` must be the unique list of artifact refs used by the task.
- Never invent placeholder refs such as "internal", "external background", "paper", or raw URLs.
- If a claim has no artifact ref, put it in `open_questions` or phrase it as uncited context outside `key_facts`.

Citations — examples:
✓ The authors evaluate toxicity with a classifier. [ref: artifacts/pdf-foo-p3-chunk2.txt]
✓ The paper compares two model families. [ref: artifacts/pdf-foo-p2-chunk1.txt; artifacts/pdf-foo-p8-chunk2.txt]
✗ The authors evaluate toxicity with a classifier. [ref: external paper]
✗ The paper compares two model families. [Evidence: internal]
✗ As stated by the paper, ... [source: research]

Final markdown contract, only when `task.produces_final` is true:
- Use concise Markdown sections.
- Include sections covering: what the paper does, method/data, main results/findings, value, limitations or caveats.
- Put citations immediately after factual claims using exactly this lowercase clickable form: `[ref: artifacts/example.txt]`.
- For multiple citations, use one bracket with refs separated by semicolons: `[ref: artifacts/a.txt; artifacts/b.txt]`.
- Use only artifact refs that appeared in `memory_search`, `rag_search`, `memory_get`, or tool observations.
- Do not use vague citation labels like "[Evidence: internal]" or invented source names.

Example final task completion:
{
  "thought_summary": "Acceptance was 'write a cited final brief'. Retrieved method/results refs and final_markdown cites them. Met.",
  "action": "complete_task",
  "completion": {
    "summary": "Wrote a 5-section evidence brief with cited claims.",
    "key_facts": [
      {
        "claim": "The authors evaluate toxicity with a classifier.",
        "ref": "artifacts/pdf-foo-p3-chunk2.txt"
      }
    ],
    "refs": ["artifacts/pdf-foo-p3-chunk2.txt"],
    "open_questions": [],
    "recommended_next_tasks": [],
    "final_markdown": "## Background\nThe authors evaluate toxicity with a classifier. [ref: artifacts/pdf-foo-p3-chunk2.txt]\n"
  }
}

Now decide your next ReAct action and emit one JSON object.
"""


class Executor:
    def __init__(
        self,
        store: RunStore,
        llm: LLMClient,
        step_delay_ms: int = 700,
        max_steps: int = 12,
        context_token_budget: int | None = DEFAULT_CONTEXT_TOKEN_BUDGET,
    ):
        self.store = store
        self.llm = llm
        self.step_delay_ms = step_delay_ms
        self.max_steps = max_steps
        self.context_token_budget = context_token_budget
        self.tool_limits = {
            "browser_search": 3,
            "browser_fetch": 4,
            "pdf_download": 3,
            "memory_get": 8,
            "memory_search": 5,
            "rag_search": 5,
        }

    async def pause(self) -> None:
        if self.step_delay_ms:
            await asyncio.sleep(self.step_delay_ms / 1000)

    async def execute(self, run_id: str, task: AgentTask, goal: str, source_paths: list[str], emit: Emit) -> TaskReport:
        await emit(type="executor", title=f"Executor starts {task.id}", summary=task.title, task_id=task.id, status="running")
        await self.pause()

        if not self.llm.enabled:
            summary = "OPENAI_API_KEY is required for the ReAct executor."
            await emit(type="error", title="Executor cannot start", summary=summary, task_id=task.id, status="failed")
            return TaskReport(task_id=task.id, status="failed", summary=summary, open_questions=[summary])

        observations: list[dict[str, Any]] = []
        tool_counts: dict[str, int] = {}

        for step in range(1, self.max_steps + 1):
            await emit(
                type="executor",
                title=f"Observe task ({task.id}, step {step})",
                summary=self._observation_summary(task, observations),
                task_id=task.id,
                status="running",
                data={"observations": observations[-5:]},
            )
            await self.pause()

            action = await asyncio.to_thread(self._next_action, run_id, task, goal, source_paths, observations)
            context_budget = action.pop("_context_budget", None)
            if context_budget and context_budget.get("compacted"):
                await emit(
                    type="state",
                    title="Context budget applied",
                    summary=(
                        "Executor context compacted "
                        f"{context_budget['estimated_tokens_before']} -> {context_budget['estimated_tokens_after']} "
                        f"tokens under {context_budget['budget_tokens']}."
                    ),
                    task_id=task.id,
                    status="success",
                    data=context_budget,
                )
                await self.pause()
            await emit(
                type="executor",
                title=f"ReAct decision ({task.id}, step {step})",
                summary=action.get("thought_summary", "Executor selected the next action."),
                task_id=task.id,
                status="running",
                data={"action": action},
            )
            await self.pause()

            action_type = action.get("action")
            if action_type == "call_tool":
                tool_name = action.get("tool")
                args = action.get("args") or {}
                result = await self._call_tool(run_id, task, emit, tool_name, args, tool_counts)
                observations.append(
                    {
                        "step": step,
                        "tool": tool_name,
                        "ok": result.ok,
                        "summary": result.summary,
                        "artifact_ref": result.artifact_ref,
                        "data_preview": self._preview_data(result.data),
                    }
                )
                continue

            if action_type == "complete_task":
                return await self._complete_task(run_id, task, emit, action)

            if action_type == "block_task":
                completion = action.get("completion") or {}
                summary = completion.get("summary") or "Task blocked by executor."
                questions = completion.get("open_questions") or [summary]
                await emit(type="executor", title="Task blocked", summary=summary, task_id=task.id, status="failed", data=completion)
                return TaskReport(task_id=task.id, status="blocked", summary=summary, open_questions=questions)

            summary = f"Invalid ReAct action from model: {action}"
            await emit(type="error", title="Invalid executor action", summary=summary, task_id=task.id, status="failed")
            return TaskReport(task_id=task.id, status="failed", summary=summary)

        summary = f"Executor reached max_steps={self.max_steps} without completing the task."
        await emit(type="error", title="Executor step limit reached", summary=summary, task_id=task.id, status="failed")
        return TaskReport(task_id=task.id, status="failed", summary=summary)

    def _next_action(
        self,
        run_id: str,
        task: AgentTask,
        goal: str,
        source_paths: list[str],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        state = self.store.read_json(run_id, "state.json", {})
        conversation_context = self.store.read_json(run_id, "conversation_context.json", None)
        index = self.store.read_index(run_id)
        indexed_sources = sorted({item.get("source", "") for item in index if item.get("source")})
        context = {
            "goal": goal,
            "conversation_context": conversation_context,
            "task": task.model_dump(),
            "selected_sources": source_paths,
            "current_state": {
                "done": state.get("done", []),
                "facts": state.get("facts", [])[-8:],
                "refs": state.get("refs", [])[-8:],
                "next_action": state.get("next_action", ""),
            },
            "indexed_sources": indexed_sources[-20:],
            "recent_observations": observations[-8:],
            "known_public_urls": [
                "https://www.nature.com/articles/s41592-024-02523-z",
                "https://www.nature.com/articles/s41592-024-02523-z.pdf",
                "https://arxiv.org/abs/2601.01090",
                "https://arxiv.org/pdf/2601.01090.pdf",
                "https://arxiv.org/abs/2601.06226",
                "https://arxiv.org/pdf/2601.06226.pdf",
            ],
        }
        context, context_budget = compact_executor_context(context, self.context_token_budget, llm=self.llm)
        if context_budget.get("overflow"):
            return {
                "thought_summary": (
                    "Executor context stayed above budget after compaction, so the task is blocked "
                    "instead of calling the model with an oversized prompt."
                ),
                "action": "block_task",
                "_context_budget": context_budget,
                "completion": {
                    "summary": (
                        "Context budget exceeded after compaction: "
                        f"{context_budget['estimated_tokens_after']} > {context_budget['budget_tokens']} tokens."
                    ),
                    "open_questions": [
                        "Narrow the task, reduce retrieved evidence, raise EVIDENCE_AGENT_CONTEXT_TOKENS, or split the task into smaller steps."
                    ],
                },
            }
        action = self.llm.json(REACT_SYSTEM, json.dumps(context, ensure_ascii=False, default=str), default_response={})
        if not action and self.llm.last_error:
            return {
                "thought_summary": "Executor could not receive a valid LLM response.",
                "action": "block_task",
                "_context_budget": context_budget,
                "completion": {
                    "summary": f"LLM call failed: {self.llm.last_error}",
                    "open_questions": [f"Resolve LLM call failure: {self.llm.last_error}"],
                },
            }
        normalized = self._normalize_action(action, task, indexed_sources)
        if normalized:
            normalized["_context_budget"] = context_budget
        return normalized

    @staticmethod
    def _normalize_action(action: dict[str, Any], task: AgentTask, indexed_sources: list[str] | None = None) -> dict[str, Any]:
        if not action:
            return action

        requires = set(task.requires or [])
        indexed_sources = indexed_sources or []
        if action.get("action") == "call_tool":
            tool = str(action.get("tool") or "")
            args = action.get("args") or {}
            if tool == "fs_read" and str(args.get("path", "")).startswith("artifacts/"):
                return {
                    "thought_summary": "Model tried to read an artifact ref with fs_read; normalized into memory_get.",
                    "action": "call_tool",
                    "tool": "memory_get",
                    "args": {
                        "ref": args.get("path"),
                        "max_chars": args.get("max_chars", 3500),
                    },
                }
            return action
        if action.get("action"):
            return action

        if "root" in action or "glob" in action:
            return {
                "thought_summary": "Model returned bare fs_list arguments; normalized into a tool call.",
                "action": "call_tool",
                "tool": "fs_list",
                "args": action,
            }
        if "query" in action:
            if "memory_search" in requires:
                tool = "memory_search"
            elif "browser_search" in requires and not indexed_sources:
                tool = "browser_search"
            else:
                tool = "rag_search"
            return {
                "thought_summary": f"Model returned bare `{tool}` arguments; normalized into a tool call.",
                "action": "call_tool",
                "tool": tool,
                "args": action,
            }
        if "ref" in action:
            return {
                "thought_summary": "Model returned bare memory_get arguments; normalized into a tool call.",
                "action": "call_tool",
                "tool": "memory_get",
                "args": action,
            }
        if "url" in action:
            url = str(action.get("url", ""))
            tool = "pdf_download" if "pdf_download" in requires or "/pdf/" in url or url.lower().endswith(".pdf") else "browser_fetch"
            return {
                "thought_summary": f"Model returned bare `{tool}` arguments; normalized into a tool call.",
                "action": "call_tool",
                "tool": tool,
                "args": action,
            }
        if "path" in action:
            path = str(action.get("path", ""))
            if path.startswith("artifacts/"):
                return {
                    "thought_summary": "Model returned bare memory_get artifact arguments; normalized into a tool call.",
                    "action": "call_tool",
                    "tool": "memory_get",
                    "args": {
                        "ref": path,
                        "max_chars": action.get("max_chars", 3500),
                    },
                }
            tool = "pdf_extract" if "pdf_extract" in requires or path.lower().endswith(".pdf") else "fs_read"
            return {
                "thought_summary": f"Model returned bare `{tool}` arguments; normalized into a tool call.",
                "action": "call_tool",
                "tool": tool,
                "args": action,
            }
        return action

    async def _call_tool(
        self,
        run_id: str,
        task: AgentTask,
        emit: Emit,
        tool_name: str,
        args: dict[str, Any],
        tool_counts: dict[str, int],
    ) -> ToolResult:
        limit = self.tool_limits.get(str(tool_name))
        count = tool_counts.get(str(tool_name), 0)
        if limit is not None and count >= limit:
            result = ToolResult(
                ok=False,
                summary=f"Tool budget exceeded for {tool_name}: limit {limit}. Choose another tool or complete/block the task.",
                data={"tool": tool_name, "limit": limit, "count": count},
            )
            await emit(
                type="error",
                title=f"{tool_name}: budget exceeded",
                summary=result.summary,
                task_id=task.id,
                tool=str(tool_name),
                status="failed",
                data=result.data,
            )
            await self.pause()
            return result

        await emit(
            type="tool_call",
            title=f"Call {tool_name}",
            summary=f"Executor calls `{tool_name}` with bounded args.",
            task_id=task.id,
            tool=tool_name,
            status="running",
            data={"args": args},
        )
        await self.pause()

        try:
            result = await asyncio.to_thread(self._dispatch_tool, run_id, tool_name, args)
        except Exception as exc:
            result = ToolResult(ok=False, summary=f"{tool_name} failed: {exc}", data={"args": args})
        tool_counts[str(tool_name)] = count + 1

        await emit(
            type="tool_result" if result.ok else "error",
            title=f"{tool_name}: {'ok' if result.ok else 'failed'}",
            summary=result.summary,
            task_id=task.id,
            tool=tool_name,
            status="success" if result.ok else "failed",
            data=result.data,
            artifact_ref=result.artifact_ref,
        )
        await self.pause()
        return result

    def _dispatch_tool(self, run_id: str, tool_name: str, args: dict[str, Any]) -> ToolResult:
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
                max_chars=int(args.get("max_chars", 4000)),
            )
        if tool_name == "pdf_extract":
            path = str(args.get("path") or args.get("source_path"))
            return extract_pdf_to_index(self.store, run_id, path)
        if tool_name == "pdf_download":
            return pdf_download(
                url=str(args["url"]),
                filename=args.get("filename"),
                max_mb=int(args.get("max_mb", 25)),
            )
        if tool_name == "rag_search":
            return rag_search(
                self.store,
                run_id,
                query=str(args["query"]),
                max_results=int(args.get("max_results", 6)),
            )
        if tool_name == "browser_search":
            return browser_search(
                query=str(args["query"]),
                max_results=int(args.get("max_results", 5)),
            )
        if tool_name == "browser_fetch":
            return browser_fetch(
                self.store,
                run_id,
                url=str(args["url"]),
                max_chars=int(args.get("max_chars", 9000)),
            )
        if tool_name == "memory_search":
            return memory_search(
                self.store,
                run_id,
                query=str(args["query"]),
                max_results=int(args.get("max_results", 8)),
            )
        if tool_name == "memory_get":
            return memory_get(
                self.store,
                run_id,
                ref=str(args["ref"]),
                max_chars=int(args.get("max_chars", 3500)),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    async def _complete_task(self, run_id: str, task: AgentTask, emit: Emit, action: dict[str, Any]) -> TaskReport:
        completion = action.get("completion") or {}
        summary = completion.get("summary") or "Task completed."
        key_facts = completion.get("key_facts") or []
        refs = completion.get("refs") or [fact.get("ref") for fact in key_facts if fact.get("ref")]
        open_questions = completion.get("open_questions") or []
        recommended_next_tasks = self._string_list(completion.get("recommended_next_tasks") or [])

        final_markdown = completion.get("final_markdown") if task.produces_final else None
        if completion.get("final_markdown") and not task.produces_final:
            await emit(
                type="executor",
                title="Ignored premature final markdown",
                summary="A non-final task returned final_markdown; executor ignored it and kept the final brief contract reserved for the produces_final task.",
                task_id=task.id,
                status="success",
                data={"ignored_final_markdown": True},
            )
            await self.pause()
        if task.produces_final and not final_markdown:
            summary = "Final synthesis task attempted to complete without `final_markdown`."
            await emit(
                type="error",
                title="Final brief contract failed",
                summary=summary,
                task_id=task.id,
                status="failed",
                data=completion,
            )
            await self.pause()
            return TaskReport(
                task_id=task.id,
                status="failed",
                summary=summary,
                open_questions=["The final task must return completion.final_markdown with the markdown evidence brief."],
            )
        refs = self._collect_refs(key_facts, refs, final_markdown)
        invalid_refs = self._invalid_artifact_refs(run_id, refs)
        if invalid_refs:
            summary = f"Task returned invalid artifact refs: {', '.join(invalid_refs[:5])}"
            await emit(
                type="error",
                title="Artifact ref contract failed",
                summary=summary,
                task_id=task.id,
                status="failed",
                data={"invalid_refs": invalid_refs, "completion": completion},
            )
            await self.pause()
            return TaskReport(
                task_id=task.id,
                status="failed",
                summary=summary,
                open_questions=["Use only artifact refs returned by search, retrieval, or tool observations."],
            )
        if final_markdown:
            self.store.write_final(run_id, final_markdown)
            await emit(
                type="final",
                title="Final brief written",
                summary="The ReAct executor completed the final markdown brief.",
                task_id=task.id,
                status="success",
                data={"refs": refs},
            )
        else:
            await emit(type="executor", title="Task completed", summary=summary, task_id=task.id, status="success", data=completion)
        await self.pause()

        return TaskReport(
            task_id=task.id,
            status="done",
            summary=summary,
            key_facts=key_facts,
            refs=refs,
            open_questions=open_questions,
            recommended_next_tasks=recommended_next_tasks,
        )

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return [str(value)] if value else []
        normalized: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                title = item.get("title") or item.get("id") or item.get("description")
                normalized.append(str(title) if title else json.dumps(item, ensure_ascii=False))
            elif item:
                normalized.append(str(item))
        return normalized

    @staticmethod
    def _collect_refs(key_facts: Any, refs: Any, final_markdown: str | None) -> list[str]:
        collected: list[str] = []
        for ref in Executor._string_list(refs):
            collected.extend(Executor._split_refs(ref))
        if isinstance(key_facts, list):
            for fact in key_facts:
                if isinstance(fact, dict) and fact.get("ref"):
                    collected.extend(Executor._split_refs(str(fact["ref"])))
        if final_markdown:
            for match in REF_PATTERN.findall(final_markdown):
                collected.extend(Executor._split_refs(match))
        return list(dict.fromkeys(ref for ref in collected if ref))

    @staticmethod
    def _split_refs(value: str) -> list[str]:
        return [item.strip() for item in value.split(";") if item.strip()]

    def _invalid_artifact_refs(self, run_id: str, refs: list[str]) -> list[str]:
        invalid: list[str] = []
        for ref in refs:
            if not ref.startswith("artifacts/"):
                invalid.append(ref)
                continue
            try:
                self.store.read_artifact(run_id, ref, max_chars=1)
            except Exception:
                invalid.append(ref)
        return invalid

    @staticmethod
    def _observation_summary(task: AgentTask, observations: list[dict[str, Any]]) -> str:
        if not observations:
            return f"Task: {task.title}. No tool observations yet."
        last = observations[-1]
        return f"Task: {task.title}. Last observation: {last.get('tool')} -> {last.get('summary')}"

    @staticmethod
    def _preview_data(data: dict[str, Any]) -> dict[str, Any]:
        preview: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                preview[key] = value[:500]
            elif isinstance(value, list):
                preview[key] = value[:5]
            else:
                preview[key] = value
        return preview
