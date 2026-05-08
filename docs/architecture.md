# Architecture

## Goal

Evidence Brief Agent demonstrates a planner-executor agent without using an agent framework. It turns a research goal into an approved plan, runs tool-using executor loops, and writes a cited final brief.

## Runtime Flow

```text
Browser UI
  -> FastAPI
  -> Orchestrator
      -> optional prior-run conversation summary
      -> Planner ReAct loop
      -> Plan review / feedback
      -> Executor ReAct loops
      -> Tool calls
      -> State and trace files
      -> Final cited brief
```

## Main Components

| Component | File | Responsibility |
|---|---|---|
| FastAPI app | `src/evidence_agent/app.py` | HTTP API, uploads, static UI, background run workers |
| Orchestrator | `src/evidence_agent/loop.py` | Run lifecycle, dispatch, parallel batches, replan, final status |
| Planner | `src/evidence_agent/planner.py` | ReAct planning loop, plan validation, plan revision |
| Executor | `src/evidence_agent/executor.py` | ReAct task execution, tool selection, final brief contract |
| RunStore | `src/evidence_agent/state/run_store.py` | Filesystem persistence, events, artifacts, indexes, state |
| Tools | `src/evidence_agent/tools/` | Filesystem, PDF, browser, RAG, memory tools |
| Context | `src/evidence_agent/context.py` | Context estimation and optional contextual summarization |

## Planner

The planner creates a task DAG before execution starts.

It can:

- inspect available local files with read-only tools,
- incorporate user feedback from Plan Mode,
- revise a plan after runtime failure,
- use compressed prior-run context for follow-up questions,
- validate dependencies and final-task constraints.

Planner output must contain exactly one final synthesis task marked `produces_final=true`.

## Executor

Each executor receives one task and runs a ReAct loop:

```text
observe task and state
choose call_tool / complete_task / block_task
execute one tool if requested
record observation
repeat until done or blocked
```

The executor can call PDF, browser, memory, RAG, and safe filesystem tools. Final synthesis tasks must return `completion.final_markdown`; otherwise they fail.

If a run is a follow-up, the executor receives the same compressed `conversation_context` as the planner. It can answer requests such as "make the previous answer shorter" or "explain the method more" without re-downloading sources unless the task requires new evidence.

## Parallelism

The orchestrator dispatches all currently ready tasks marked `parallelizable=true`. Non-parallelizable tasks run alone. This keeps dependency order explicit while allowing independent evidence-gathering tasks to run concurrently.

## Replan And Human Recovery

If a task fails or blocks, the orchestrator can ask the planner to revise the remaining DAG while preserving completed tasks. Automatic replans are capped. After the cap, the run becomes blocked and the user can provide recovery feedback in the same run.

## Chat Session Model

The product has two levels of state:

- **Run state**: one immutable task graph and trace under `runs/<run_id>/`.
- **Chat session state**: the browser's current chain of runs.

By default, a new prompt in the same browser session sends the current `run_id` as `parent_run_id`. The backend reads the parent run, creates a compact `conversation_context.json`, and injects that summary into the next planner/executor prompts.

Clicking **New session** clears the browser session chain. The next prompt starts without `parent_run_id`, so it does not inherit prior context.

## Context Strategy

Default behavior uses the selected OpenAI model's native context window. The optional `EVIDENCE_AGENT_CONTEXT_TOKENS` environment variable enables a small context budget for stress testing.

Planner context is task-level:

- user goal,
- optional `conversation_context` from the previous run,
- feedback,
- previous plan,
- task reports,
- blocked or failed tasks,
- current state.

Executor context is task-level:

- assigned task,
- optional `conversation_context` from the previous run,
- acceptance criteria,
- current state,
- recent observations,
- retrieved refs and snippets.

Raw evidence is not kept directly in every prompt. It is written to `artifacts/` and retrieved by ref through `memory_get`.

Prior-run context is also not loaded raw. The backend summarizes the parent run's goal, final answer, task reports, refs, and open questions before passing it forward.

## Persistence

Each run is stored under `runs/<run_id>/`:

```text
events.jsonl       # full trace
state.json         # machine-readable state
current_state.md   # human-readable state
plan.json          # task DAG and task reports
conversation.json  # parent run id, when this is a follow-up
conversation_context.json # compressed prior-run context, when present
index.jsonl        # searchable evidence refs
artifacts/         # extracted chunks and fetched pages
final.md           # final cited brief
```

This makes the agent inspectable without an external tracing service.
