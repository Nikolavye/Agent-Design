# Evidence Brief Agent

A framework-light AI agent that reads research sources, plans the work, executes the plan with real tools, and produces a cited evidence brief.

The implementation does not use LangChain, LangGraph, AutoGen, CrewAI, or other agent frameworks. The planner loop, executor loop, tool routing, state store, RAG index, and trace logging are implemented directly in Python.

## Where to start

Read in this order:

1. **This README** — what the agent does and how to run it.
2. **`docs/architecture.md`** — system shape, planner/executor split, parallel execution, replan flow.
3. **`docs/prompt-contract.md`** — prompts as a JSON protocol; the framing that ties the system together.
4. **`docs/evaluation.md` + `tests/`** — what success looks like and how it is verified.
5. **`docs/example-transcript.md`** — what a real run produces end to end.

## Problem

The take-home asks for an agent that can:

1. accept a high-level user goal,
2. create a task plan,
3. use tools to execute the plan,
4. manage context explicitly,
5. produce a final answer with an auditable execution trace.

This project implements that as a research evidence agent for papers and PDFs.

## What The Agent Does

Default demo goal:

```text
Read this paper: https://arxiv.org/abs/2601.06700. Summarize what the paper is doing, its main results and findings, and why it is valuable. Produce a structured evidence brief with citations for key factual claims.
```

Flow:

1. Planner runs a ReAct loop and proposes a task DAG.
2. The UI pauses for human review and feedback.
3. Executor workers run ReAct loops with PDF, browser, RAG, memory, and filesystem tools.
4. Independent tasks can run in parallel.
5. Failed or blocked work can trigger replan or human recovery.
6. The final brief cites local artifact refs such as `[ref: artifacts/...txt]`.
7. Follow-up questions in the same chat session inherit a compressed summary of the previous run.

## Chat Sessions

The UI behaves like a chat:

- after a prompt is submitted, the input box is cleared for the next question;
- the next question in the same session is treated as a follow-up;
- the backend attaches a compressed `conversation_context` from the previous run;
- clicking **New session** clears the visible chat chain and starts the next question without prior context.

This keeps normal follow-ups useful while still preserving each run under its own `runs/<run_id>/` directory for auditability.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
cp .env.example .env
```

Add your OpenAI key to `.env`:

```bash
OPENAI_API_KEY=...
EVIDENCE_AGENT_MODEL=gpt-5.1
```

Start the app:

```bash
make demo
```

Open:

```text
http://127.0.0.1:8000
```

## Recommended Demo Workflow

For the smoothest review experience, open this repository in **Claude Code** or **Codex**. Use the agent terminal to install dependencies and start the app, then open the local URL above in a browser.

In the UI:

1. Click one of the three prompt cards on the start screen.
2. Click **Create plan** and review the planner's task DAG.
3. Click **Approve & run** to watch the executor trace, tool calls, refs, and final brief.
4. Ask a follow-up question in the same chat to test session memory and prior-run context.
5. Click **New session** when you want to start a clean conversation without previous context.

The three cards are designed to cover the main demo paths: paper summary, first-time understanding, and discussion/interview preparation.

## Tests

```bash
make lint
make test
```

The tests are offline and do not call the OpenAI API. They cover planner validation, executor contracts, context compaction, API state guards, path safety, replan behavior, and final citation validation.

## Tools

| Tool | Purpose |
|---|---|
| `fs_list` | List allowed local files |
| `fs_read` | Read bounded text slices |
| `pdf_download` | Download remote PDFs |
| `pdf_extract` | Extract and index PDF text |
| `rag_search` | Search indexed PDF/web chunks |
| `browser_search` | Search the web |
| `browser_fetch` | Fetch and index web pages |
| `memory_search` | Search run memory and artifact summaries |
| `memory_get` | Load bounded artifact snippets |

## Trace Files

Each run writes:

```text
runs/<run_id>/events.jsonl       # planner, executor, tool, state, final events
runs/<run_id>/current_state.md   # compact working state
runs/<run_id>/index.jsonl        # searchable evidence refs
runs/<run_id>/conversation.json  # parent run id for follow-up questions
runs/<run_id>/conversation_context.json # compressed prior-run context, if any
runs/<run_id>/artifacts/         # extracted chunks and fetched pages
runs/<run_id>/final.md           # final cited brief
```

The app also mirrors system logs to `logs/`.

## Context Strategy

By default, the planner and executor use the selected OpenAI model's native context window. For stress testing, set `EVIDENCE_AGENT_CONTEXT_TOKENS` to enable an explicit context budget. When enabled, oversized planner/executor context is compressed through an LLM contextual summary before deterministic fallback trimming.

The planner sees task-level state: goals, feedback, previous plans, task reports, blockers, and dependencies. The executor sees task-level working memory: current state, recent observations, retrieved refs, and bounded snippets.

For follow-up questions, the agent does not copy the previous run's full trace into the next prompt. It first summarizes the previous goal, final answer, task reports, refs, and open questions into `conversation_context.json`, then gives that compact context to the next planner/executor. `New session` disables this parent context for the next question.

## Deliverables

| Requirement | Location |
|---|---|
| Source code | `src/evidence_agent/` |
| Frontend | `src/evidence_agent/static/` |
| README | `README.md` |
| Architecture | `docs/architecture.md` |
| Example transcript | `docs/example-transcript.md` |
| Evaluation | `docs/evaluation.md` |
| Prompt/context contract | `docs/prompt-contract.md` |

## Repository Hygiene

The repository intentionally ships without `.env`, downloaded PDFs, run artifacts, logs, virtual environments, or caches. `.env.example` documents the local environment variables. Runtime directories such as `runs/`, `logs/`, and `data/pdfs/` are created automatically when the app runs.
