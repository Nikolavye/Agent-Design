# Evaluation

The project is evaluated through offline tests plus one live demo run.

## Offline Tests

Run:

```bash
make lint
make test
```

The test suite covers:

- planner DAG validation,
- dependency cycle rejection,
- final-task contract,
- runtime replan behavior,
- human recovery state transitions,
- safe artifact path handling,
- PDF download guardrails,
- context compaction behavior,
- API state guards,
- final citation ref validation,
- trace/process contracts over `events.jsonl`,
- outcome checks for a simulated full arXiv run,
- blocked-source behavior that must not create a fake final brief.

Expected result:

```text
all tests pass
```

## What Is Not Measured

This project does not report retrieval `precision@k` or `recall@k` because the
repository does not include a manually labeled gold set of relevant chunks.
Those metrics would be appropriate after creating a labeled corpus. The current
evaluation focuses on deterministic agent-contract checks, trace correctness,
citation validity, and live-demo behavior.

## Live Demo Check

Start the app:

```bash
make demo
```

Open:

```text
http://127.0.0.1:8000
```

Use the default prompt:

```text
Read this paper: https://arxiv.org/abs/2601.06700. Summarize what the paper is doing, its main results and findings, and why it is valuable. Produce a structured evidence brief with citations for key factual claims.
```

Success criteria:

1. Planner creates a visible task DAG.
2. UI pauses at plan review before execution.
3. Executor downloads and extracts the PDF.
4. Evidence tasks use RAG and memory tools.
5. Independent evidence tasks may run in parallel.
6. Final brief is written to `runs/<run_id>/final.md`.
7. Citations use `[ref: artifacts/...txt]`.
8. Every cited artifact exists under the run directory.

Follow-up check:

1. Ask a second question such as `Give me a one-sentence version.`
2. The new run should emit `Previous run context attached`.
3. The new run should create `conversation_context.json`.
4. The answer should use the prior run's compressed context instead of redoing the full PDF acquisition path unless new evidence is needed.
5. Click **New session**, then ask another question; the new run should not attach prior context.

## Failure Checks

Expected safe behavior:

- missing API key fails clearly,
- invalid artifact refs fail the task,
- unsafe artifact paths are rejected,
- tool budgets stop repeated calls,
- failed tasks can trigger replan or human recovery.
