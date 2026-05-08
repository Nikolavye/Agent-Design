# Example Transcript

This is the expected shape of a successful default run. Event ids and exact task names may vary.

## User Goal

```text
Read this paper: https://arxiv.org/abs/2601.06700. Summarize what the paper is doing, its main results and findings, and why it is valuable. Produce a structured evidence brief with citations for key factual claims.
```

## Trace Shape

```text
evt_0001 Planner observe
evt_0002 Planner ReAct decision
evt_0003 Planner completed task DAG
evt_0004 Plan ready for review

user approves plan

evt_0005 User approved plan
evt_0006 Executor starts t1
evt_0007 Call pdf_download
evt_0008 pdf_download: ok
evt_0009 Task completed

evt_0010 Executor starts t2
evt_0011 Call pdf_extract
evt_0012 pdf_extract: ok
evt_0013 Task completed

evt_0014 Spawn parallel executors
evt_0015 Executor starts t3
evt_0016 Executor starts t4
evt_0017 Executor starts t5
evt_0018 Call rag_search
evt_0019 rag_search: ok
evt_0020 Call memory_get
evt_0021 memory_get: ok

evt_0022 Executor starts final task
evt_0023 Call memory_search
evt_0024 memory_search: ok
evt_0025 Final brief written
evt_0026 Run complete
```

## Follow-Up Shape

If the user asks a follow-up in the same chat session:

```text
User: Give me a one-sentence version.

evt_0001 Previous run context attached
evt_0002 Planner observe
evt_0003 Planner ReAct decision
evt_0004 Planner completed task DAG
evt_0005 Plan ready for review

user approves plan

evt_0006 User approved plan
evt_0007 Executor starts t1
evt_0008 Final brief written
evt_0009 Run complete
```

The follow-up run receives `conversation_context.json`, not the full previous trace. Clicking **New session** starts a clean chain and skips this step.

## Final Output Shape

```markdown
## Working Conclusion

Short answer with cited claims. [ref: artifacts/...txt]

## What The Paper Does

Explanation of the research goal. [ref: artifacts/...txt]

## Method And Data

Summary of the method and evidence source. [ref: artifacts/...txt]

## Results And Findings

Key findings with citations. [ref: artifacts/...txt; artifacts/...txt]

## Why It Is Valuable

Practical or research value. [ref: artifacts/...txt]

## Caveats

Limitations or open questions.
```

## Files Created

```text
runs/<run_id>/events.jsonl
runs/<run_id>/plan.json
runs/<run_id>/conversation.json
runs/<run_id>/conversation_context.json # only for follow-up runs
runs/<run_id>/index.jsonl
runs/<run_id>/artifacts/
runs/<run_id>/final.md
```
