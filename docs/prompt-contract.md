# Prompt Contract

The prompts are written as runtime protocols. The model is not asked to produce free-form prose during planning or execution. It must return one JSON object matching a small action schema.

## Planner Actions

The planner returns one of:

```json
{"thought_summary": "...", "action": "call_tool", "tool": "fs_list", "args": {}}
```

```json
{"thought_summary": "...", "action": "complete_plan", "plan": {"tasks": []}}
```

```json
{"thought_summary": "...", "action": "block_plan", "open_questions": []}
```

Planner rules:

- do not execute the research task,
- only inspect local files through read-only tools,
- produce a task DAG with dependencies,
- mark exactly one task as `produces_final=true`,
- preserve completed work during replan,
- use `conversation_context` when the current user goal is a follow-up.

## Executor Actions

The executor returns one of:

```json
{"thought_summary": "...", "action": "call_tool", "tool": "memory_search", "args": {}}
```

```json
{"thought_summary": "...", "action": "complete_task", "completion": {"summary": "...", "refs": []}}
```

```json
{"thought_summary": "...", "action": "block_task", "completion": {"summary": "...", "open_questions": []}}
```

Executor rules:

- call at most one tool per step,
- use search tools before `memory_get`,
- use `memory_get` for `artifacts/...` refs, not `fs_read`,
- restate acceptance criteria before completing a task,
- include `final_markdown` only for final tasks,
- use `conversation_context` before reacquiring sources for shorten/reframe/explain follow-ups.

## Conversation Context Contract

Follow-up runs receive a compact prior-run summary:

```json
{
  "summary": "what the previous question and answer established",
  "must_keep": ["facts, constraints, user preferences, or conclusions"],
  "task_statuses": [{"id": "t1", "status": "done", "note": "..."}],
  "evidence_refs": ["artifacts/..."],
  "open_questions": [],
  "parent_run_id": "..."
}
```

Rules:

- preserve artifact refs exactly,
- do not copy the full previous trace into the prompt,
- use the summary for follow-up questions in the same chat session,
- do not attach it after the user clicks **New session**.

## Citation Contract

Final factual claims must cite real artifact refs:

```text
The paper evaluates toxicity with a classifier. [ref: artifacts/pdf-paper-p3-chunk2.txt]
```

Multiple refs use one bracket:

```text
[ref: artifacts/a.txt; artifacts/b.txt]
```

Invalid examples:

```text
[ref: external paper]
[Evidence: internal]
[source: research]
```

The backend validates refs before accepting task completion. Invalid refs fail the task.
