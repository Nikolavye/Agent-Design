from __future__ import annotations

from evidence_agent.tools.pdf_tools import score_records
from evidence_agent.types import ToolResult


def memory_search(store, run_id: str, query: str, max_results: int = 8) -> ToolResult:
    records = store.read_index(run_id)
    results = score_records(query, records)[:max_results]
    compact = [
        {
            "ref": item.get("ref"),
            "kind": item.get("kind"),
            "source": item.get("source"),
            "summary": item.get("summary"),
            "score": item.get("score"),
            "page": item.get("page"),
        }
        for item in results
    ]
    return ToolResult(ok=True, summary=f"Found {len(compact)} memory refs.", data={"query": query, "results": compact})


def memory_get(store, run_id: str, ref: str, max_chars: int = 3500) -> ToolResult:
    text = store.read_artifact(run_id, ref, max_chars=max_chars)
    return ToolResult(ok=True, summary=f"Loaded {len(text)} characters from {ref}.", data={"ref": ref, "text": text})
