from __future__ import annotations

from pathlib import Path

from evidence_agent.state.run_store import DATA_DIR, PDF_DIR, PROJECT_ROOT, safe_resolve
from evidence_agent.types import SourceFile, ToolResult

READABLE_SOURCE_ROOTS = [DATA_DIR, PROJECT_ROOT / "docs"]


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".txt", ".md", ".json", ".csv", ".py", ".ts", ".tsx", ".html", ".css"}:
        return "text"
    return "other"


def list_sources() -> list[SourceFile]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    candidates = list(PDF_DIR.glob("*.pdf")) + list((PROJECT_ROOT / "docs").glob("*.md"))
    return [
        SourceFile(
            name=path.name,
            path=str(path.relative_to(PROJECT_ROOT)),
            kind=classify_file(path),  # type: ignore[arg-type]
            size_bytes=path.stat().st_size,
        )
        for path in sorted(candidates)
        if path.is_file()
    ]


def fs_list(root: str = "data", glob: str = "**/*", max_results: int = 50) -> ToolResult:
    resolved = safe_resolve(root, allowed_roots=READABLE_SOURCE_ROOTS)
    paths = []
    for path in resolved.glob(glob):
        if path.is_file():
            try:
                paths.append(str(path.relative_to(PROJECT_ROOT)))
            except ValueError:
                paths.append(str(path))
        if len(paths) >= max_results:
            break
    return ToolResult(ok=True, summary=f"Found {len(paths)} files.", data={"paths": paths})


def fs_read(path: str, start_line: int = 1, max_chars: int = 4000) -> ToolResult:
    resolved = safe_resolve(path, allowed_roots=READABLE_SOURCE_ROOTS)
    if resolved.suffix.lower() == ".pdf":
        return ToolResult(
            ok=False,
            summary=f"{resolved.name} is a PDF. Use pdf_extract for PDFs instead of fs_read.",
            data={"path": path, "recommended_tool": "pdf_extract"},
        )
    text = resolved.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    selected = "\n".join(lines[max(0, start_line - 1) :])
    selected = selected[:max_chars]
    summary = f"Read {len(selected)} characters from {resolved.name}."
    return ToolResult(ok=True, summary=summary, data={"path": path, "text": selected})
