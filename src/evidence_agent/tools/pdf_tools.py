from __future__ import annotations

import re
import threading
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader

from evidence_agent.state.run_store import PDF_DIR, PROJECT_ROOT, safe_resolve, slugify
from evidence_agent.types import ToolResult

PDF_WRITE_LOCK = threading.Lock()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 180) -> Iterable[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        yield cleaned[start:end]
        if end == len(cleaned):
            break
        start = max(0, end - overlap)


def normalize_pdf_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("pdf_download only accepts http(s) URLs.")

    path = parsed.path.rstrip("/")
    if parsed.netloc.endswith("arxiv.org") and path.startswith("/abs/"):
        paper_id = path.removeprefix("/abs/")
        return f"{parsed.scheme}://{parsed.netloc}/pdf/{paper_id}.pdf"
    if parsed.netloc.endswith("arxiv.org") and path.startswith("/pdf/") and not path.endswith(".pdf"):
        return f"{parsed.scheme}://{parsed.netloc}{path}.pdf"
    if "nature.com" in parsed.netloc and path.startswith("/articles/") and not path.endswith(".pdf"):
        return f"{parsed.scheme}://{parsed.netloc}{path}.pdf"
    return url.strip()


def _filename_from_response(url: str, content_disposition: str | None, requested_filename: str | None) -> str:
    if requested_filename:
        candidate = requested_filename
    elif content_disposition:
        match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', content_disposition, flags=re.IGNORECASE)
        candidate = match.group(1) if match else Path(urlparse(url).path).name
    else:
        candidate = Path(urlparse(url).path).name

    stem = slugify(Path(candidate).stem)
    return f"{stem}.pdf"


def _unique_pdf_path(filename: str) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    stem = slugify(Path(filename).stem)
    destination = PDF_DIR / f"{stem}.pdf"
    counter = 2
    while destination.exists():
        destination = PDF_DIR / f"{stem}-{counter}.pdf"
        counter += 1
    return destination


def pdf_download(url: str, filename: str | None = None, max_mb: int = 25) -> ToolResult:
    normalized_url = normalize_pdf_url(url)
    max_bytes = max(1, max_mb) * 1024 * 1024
    response = httpx.get(
        normalized_url,
        follow_redirects=True,
        timeout=30.0,
        headers={"accept": "application/pdf,*/*;q=0.5", "user-agent": "evidence-brief-agent/0.1"},
    )
    response.raise_for_status()

    content = response.content
    if len(content) > max_bytes:
        return ToolResult(
            ok=False,
            summary=f"PDF download exceeded max_mb={max_mb}.",
            data={"url": url, "resolved_url": str(response.url), "size_bytes": len(content), "max_bytes": max_bytes},
        )

    content_type = response.headers.get("content-type", "").lower()
    looks_like_pdf = b"%PDF" in content[:1024]
    if "pdf" not in content_type and not looks_like_pdf:
        return ToolResult(
            ok=False,
            summary="Downloaded URL did not look like a PDF. Use browser_fetch for HTML pages or find a direct PDF URL.",
            data={
                "url": url,
                "resolved_url": str(response.url),
                "content_type": content_type,
                "size_bytes": len(content),
            },
        )

    resolved_filename = _filename_from_response(str(response.url), response.headers.get("content-disposition"), filename)
    with PDF_WRITE_LOCK:
        destination = _unique_pdf_path(resolved_filename)
        destination.write_bytes(content)
    local_path = str(destination.relative_to(PROJECT_ROOT))
    return ToolResult(
        ok=True,
        summary=f"Downloaded PDF to {local_path}. Next call pdf_extract with this path.",
        data={
            "url": url,
            "resolved_url": str(response.url),
            "path": local_path,
            "size_bytes": len(content),
            "content_type": content_type,
        },
    )


def extract_pdf_to_index(store, run_id: str, source_path: str) -> ToolResult:
    path = safe_resolve(source_path, allowed_roots=[PDF_DIR])
    reader = PdfReader(str(path))
    page_texts: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            page_texts.append((idx, text))

    full_text = "\n\n".join(f"[page {page}]\n{text}" for page, text in page_texts)
    artifact_ref = store.write_artifact(run_id, f"pdf-{path.stem}-full-text", full_text)
    store.append_index(
        run_id,
        {
            "kind": "pdf_full_text",
            "source": str(path.name),
            "summary": f"Full extracted text for {path.name}; {len(page_texts)} pages with text.",
            "ref": artifact_ref,
            "tokens": tokenize(path.stem),
        },
    )

    chunk_count = 0
    for page, text in page_texts:
        for chunk_idx, chunk in enumerate(chunk_text(text), start=1):
            chunk_ref = store.write_artifact(run_id, f"pdf-{path.stem}-p{page}-chunk{chunk_idx}", chunk)
            store.append_index(
                run_id,
                {
                    "kind": "pdf_chunk",
                    "source": path.name,
                    "page": page,
                    "chunk": chunk_idx,
                    "summary": chunk[:280],
                    "ref": chunk_ref,
                    "tokens": tokenize(chunk),
                },
            )
            chunk_count += 1

    return ToolResult(
        ok=True,
        summary=f"Extracted {len(page_texts)} text pages and indexed {chunk_count} chunks from {path.name}.",
        data={"path": source_path, "pages": len(reader.pages), "text_pages": len(page_texts), "chunks": chunk_count},
        artifact_ref=artifact_ref,
    )


def score_records(query: str, records: list[dict], kinds: set[str] | None = None) -> list[dict]:
    q_tokens = Counter(tokenize(query))
    if not q_tokens:
        return []
    scored = []
    for record in records:
        if kinds and record.get("kind") not in kinds:
            continue
        tokens = record.get("tokens") or tokenize(record.get("summary", ""))
        counts = Counter(tokens)
        score = sum(min(q_tokens[token], counts[token]) for token in q_tokens)
        if score:
            enriched = dict(record)
            enriched["score"] = score
            scored.append(enriched)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def rag_search(store, run_id: str, query: str, max_results: int = 6) -> ToolResult:
    records = store.read_index(run_id)
    results = score_records(query, records, kinds={"pdf_chunk", "web_page"})[:max_results]
    return ToolResult(
        ok=True,
        summary=f"Found {len(results)} relevant indexed evidence chunks.",
        data={"query": query, "results": results},
    )
