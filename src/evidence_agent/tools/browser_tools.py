from __future__ import annotations

import re
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from evidence_agent.tools.pdf_tools import tokenize
from evidence_agent.types import ToolResult

USER_AGENT = "EvidenceBriefAgent/0.1 (+local take-home demo)"


def browser_search(query: str, max_results: int = 5) -> ToolResult:
    try:
        with httpx.Client(timeout=12, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            response = client.get("https://duckduckgo.com/html/", params={"q": query})
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for item in soup.select(".result"):
            link = item.select_one(".result__a")
            snippet = item.select_one(".result__snippet")
            if not link:
                continue
            href = link.get("href", "")
            results.append(
                {
                    "title": link.get_text(" ", strip=True),
                    "url": href,
                    "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                }
            )
            if len(results) >= max_results:
                break
        return ToolResult(ok=True, summary=f"Search returned {len(results)} results.", data={"query": query, "results": results})
    except Exception as exc:
        return ToolResult(ok=False, summary=f"Search failed: {exc}", data={"query": query, "results": []})


def browser_fetch(store, run_id: str, url: str, max_chars: int = 9000) -> ToolResult:
    try:
        with httpx.Client(timeout=16, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            response = client.get(url)
            response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" in content_type:
            text = f"PDF URL fetched but not parsed by browser_fetch. Use pdf_extract after download: {url}"
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()
            title = soup.title.get_text(" ", strip=True) if soup.title else url
            body = soup.get_text(" ", strip=True)
            compact_body = re.sub(r"\s+", " ", body)
            text = f"# {title}\n\n{compact_body}"
        text = text[:max_chars]
        artifact_ref = store.write_artifact(run_id, "web-page", text)
        store.append_index(
            run_id,
            {
                "kind": "web_page",
                "source": url,
                "summary": text[:320],
                "ref": artifact_ref,
                "tokens": tokenize(text),
            },
        )
        return ToolResult(ok=True, summary=f"Fetched {len(text)} characters from {url}.", data={"url": url}, artifact_ref=artifact_ref)
    except Exception as exc:
        return ToolResult(ok=False, summary=f"Fetch failed: {exc}", data={"url": url})


def normalize_search_url(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return urljoin("https://duckduckgo.com", url)
    return url
