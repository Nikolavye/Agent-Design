from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from evidence_agent.tools.browser_tools import browser_fetch, browser_search
from evidence_agent.tools.fs_tools import fs_list, fs_read
from evidence_agent.tools.memory_tools import memory_get, memory_search
from evidence_agent.tools.pdf_tools import extract_pdf_to_index, pdf_download, rag_search


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    callable: Callable


def tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec("fs_list", "List allowed workspace files.", fs_list),
        ToolSpec("fs_read", "Read a bounded text file slice.", fs_read),
        ToolSpec("pdf_download", "Download a remote PDF into data/pdfs for later extraction.", pdf_download),
        ToolSpec("pdf_extract", "Extract and index PDF pages into chunks.", extract_pdf_to_index),
        ToolSpec("rag_search", "Lexical RAG search over indexed PDF and web chunks.", rag_search),
        ToolSpec("browser_search", "Search the web through a lightweight browser query.", browser_search),
        ToolSpec("browser_fetch", "Fetch a web page and index its text.", browser_fetch),
        ToolSpec("memory_search", "Search event and artifact summaries for precise refs.", memory_search),
        ToolSpec("memory_get", "Load a small artifact slice by ref.", memory_get),
    ]


def tool_manifest() -> list[dict[str, str]]:
    return [{"name": spec.name, "description": spec.description} for spec in tool_specs()]
