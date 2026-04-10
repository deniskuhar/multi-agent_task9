from __future__ import annotations

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
import json
import time
from pathlib import Path
from typing import Any

import trafilatura
from ddgs import DDGS
from fastmcp import FastMCP

from config import get_settings
from retriever import get_retriever

settings = get_settings()
mcp = FastMCP(
    'SearchMCP',
    instructions='Provides web_search, read_url, knowledge_search, and knowledge-base-stats resource.',
)

_RETRIEVER = None


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + '...'


@mcp.tool
def web_search(query: str) -> str:
    """Search the public web for relevant pages and return concise results with URLs."""
    results: list[dict[str, Any]] = []
    started = time.time()
    with DDGS(timeout=10) as ddgs:
        for item in ddgs.text(query, max_results=min(settings.max_search_results, 5)):
            if time.time() - started > 15:
                break
            results.append(
                {
                    'title': item.get('title', ''),
                    'url': item.get('href', ''),
                    'snippet': item.get('body', ''),
                }
            )
    if not results:
        return f'No web results found for query: {query}'
    return _truncate(json.dumps(results, ensure_ascii=False, indent=2), settings.max_search_content_length)


@mcp.tool
def read_url(url: str) -> str:
    """Read and extract the main textual content from a URL."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return f'Error: failed to download URL: {url}'
    extracted = trafilatura.extract(downloaded, include_links=True, include_formatting=False)
    if not extracted:
        return f'Error: failed to extract readable content from URL: {url}'
    return _truncate(extracted, settings.max_url_content_length)


@mcp.tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base using hybrid retrieval and reranking."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = get_retriever()
    docs = _RETRIEVER.hybrid_search(query)
    if not docs:
        return f'No local knowledge base results found for query: {query}'
    docs = docs[:5]
    lines = [f'Found {len(docs)} knowledge base results for query: {query}']
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page')
        page_label = f', page {page + 1}' if isinstance(page, int) else ''
        snippet = _truncate(doc.page_content.strip().replace('\n', ' '), 500)
        lines.append(f'{idx}. [{source}{page_label}] {snippet}')
    return '\n'.join(lines)


@mcp.resource('resource://knowledge-base-stats')
def knowledge_base_stats() -> str:
    """Return knowledge base statistics such as document count and last update."""
    data_dir = settings.data_path
    files = [p for p in data_dir.rglob('*') if p.is_file()]
    latest = max((p.stat().st_mtime for p in files), default=None)
    payload = {
        'data_dir': str(data_dir),
        'document_count': len(files),
        'last_updated_epoch': latest,
        'last_updated_iso': None if latest is None else __import__('datetime').datetime.fromtimestamp(latest).isoformat(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    mcp.run(transport='http', host=settings.host, port=settings.search_mcp_port)
