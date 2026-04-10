from __future__ import annotations

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
import json
import re
from pathlib import Path

from fastmcp import FastMCP

from config import get_settings

settings = get_settings()
mcp = FastMCP(
    'ReportMCP',
    instructions='Provides save_report and output-dir resource.',
)

OUTPUT_DIR = settings.output_path
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', filename.strip())
    cleaned = cleaned.strip('._')
    if not cleaned:
        cleaned = 'research_report'
    if not cleaned.lower().endswith('.md'):
        cleaned += '.md'
    return cleaned


@mcp.tool
def save_report(filename: str, content: str) -> str:
    """Save the final Markdown report to disk and return the saved file path."""
    safe_name = sanitize_filename(filename)
    path = OUTPUT_DIR / safe_name
    path.write_text(content, encoding='utf-8')
    return f'Report saved to {path}'


@mcp.resource('resource://output-dir')
def output_dir_resource() -> str:
    """Return the output directory path and the list of saved reports."""
    reports = sorted(p.name for p in OUTPUT_DIR.glob('*.md'))
    payload = {'output_dir': str(OUTPUT_DIR), 'reports': reports}
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    mcp.run(transport='http', host=settings.host, port=settings.report_mcp_port)
