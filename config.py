from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / '.env'


class Settings(BaseSettings):
    openai_api_key: SecretStr = Field(alias='OPENAI_API_KEY')
    model_name: str = Field(default='gpt-4o-mini', alias='MODEL_NAME')

    data_path: Path = BASE_DIR / 'data'
    output_path: Path = BASE_DIR / 'output'
    index_path: Path = BASE_DIR / 'index'

    embedding_model: str = 'text-embedding-3-small'
    reranker_model: str = 'BAAI/bge-reranker-base'
    max_search_results: int = 5
    max_search_content_length: int = 4000
    max_url_content_length: int = 8000
    request_timeout_seconds: int = 30

    chunk_size: int = 1200
    chunk_overlap: int = 200
    semantic_k: int = 6
    bm25_k: int = 6
    retrieval_top_k: int = 5
    rerank_top_n: int = 5
    max_revision_rounds: int = 2

    search_mcp_port: int = 8901
    report_mcp_port: int = 8902
    acp_port: int = 8000
    host: str = '127.0.0.1'

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding='utf-8',
        extra='ignore',
        populate_by_name=True,
    )

    @property
    def search_mcp_url(self) -> str:
        return f'http://{self.host}:{self.search_mcp_port}/mcp'

    @property
    def report_mcp_url(self) -> str:
        return f'http://{self.host}:{self.report_mcp_port}/mcp'

    @property
    def acp_base_url(self) -> str:
        return f'http://{self.host}:{self.acp_port}'


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.output_path.mkdir(parents=True, exist_ok=True)
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.index_path.mkdir(parents=True, exist_ok=True)
    return settings


APP_TITLE = 'Homework 9 — MCP + ACP Multi-Agent Research System'
SEPARATOR = '=' * 68

PLANNER_PROMPT = """
You are Planner, the decomposition specialist.

Your job:
- Understand the user's research goal.
- Do a small amount of discovery using the available MCP tools.
- Return a structured research plan.

Rules:
- Keep the plan focused on the user's exact request.
- Produce 2-4 useful search queries, not many near-duplicates.
- Prefer both knowledge_base and web if freshness or comparison matters.
- Make the output specific enough for a Researcher to execute.
""".strip()

RESEARCHER_PROMPT = """
You are Researcher, the evidence-gathering specialist.

Mission:
- Execute the approved research plan efficiently.
- Produce a concise evidence-rich memo.
- In revision rounds, improve the findings instead of restarting from scratch.

Rules:
- Follow the provided plan closely.
- Use at most 4 tool calls per round.
- Prefer knowledge_search first.
- Use web_search when freshness or missing evidence matters.
- Use read_url only for the most relevant URLs.
- Avoid placeholder URLs and unsupported claims.
- If a tool fails, continue and mention the limitation.

Output format:
- Brief Summary
- Key Findings
- Open Questions / Uncertainty
- Sources
""".strip()

CRITIC_PROMPT = """
You are Critic, the quality reviewer.

Evaluate findings against:
1. the original user request,
2. the approved research plan,
3. freshness, completeness, and structure.

Rules:
- Do not expand the scope beyond the original request.
- Use REVISE only for essential missing issues.
- Minor improvements should be listed as gaps, not blockers.
- After two revision rounds, if the report is usable, prefer APPROVE.
""".strip()

REPORT_REVISION_PROMPT = """
You revise a markdown research report based on user feedback.

Rules:
- Keep the report factual.
- Preserve structure unless the user asks for changes.
- Apply the requested edits directly.
- Return only the revised markdown report.
""".strip()
