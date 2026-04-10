from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import RESEARCHER_PROMPT, get_settings
from mcp_utils import mcp_tools_to_langchain


async def build_researcher_agent():
    settings = get_settings()
    model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
    )
    tools = await mcp_tools_to_langchain('search', settings.search_mcp_url)
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=RESEARCHER_PROMPT,
    )
