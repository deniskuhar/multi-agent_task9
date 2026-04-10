from __future__ import annotations

from typing import Literal
import inspect
import uvicorn.config

if not hasattr(uvicorn.config, "LoopSetupType"):
    uvicorn.config.LoopSetupType = Literal["none", "auto", "asyncio", "uvloop"]

# Force-disable SSL args that ACP/uvicorn may accidentally populate.
_original_uvicorn_config_init = uvicorn.config.Config.__init__

def _patched_uvicorn_config_init(self, *args, **kwargs):
    sig = inspect.signature(_original_uvicorn_config_init)
    bound = sig.bind_partial(self, *args, **kwargs)
    bound.arguments["ssl_keyfile"] = None
    bound.arguments["ssl_certfile"] = None
    bound.arguments["ssl_ca_certs"] = None
    bound.arguments["ssl_keyfile_password"] = None
    return _original_uvicorn_config_init(*bound.args, **bound.kwargs)

uvicorn.config.Config.__init__ = _patched_uvicorn_config_init
    
import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from langchain.messages import AIMessage

from agents import build_critic_agent, build_planner_agent, build_researcher_agent
from schemas import CritiqueResult, ResearchPlan
from config import get_settings

server = Server()
settings = get_settings()


def _extract_text_from_state(state: Any) -> str:
    if isinstance(state, dict):
        messages = state.get('messages', [])
        if messages:
            last = messages[-1]
            content = getattr(last, 'content', None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get('text') or item.get('content')
                        if text:
                            parts.append(str(text))
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return '\n'.join(parts)
        structured = state.get('structured_response')
        if structured is not None:
            return str(structured)
    if isinstance(state, AIMessage):
        return str(state.content)
    return str(state)


def _prompt_from_messages(input: list[Message]) -> str:
    chunks: list[str] = []
    for message in input:
        role = getattr(message, 'role', 'user')
        parts = getattr(message, 'parts', []) or []
        text_parts: list[str] = []
        for part in parts:
            content = getattr(part, 'content', None)
            if content:
                text_parts.append(str(content))
        joined = '\n'.join(text_parts).strip()
        if joined:
            chunks.append(f'{role}: {joined}')
    return '\n\n'.join(chunks).strip()


def _yield_text(text: str) -> Message:
    return Message(role='agent', parts=[MessagePart(content=text)])


@server.agent(name='planner')
async def planner(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Planner agent that returns a structured ResearchPlan."""
    prompt = _prompt_from_messages(input)
    agent = await build_planner_agent()
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': prompt}]})
    plan_obj: ResearchPlan = result['structured_response']
    yield _yield_text(json.dumps(plan_obj.model_dump(), ensure_ascii=False, indent=2))


@server.agent(name='researcher')
async def researcher(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Researcher agent that returns an evidence-rich findings memo."""
    prompt = _prompt_from_messages(input)
    agent = await build_researcher_agent()
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': prompt}]})
    yield _yield_text(_extract_text_from_state(result))


@server.agent(name='critic')
async def critic(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Critic agent that returns a structured CritiqueResult."""
    prompt = _prompt_from_messages(input)
    agent = await build_critic_agent()
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': prompt}]})
    critique_obj: CritiqueResult = result['structured_response']
    yield _yield_text(json.dumps(critique_obj.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    server.run()
