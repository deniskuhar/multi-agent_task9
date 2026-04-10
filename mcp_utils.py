from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient


async def mcp_tools_to_langchain(server_name: str, mcp_url: str):
    client = MultiServerMCPClient(
        {
            server_name: {
                'transport': 'http',
                'url': mcp_url,
            }
        }
    )
    tools = await client.get_tools()
    return tools
