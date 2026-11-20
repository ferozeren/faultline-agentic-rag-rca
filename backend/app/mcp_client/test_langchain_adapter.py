"""Demonstrate dynamic MCP-to-LangChain tool adaptation without an LLM."""

from __future__ import annotations

import asyncio
import json

from app.mcp_client.langchain_adapter import LangChainMCPAdapter
from app.mcp_client.multi_server_client import MultiServerMCPClient


async def main() -> None:
    client = MultiServerMCPClient()
    try:
        statuses = await client.connect_all()
        for server_name, status in statuses.items():
            print(f"{server_name}: {status}")

        adapter = LangChainMCPAdapter(client)
        tools = await adapter.get_langchain_tools()
        for tool in tools:
            print(f"{tool.name}: {tool.args_schema.model_json_schema()}")

        sample_tool = next(
            (tool for tool in tools if tool.name == "log__get_recent_logs"),
            None,
        )
        if sample_tool is None:
            print("Read-only Log MCP tool unavailable; skipping sample invocation.")
            return
        result = await sample_tool.ainvoke({"limit": 3})
        print("Read-only LangChain tool result:")
        print(json.dumps(result, indent=2, default=str))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
