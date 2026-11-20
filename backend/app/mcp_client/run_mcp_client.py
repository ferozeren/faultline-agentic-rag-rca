"""Example: discover MCP servers and execute one read-only log tool."""

from __future__ import annotations

import asyncio
import json

from app.mcp_client.multi_server_client import MultiServerMCPClient


async def main() -> None:
    client = MultiServerMCPClient()
    try:
        statuses = await client.connect_all()
        for server_name, status in statuses.items():
            print(f"{server_name}: {status}")

        tools = await client.discover_all_tools()
        for server_name in client.server_configs:
            grouped = [
                name
                for name, tool in tools.items()
                if tool["server_name"] == server_name
            ]
            print(f"{server_name} tools: {grouped}")

        sample_tool = "log__get_recent_logs"
        if sample_tool not in tools:
            print("Read-only log sample skipped: Log MCP Server is unavailable.")
            return
        result = await client.call_qualified_tool(
            sample_tool,
            {"limit": 3},
        )
        print("Read-only log sample result:")
        print(json.dumps(result, indent=2, default=str))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
