"""Manage sessions and dynamically discovered tools from multiple MCP servers."""

from __future__ import annotations

import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, Tool

from app.mcp_client.server_config import MCPServerConfig, load_server_configs


class MCPClientError(RuntimeError):
    """A safe, user-facing client configuration or execution error."""


@dataclass
class ServerConnection:
    """A connected MCP session and its discovered metadata."""

    config: MCPServerConfig
    session: ClientSession
    tools: dict[str, Tool] = field(default_factory=dict)
    resources: list[Any] = field(default_factory=list)


class MultiServerMCPClient:
    """Connect to configured stdio servers and unify their tool registries."""

    def __init__(
        self,
        server_configs: dict[str, MCPServerConfig] | None = None,
    ) -> None:
        self.server_configs = (
            load_server_configs() if server_configs is None else server_configs
        )
        self.connections: dict[str, ServerConnection] = {}
        self.connection_status: dict[str, dict[str, Any]] = {
            name: {
                "connected": False,
                "error": None,
                "tools_count": 0,
                "resources_count": 0,
                "resources_error": None,
            }
            for name in self.server_configs
        }
        self.tool_registry: dict[str, dict[str, Any]] = {}
        self._exit_stack: AsyncExitStack | None = None
        self._attempted: set[str] = set()

    async def connect_all(self) -> dict[str, dict[str, Any]]:
        """Connect to each configured server, retaining successful sessions."""
        if self._exit_stack is None:
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

        for name, config in self.server_configs.items():
            if name in self._attempted:
                continue
            self._attempted.add(name)
            try:
                parameters = StdioServerParameters(
                    command=config.command,
                    args=list(config.args),
                    cwd=config.cwd,
                )
                errlog = open(os.devnull, "w", encoding="utf-8")
                self._exit_stack.callback(errlog.close)
                streams = await self._exit_stack.enter_async_context(
                    stdio_client(parameters, errlog=errlog)
                )
                read_stream, write_stream = streams
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                tools = await self._list_all_tools(session)
                resources_error = None
                try:
                    resources = await self._list_all_resources(session)
                except Exception as error:
                    resources = []
                    resources_error = (
                        f"resource discovery failed ({type(error).__name__})."
                    )
                connection = ServerConnection(
                    config=config,
                    session=session,
                    tools={tool.name: tool for tool in tools},
                    resources=resources,
                )
                self.connections[name] = connection
                self.connection_status[name] = {
                    "connected": True,
                    "error": None,
                    "tools_count": len(connection.tools),
                    "resources_count": len(resources),
                    "resources_error": resources_error,
                }
            except Exception as error:
                self.connection_status[name] = {
                    "connected": False,
                    "error": self._safe_connection_error(error),
                    "tools_count": 0,
                    "resources_count": 0,
                    "resources_error": None,
                }
        self._refresh_tool_registry()
        return {name: dict(status) for name, status in self.connection_status.items()}

    async def _list_all_tools(self, session: ClientSession) -> list[Tool]:
        tools: list[Tool] = []
        cursor = None
        while True:
            result = await session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if not cursor:
                return tools

    async def _list_all_resources(self, session: ClientSession) -> list[Any]:
        resources = []
        cursor = None
        while True:
            result = await session.list_resources(cursor=cursor)
            resources.extend(result.resources)
            cursor = result.nextCursor
            if not cursor:
                return resources

    def _refresh_tool_registry(self) -> None:
        registry: dict[str, dict[str, Any]] = {}
        for server_name, connection in self.connections.items():
            for tool_name, tool in connection.tools.items():
                qualified_name = f"{server_name}__{tool_name}"
                registry[qualified_name] = {
                    "qualified_name": qualified_name,
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
        self.tool_registry = registry

    async def discover_all_tools(self) -> dict[str, dict[str, Any]]:
        """Connect to servers if needed and return a unified dynamic registry."""
        if self._exit_stack is None or len(self.connections) < len(self.server_configs):
            await self.connect_all()
        self._refresh_tool_registry()
        return {name: dict(tool) for name, tool in self.tool_registry.items()}

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke a discovered tool on its originating server session."""
        connection = self.connections.get(server_name)
        if connection is None:
            if server_name not in self.server_configs:
                raise MCPClientError(f"Unknown MCP server '{server_name}'.")
            status = self.connection_status.get(server_name, {})
            message = status.get("error") or "server is not connected."
            raise MCPClientError(f"MCP server '{server_name}' {message}")
        if tool_name not in connection.tools:
            raise MCPClientError(
                f"Unknown tool '{tool_name}' on MCP server '{server_name}'."
            )
        if arguments is not None and not isinstance(arguments, dict):
            raise MCPClientError("Tool arguments must be a mapping.")

        try:
            result = await connection.session.call_tool(
                tool_name,
                arguments=arguments or {},
            )
        except Exception as error:
            raise MCPClientError(
                f"Tool '{tool_name}' execution failed on server '{server_name}' "
                f"({type(error).__name__})."
            ) from None
        return self._format_tool_result(server_name, tool_name, result)

    async def call_qualified_tool(
        self,
        qualified_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve and invoke a server-qualified tool name."""
        tool = self.tool_registry.get(qualified_name)
        if tool is None:
            raise MCPClientError(f"Unknown server-qualified tool '{qualified_name}'.")
        return await self.call_tool(
            tool["server_name"],
            tool["tool_name"],
            arguments,
        )

    @staticmethod
    def _format_tool_result(
        server_name: str,
        tool_name: str,
        result: CallToolResult,
    ) -> dict[str, Any]:
        if result.isError:
            raise MCPClientError(
                f"Tool '{tool_name}' reported an execution error on "
                f"server '{server_name}'."
            )
        content = [
            item.model_dump(mode="json", exclude_none=True)
            for item in result.content
        ]
        return {
            "server_name": server_name,
            "tool_name": tool_name,
            "content": content,
            "structured_content": result.structuredContent,
            "is_error": False,
        }

    @staticmethod
    def _safe_connection_error(error: Exception) -> str:
        if isinstance(error, MCPClientError):
            return str(error)
        return f"connection failed ({type(error).__name__})."

    async def close(self) -> None:
        """Close all sessions and subprocess transports gracefully."""
        stack = self._exit_stack
        self._exit_stack = None
        if stack is None:
            return
        self.connections.clear()
        self.tool_registry.clear()
        self._attempted.clear()
        for status in self.connection_status.values():
            status["connected"] = False
        try:
            await stack.aclose()
        except Exception as error:
            raise MCPClientError(
                f"One or more MCP connections failed to close "
                f"({type(error).__name__})."
            ) from None

    async def __aenter__(self) -> MultiServerMCPClient:
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()
