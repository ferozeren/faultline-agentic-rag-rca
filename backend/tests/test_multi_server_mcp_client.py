"""Tests for multi-server MCP lifecycle, discovery, and tool execution."""

import asyncio
import sys
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import pytest
from mcp.types import (
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    Resource,
    TextContent,
    Tool,
)

from app.mcp_client import multi_server_client as client_module
from app.mcp_client.multi_server_client import MCPClientError, MultiServerMCPClient
from app.mcp_client.server_config import BACKEND_ROOT, MCPServerConfig, load_server_configs

SECRET = "very-secret-value"
SERVER_NAMES = ("log", "metrics", "prometheus", "grafana")


def _async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def _configs(*names: str) -> dict[str, MCPServerConfig]:
    return {
        name: MCPServerConfig(
            name=name,
            module=f"app.mcp_servers.{name}_mcp_server",
            command=sys.executable,
            cwd=BACKEND_ROOT,
            args=("-m", f"app.mcp_servers.{name}_mcp_server"),
        )
        for name in names
    }


def test_server_configuration_uses_expected_modules_and_backend_environment():
    configs = load_server_configs()

    assert tuple(configs) == SERVER_NAMES
    assert configs["log"].module == "app.mcp_servers.log_mcp_server"
    assert configs["metrics"].module == "app.mcp_servers.metrics_mcp_server"
    assert configs["prometheus"].module == "app.mcp_servers.prometheus_mcp_server"
    assert configs["grafana"].module == "app.mcp_servers.grafana_mcp_server"
    assert all(config.command == sys.executable for config in configs.values())
    assert all(config.cwd == Path(BACKEND_ROOT) for config in configs.values())
    assert all(config.args[0] == "-m" for config in configs.values())


class FakeSession:
    def __init__(self, read_stream, write_stream, state):
        self.server_name = read_stream
        self.state = state
        self.closed = False

    async def __aenter__(self):
        self.state["sessions"].append(self)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.closed = True

    async def initialize(self):
        self.state["initialized"].append(self.server_name)

    async def list_tools(self, cursor=None):
        names = self.state["tool_names"][self.server_name]
        if cursor is None and len(names) > 1:
            page_names = names[:1]
            next_cursor = "next"
        else:
            page_names = names[1:] if cursor == "next" else names
            next_cursor = None
        return ListToolsResult(
            tools=[
                Tool(
                    name=name,
                    description=f"{self.server_name} tool {name}",
                    inputSchema={"type": "object"},
                )
                for name in page_names
            ],
            nextCursor=next_cursor,
        )

    async def list_resources(self, cursor=None):
        return ListResourcesResult(
            resources=[
                Resource(
                    uri=f"{self.server_name}://available",
                    name=f"{self.server_name} available",
                )
            ],
        )

    async def call_tool(self, name, arguments=None, **kwargs):
        self.state["calls"].append((self.server_name, name, arguments))
        if self.state.get("execution_error"):
            raise RuntimeError(f"unexpected detail: {SECRET}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"{self.server_name}:{name}")],
            structuredContent={"arguments": arguments or {}},
            isError=self.state.get("is_error", False),
        )


def _mock_client(monkeypatch, tool_names=None, unavailable=()):
    state = {
        "tool_names": tool_names
        or {name: [f"tool_{name}"] for name in SERVER_NAMES},
        "unavailable": set(unavailable),
        "started": [],
        "transport_closed": [],
        "sessions": [],
        "initialized": [],
        "calls": [],
        "execution_error": False,
        "is_error": False,
    }

    @asynccontextmanager
    async def fake_stdio_client(parameters, errlog):
        server_name = parameters.args[-1].rsplit(".", 1)[-1].removesuffix(
            "_mcp_server"
        )
        state["started"].append(server_name)
        if server_name in state["unavailable"]:
            raise OSError(f"connection failed with credential {SECRET}")
        try:
            yield server_name, server_name
        finally:
            state["transport_closed"].append(server_name)

    def fake_session(read_stream, write_stream):
        return FakeSession(read_stream, write_stream, state)

    monkeypatch.setattr(client_module, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(client_module, "ClientSession", fake_session)
    return state


@_async_test
async def test_connects_to_multiple_servers_and_reuses_active_sessions(monkeypatch):
    state = _mock_client(monkeypatch)
    client = MultiServerMCPClient(_configs("log", "metrics"))

    first = await client.connect_all()
    second = await client.connect_all()

    assert first["log"]["connected"] is True
    assert first["metrics"]["connected"] is True
    assert len(state["started"]) == 2
    assert len(state["sessions"]) == 2
    assert second == first
    await client.close()


@_async_test
async def test_one_unavailable_server_does_not_prevent_other_connections(monkeypatch):
    state = _mock_client(
        monkeypatch,
        unavailable=("metrics",),
    )
    client = MultiServerMCPClient(_configs("log", "metrics", "grafana"))

    statuses = await client.connect_all()

    assert statuses["log"]["connected"] is True
    assert statuses["metrics"]["connected"] is False
    assert "OSError" in statuses["metrics"]["error"]
    assert SECRET not in repr(statuses)
    assert statuses["grafana"]["connected"] is True
    assert set(client.connections) == {"log", "grafana"}
    await client.close()


@_async_test
async def test_dynamic_tool_discovery_paginates_and_qualifies_names(monkeypatch):
    state = _mock_client(
        monkeypatch,
        tool_names={
            "log": ["get_recent_logs", "shared"],
            "metrics": ["get_latest_metrics", "shared"],
        },
    )
    client = MultiServerMCPClient(_configs("log", "metrics"))

    registry = await client.discover_all_tools()

    assert set(registry) == {
        "log__get_recent_logs",
        "log__shared",
        "metrics__get_latest_metrics",
        "metrics__shared",
    }
    assert registry["log__shared"]["server_name"] == "log"
    assert registry["metrics__shared"]["server_name"] == "metrics"
    assert client.connection_status["log"]["resources_count"] == 1
    assert all(name in state["initialized"] for name in ("log", "metrics"))
    await client.close()


@_async_test
async def test_tool_call_uses_correct_server_and_returns_consistent_result(monkeypatch):
    state = _mock_client(monkeypatch)
    client = MultiServerMCPClient(_configs("log", "metrics"))
    await client.connect_all()

    result = await client.call_tool("metrics", "tool_metrics", {"limit": 4})
    qualified_result = await client.call_qualified_tool(
        "log__tool_log",
        {"limit": 2},
    )

    assert result == {
        "server_name": "metrics",
        "tool_name": "tool_metrics",
        "content": [{"type": "text", "text": "metrics:tool_metrics"}],
        "structured_content": {"arguments": {"limit": 4}},
        "is_error": False,
    }
    assert qualified_result["server_name"] == "log"
    assert state["calls"] == [
        ("metrics", "tool_metrics", {"limit": 4}),
        ("log", "tool_log", {"limit": 2}),
    ]
    await client.close()


@_async_test
async def test_unknown_server_and_tool_are_clear_errors(monkeypatch):
    _mock_client(monkeypatch)
    client = MultiServerMCPClient(_configs("log"))
    await client.connect_all()

    with pytest.raises(MCPClientError, match="Unknown MCP server"):
        await client.call_tool("unknown", "tool")
    with pytest.raises(MCPClientError, match="Unknown tool"):
        await client.call_tool("log", "missing_tool")
    with pytest.raises(MCPClientError, match="server-qualified"):
        await client.call_qualified_tool("missing__tool")

    await client.close()


@_async_test
async def test_tool_execution_errors_do_not_expose_secret(monkeypatch):
    state = _mock_client(monkeypatch)
    state["execution_error"] = True
    client = MultiServerMCPClient(_configs("log"))
    await client.connect_all()

    with pytest.raises(MCPClientError) as error:
        await client.call_tool("log", "tool_log")

    assert SECRET not in str(error.value)
    await client.close()


@_async_test
async def test_mcp_tool_error_result_is_not_returned_as_success(monkeypatch):
    state = _mock_client(monkeypatch)
    state["is_error"] = True
    client = MultiServerMCPClient(_configs("log"))
    await client.connect_all()

    with pytest.raises(MCPClientError, match="reported an execution error"):
        await client.call_tool("log", "tool_log")
    await client.close()


@_async_test
async def test_close_cleans_all_sessions_and_transports(monkeypatch):
    state = _mock_client(monkeypatch)
    client = MultiServerMCPClient(_configs("log", "metrics"))
    await client.connect_all()
    await client.close()

    assert all(session.closed for session in state["sessions"])
    assert set(state["transport_closed"]) == {"log", "metrics"}
    assert client.connections == {}
    assert client.tool_registry == {}
    assert all(not status["connected"] for status in client.connection_status.values())


@_async_test
async def test_resource_discovery_failure_does_not_drop_tools(monkeypatch):
    state = _mock_client(monkeypatch)

    async def failing_resources(self, cursor=None):
        raise RuntimeError(f"resource error {SECRET}")

    monkeypatch.setattr(FakeSession, "list_resources", failing_resources)
    client = MultiServerMCPClient(_configs("log"))
    statuses = await client.connect_all()

    assert statuses["log"]["connected"] is True
    assert statuses["log"]["resources_error"] == "resource discovery failed (RuntimeError)."
    assert "log__tool_log" in await client.discover_all_tools()
    assert SECRET not in repr(statuses)
    await client.close()


def test_server_configs_do_not_contain_secret_values():
    configs = load_server_configs()
    assert SECRET not in repr(configs)
    assert all(config.cwd == BACKEND_ROOT for config in configs.values())
