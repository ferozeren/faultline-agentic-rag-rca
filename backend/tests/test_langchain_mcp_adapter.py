"""Tests for converting dynamically discovered MCP tools to LangChain tools."""

import asyncio
from functools import wraps
from unittest.mock import patch

import app.mcp_client.langchain_adapter as adapter_module
import pytest
from app.mcp_client.langchain_adapter import (
    LangChainMCPAdapter,
    MCPToolAdapterError,
)
from app.mcp_client.multi_server_client import MCPClientError
from langchain_core.tools import StructuredTool
from mcp.types import Tool
from pydantic import ValidationError

SECRET = "adapter-test-secret"


def _async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def _tool(name, schema, description="A useful MCP tool."):
    return Tool(
        name=name,
        description=description,
        inputSchema=schema,
    )


class FakeMCPClient:
    def __init__(self, definitions, result=None, error=None):
        self.definitions = definitions
        self.result = result or {
            "server_name": "log",
            "tool_name": "search_logs",
            "content": [{"type": "text", "text": "found records"}],
            "structured_content": {"result": []},
            "is_error": False,
        }
        self.error = error
        self.discovery_calls = 0
        self.tool_calls = []

    async def discover_all_tools(self):
        self.discovery_calls += 1
        return self.definitions

    async def call_tool(self, server_name, tool_name, arguments):
        self.tool_calls.append((server_name, tool_name, arguments))
        if self.error:
            raise self.error
        return self.result


def _definitions():
    return {
        "log__search_logs": {
            "qualified_name": "log__search_logs",
            "server_name": "log",
            "tool_name": "search_logs",
            "description": "Search existing log records.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Text to find.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum result count.",
                    },
                },
                "required": ["keyword"],
            },
        },
        "metrics__get_latest_metrics": {
            "qualified_name": "metrics__get_latest_metrics",
            "server_name": "metrics",
            "tool_name": "get_latest_metrics",
            "description": "Return latest local metrics.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }


@_async_test
async def test_dynamic_discovery_calls_existing_multi_server_client():
    client = FakeMCPClient(_definitions())
    adapter = LangChainMCPAdapter(client)

    tools = await adapter.get_langchain_tools()

    assert client.discovery_calls == 1
    assert [tool.name for tool in tools] == [
        "log__search_logs",
        "metrics__get_latest_metrics",
    ]
    assert all(isinstance(tool, StructuredTool) for tool in tools)


@_async_test
async def test_tool_description_and_input_schema_are_preserved():
    adapter = LangChainMCPAdapter(FakeMCPClient(_definitions()))

    tools = await adapter.get_langchain_tools()
    search_tool = tools[0]
    schema = search_tool.args_schema.model_json_schema()

    assert search_tool.description == "Search existing log records."
    assert schema["required"] == ["keyword"]
    assert schema["properties"]["keyword"]["type"] == "string"
    assert schema["properties"]["keyword"]["description"] == "Text to find."
    assert {option["type"] for option in schema["properties"]["limit"]["anyOf"]} == {
        "integer",
        "null",
    }
    assert schema["properties"]["limit"]["default"] is None


@_async_test
async def test_empty_input_schema_creates_valid_no_argument_tool():
    adapter = LangChainMCPAdapter(FakeMCPClient(_definitions()))

    tools = await adapter.get_langchain_tools()

    result = await tools[1].ainvoke({})

    assert result["tool_name"] == "search_logs"
    assert tools[1].args_schema.model_json_schema()["properties"] == {}


@_async_test
async def test_server_qualified_names_prevent_cross_server_collisions():
    definitions = {
        "log__query": {
            "server_name": "log",
            "tool_name": "query",
            "description": "log query",
            "input_schema": {"type": "object"},
        },
        "grafana__query": {
            "server_name": "grafana",
            "tool_name": "query",
            "description": "Grafana query",
            "input_schema": {"type": "object"},
        },
    }
    adapter = LangChainMCPAdapter(FakeMCPClient(definitions))

    tools = await adapter.get_langchain_tools()

    assert [tool.name for tool in tools] == ["log__query", "grafana__query"]


@_async_test
async def test_langchain_invocation_routes_correctly_and_returns_actual_result():
    client = FakeMCPClient(_definitions())
    adapter = LangChainMCPAdapter(client)
    tools = await adapter.get_langchain_tools()

    result = await tools[0].ainvoke({"keyword": "timeout", "limit": 4})

    assert client.tool_calls == [
        ("log", "search_logs", {"keyword": "timeout", "limit": 4})
    ]
    assert result == client.result


@_async_test
async def test_tool_inputs_are_validated_before_mcp_execution():
    client = FakeMCPClient(_definitions())
    adapter = LangChainMCPAdapter(client)
    search_tool = (await adapter.get_langchain_tools())[0]

    with pytest.raises(ValidationError):
        await search_tool.ainvoke({"limit": "not-an-integer"})
    with pytest.raises(ValidationError):
        await search_tool.ainvoke({"keyword": "x", "unexpected": True})
    with pytest.raises(ValidationError) as error:
        await search_tool.ainvoke({"keyword": "x", "limit": SECRET})

    assert client.tool_calls == []
    assert SECRET not in str(error.value)


@_async_test
async def test_disconnected_server_fails_without_fabricated_result():
    client = FakeMCPClient(
        {
            "prometheus__query_prometheus": {
                "server_name": "prometheus",
                "tool_name": "query_prometheus",
                "description": "Query Prometheus.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        },
        error=MCPClientError("MCP server 'prometheus' is not connected."),
    )
    tool = (await LangChainMCPAdapter(client).get_langchain_tools())[0]

    with pytest.raises(MCPToolAdapterError, match="MCPClientError"):
        await tool.ainvoke({"query": "up"})


@_async_test
async def test_mcp_execution_error_is_reported_safely():
    client = FakeMCPClient(
        _definitions(),
        error=RuntimeError(f"untrusted detail {SECRET}"),
    )
    tool = (await LangChainMCPAdapter(client).get_langchain_tools())[0]

    with pytest.raises(MCPToolAdapterError) as error:
        await tool.ainvoke({"keyword": "x"})

    assert SECRET not in str(error.value)
    assert "RuntimeError" in str(error.value)


@_async_test
async def test_mcp_error_result_is_not_returned_as_success():
    client = FakeMCPClient(
        _definitions(),
        result={"is_error": True, "content": [{"text": "tool failed"}]},
    )
    tool = (await LangChainMCPAdapter(client).get_langchain_tools())[0]

    with pytest.raises(MCPToolAdapterError, match="reported an execution error"):
        await tool.ainvoke({"keyword": "x"})


@_async_test
async def test_async_invocation_uses_client_coroutine():
    client = FakeMCPClient(_definitions())
    tool = (await LangChainMCPAdapter(client).get_langchain_tools())[0]

    result = await tool.ainvoke({"keyword": "async"})

    assert result["tool_name"] == "search_logs"
    assert client.tool_calls[0][2]["keyword"] == "async"


@_async_test
async def test_secret_values_are_redacted_from_descriptions_and_results(monkeypatch):
    monkeypatch.setenv("GRAFANA_SERVICE_ACCOUNT_TOKEN", SECRET)
    definitions = _definitions()
    definitions["log__search_logs"]["description"] = f"Contains {SECRET}."
    client = FakeMCPClient(
        definitions,
        result={
            "server_name": "log",
            "tool_name": "search_logs",
            "content": [{"type": "text", "text": f"output {SECRET}"}],
            "structured_content": {"token_echo": SECRET},
            "is_error": False,
        },
    )
    adapter = LangChainMCPAdapter(client)

    tools = await adapter.get_langchain_tools()
    result = await tools[0].ainvoke({"keyword": "x"})

    assert SECRET not in tools[0].description
    assert SECRET not in repr(result)


@_async_test
async def test_explicit_secrets_bypass_dotenv_lookup(monkeypatch):
    def reject_dotenv_lookup():
        pytest.fail("explicit secrets should not inspect the .env file")

    monkeypatch.setattr(adapter_module, "_configured_secrets", reject_dotenv_lookup)
    definitions = _definitions()
    definitions["log__search_logs"]["description"] = f"Contains {SECRET}."
    client = FakeMCPClient(
        definitions,
        result={
            "server_name": "log",
            "tool_name": "search_logs",
            "content": [{"type": "text", "text": f"output {SECRET}"}],
            "structured_content": {"token_echo": SECRET},
            "is_error": False,
        },
    )
    adapter = LangChainMCPAdapter(client, secrets=(SECRET,))

    tools = await adapter.get_langchain_tools()
    result = await tools[0].ainvoke({"keyword": "x"})

    assert SECRET not in tools[0].description
    assert SECRET not in repr(result)


@_async_test
async def test_discovery_errors_are_sanitized():
    client = FakeMCPClient(_definitions())

    async def fail_discovery():
        raise RuntimeError(f"private {SECRET}")

    client.discover_all_tools = fail_discovery
    with pytest.raises(MCPToolAdapterError) as error:
        await LangChainMCPAdapter(client).get_langchain_tools()

    assert SECRET not in str(error.value)
    assert "RuntimeError" in str(error.value)
