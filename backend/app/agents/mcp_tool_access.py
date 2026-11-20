"""Server-scoped access to dynamically discovered LangChain MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from pydantic import JsonValue, TypeAdapter

from app.mcp_client.langchain_adapter import (
    LangChainMCPAdapter,
    _configured_environment_secrets,
)
from app.mcp_client.multi_server_client import MultiServerMCPClient
from app.mcp_client.server_config import load_server_configs

ToolArguments: TypeAlias = dict[str, JsonValue]
_JSON_VALUE = TypeAdapter(JsonValue)


@dataclass(frozen=True)
class ToolInvocation:
    """One validated tool response."""

    tool_name: str
    output: JsonValue


@dataclass(frozen=True)
class ToolInvocationError:
    """One safe tool invocation diagnostic."""

    tool_name: str
    message: str


@dataclass(frozen=True)
class ToolBatchResult:
    """Results from one server session, with safe per-tool diagnostics."""

    connected: bool
    outputs: tuple[ToolInvocation, ...] = ()
    errors: tuple[ToolInvocationError, ...] = ()
    unavailable_tools: tuple[str, ...] = ()
    connection_error: str | None = None


async def call_server_tools(
    server_name: str,
    requests: list[tuple[str, ToolArguments]],
    *,
    follow_up: Callable[
        [tuple[ToolInvocation, ...]],
        list[tuple[str, ToolArguments]],
    ]
    | None = None,
) -> ToolBatchResult:
    """Connect once, discover/validate tools, invoke requested tools, then close."""
    try:
        configs = load_server_configs()
    except Exception as error:  # noqa: BLE001 - configuration failures are branch-local.
        return ToolBatchResult(
            connected=False,
            connection_error=f"MCP configuration failed ({type(error).__name__}).",
        )
    if server_name not in configs:
        return ToolBatchResult(
            connected=False,
            connection_error="MCP server is not configured.",
        )

    try:
        client = MultiServerMCPClient({server_name: configs[server_name]})
    except Exception as error:  # noqa: BLE001 - isolate client construction failures.
        return ToolBatchResult(
            connected=False,
            connection_error=f"MCP client setup failed ({type(error).__name__}).",
        )
    outputs: list[ToolInvocation] = []
    errors: list[ToolInvocationError] = []
    unavailable: list[str] = []
    connection_error = None
    connected = False

    try:
        statuses = await client.connect_all()
        connected = statuses.get(server_name, {}).get("connected") is True
        if not connected:
            connection_error = "MCP server connection is unavailable."
        else:
            tools = await LangChainMCPAdapter(
                client,
                secrets=_configured_environment_secrets(),
            ).get_langchain_tools()
            qualified_tools = {
                tool.name.removeprefix(f"{server_name}__"): tool
                for tool in tools
                if tool.name.startswith(f"{server_name}__")
            }

            requests_to_run = requests
            while requests_to_run:
                for tool_name, arguments in requests_to_run:
                    tool = qualified_tools.get(tool_name)
                    if tool is None:
                        unavailable.append(tool_name)
                        continue
                    try:
                        raw_result = await tool.ainvoke(arguments)
                        if (
                            isinstance(raw_result, dict)
                            and "structured_content" in raw_result
                        ):
                            structured_content = raw_result["structured_content"]
                            if structured_content is not None:
                                raw_result = structured_content
                        outputs.append(
                            ToolInvocation(
                                tool_name=tool_name,
                                output=_JSON_VALUE.validate_python(raw_result),
                            )
                        )
                    except Exception as error:  # noqa: BLE001 - tool/plugin errors vary by server.
                        errors.append(
                            ToolInvocationError(
                                tool_name=tool_name,
                                message=(
                                    f"Tool invocation failed ({type(error).__name__})."
                                ),
                            )
                        )
                if follow_up is None:
                    break
                follow_up_requests = follow_up(tuple(outputs))
                follow_up = None
                requests_to_run = follow_up_requests
    except Exception as error:  # noqa: BLE001 - redact arbitrary MCP startup errors.
        connection_error = f"MCP tool discovery failed ({type(error).__name__})."
        connected = False
    finally:
        try:
            await client.close()
        except Exception as error:  # noqa: BLE001 - redact transport close errors.
            errors.append(
                ToolInvocationError(
                    tool_name="session_close",
                    message=f"MCP session close failed ({type(error).__name__}).",
                )
            )

    return ToolBatchResult(
        connected=connected,
        outputs=tuple(outputs),
        errors=tuple(errors),
        unavailable_tools=tuple(unavailable),
        connection_error=connection_error,
    )
