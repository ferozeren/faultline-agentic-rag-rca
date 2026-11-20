"""Convert discovered MCP tools into validated LangChain structured tools."""

from __future__ import annotations

import os
import re
from io import StringIO
from typing import Any, Literal, Union

from dotenv import dotenv_values
from langchain_core.tools import StructuredTool
from pydantic import ConfigDict, Field, create_model

from app.mcp_client.multi_server_client import MCPClientError, MultiServerMCPClient
from app.mcp_client.server_config import BACKEND_ROOT

SECRET_ENV_NAMES = {
    "GRAFANA_SERVICE_ACCOUNT_TOKEN",
    "PROMETHEUS_BEARER_TOKEN",
    "NVIDIA_API_KEY",
}


class MCPToolAdapterError(RuntimeError):
    """A safe, user-facing MCP-to-LangChain integration error."""


def _configured_environment_secrets() -> tuple[str, ...]:
    secrets = {
        value
        for name, value in os.environ.items()
        if value
        and (
            name in SECRET_ENV_NAMES
            or any(part in name.casefold() for part in ("token", "secret", "password"))
        )
    }

    return tuple(sorted(secrets, key=len, reverse=True))


def _configured_secrets() -> tuple[str, ...]:
    secrets = set(_configured_environment_secrets())
    env_path = BACKEND_ROOT / ".env"
    if env_path.exists():
        selected_lines = []
        with env_path.open(encoding="utf-8") as env_file:
            for line in env_file:
                match = re.match(
                    r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=",
                    line,
                )
                if match and (
                    match.group(1) in SECRET_ENV_NAMES
                    or any(
                        part in match.group(1).casefold()
                        for part in ("token", "secret", "password")
                    )
                ):
                    selected_lines.append(line)
        values = dotenv_values(stream=StringIO("".join(selected_lines)))
        secrets.update(value for value in values.values() if value)
    return tuple(sorted(secrets, key=len, reverse=True))


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, secrets) for item in value)
    if isinstance(value, dict):
        return {
            _redact(key, secrets): _redact(item, secrets)
            for key, item in value.items()
        }
    return value


def _schema_type(
    schema: dict[str, Any],
    model_name: str,
    definitions: dict[str, Any],
) -> Any:
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        definition_name = reference.rsplit("/", 1)[-1]
        definition = definitions.get(definition_name)
        if isinstance(definition, dict):
            return _schema_type(definition, f"{model_name}{definition_name}", definitions)

    for union_key in ("anyOf", "oneOf"):
        options = schema.get(union_key)
        if isinstance(options, list) and options:
            return Union[tuple(_schema_type(option, model_name, definitions) for option in options)]

    values = schema.get("enum")
    if isinstance(values, list) and values:
        return Literal[tuple(values)]

    schema_type = schema.get("type")
    nullable = schema.get("nullable", False)
    if isinstance(schema_type, list):
        nullable = nullable or "null" in schema_type
        non_null_types = [value for value in schema_type if value != "null"]
        if len(non_null_types) == 1:
            schema_type = non_null_types[0]
        elif non_null_types:
            return Union[tuple(_schema_type({"type": value}, model_name, definitions) for value in non_null_types)]
        else:
            return Any

    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return dict[str, Any]
        if not properties:
            return dict[str, Any]
        required = set(schema.get("required", []))
        fields = {}
        for property_name, property_schema in properties.items():
            if not isinstance(property_name, str) or not property_name.isidentifier():
                raise MCPToolAdapterError(
                    "MCP input schema contains an unsupported argument name."
                )
            if not isinstance(property_schema, dict):
                property_schema = {}
            annotation = _schema_type(
                property_schema,
                f"{model_name}{property_name.title()}",
                definitions,
            )
            if property_schema.get("nullable") or (
                isinstance(property_schema.get("type"), list)
                and "null" in property_schema["type"]
            ):
                annotation = Union[annotation, type(None)]
            default = property_schema.get(
                "default",
                ... if property_name in required else None,
            )
            if default is None and not _allows_none(annotation):
                annotation = Union[annotation, type(None)]
            field_kwargs = {}
            description = property_schema.get("description")
            if isinstance(description, str):
                field_kwargs["description"] = description
            fields[property_name] = (annotation, Field(default, **field_kwargs))
        return create_model(
            model_name,
            __config__=ConfigDict(extra="forbid", hide_input_in_errors=True),
            **fields,
        )

    if schema_type == "array":
        item_schema = schema.get("items", {})
        item_type = (
            _schema_type(item_schema, f"{model_name}Item", definitions)
            if isinstance(item_schema, dict)
            else Any
        )
        return list[item_type]

    primitive_types = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": type(None),
    }
    annotation = primitive_types.get(schema_type, Any)
    if nullable and annotation is not Any:
        return Union[annotation, type(None)]
    return annotation


def _input_model(
    qualified_name: str,
    schema: dict[str, Any],
) -> type:
    if schema.get("type", "object") != "object":
        raise MCPToolAdapterError(
            f"MCP tool '{qualified_name}' must have an object input schema."
        )
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise MCPToolAdapterError(
            f"MCP tool '{qualified_name}' has an invalid input schema."
        )
    required = set(schema.get("required", []))
    fields = {}
    definitions = schema.get("$defs", {})
    if not isinstance(definitions, dict):
        definitions = {}
    for property_name, property_schema in properties.items():
        if not isinstance(property_name, str) or not property_name.isidentifier():
            raise MCPToolAdapterError(
                "MCP input schema contains an unsupported argument name."
            )
        if not isinstance(property_schema, dict):
            raise MCPToolAdapterError(
                f"MCP tool '{qualified_name}' has an invalid input schema."
            )
        annotation = _schema_type(
            property_schema,
            f"{qualified_name.replace('__', '_').title()}{property_name.title()}",
            definitions,
        )
        schema_type = property_schema.get("type")
        nullable = property_schema.get("nullable", False) or (
            isinstance(schema_type, list) and "null" in schema_type
        )
        if nullable and not _allows_none(annotation):
            annotation = Union[annotation, type(None)]

        default = property_schema.get(
            "default",
            ... if property_name in required else None,
        )
        if default is None and not _allows_none(annotation):
            annotation = Union[annotation, type(None)]
        field_kwargs = {}
        description = property_schema.get("description")
        if isinstance(description, str):
            field_kwargs["description"] = description
        fields[property_name] = (annotation, Field(default, **field_kwargs))

    return create_model(
        f"{qualified_name.replace('__', '_').title()}Input",
        __config__=ConfigDict(extra="forbid", hide_input_in_errors=True),
        **fields,
    )


def _allows_none(annotation: Any) -> bool:
    return type(None) in getattr(annotation, "__args__", ())


class LangChainMCPAdapter:
    """Adapt a connected multi-server MCP client to LangChain tools."""

    def __init__(
        self,
        client: MultiServerMCPClient,
        *,
        secrets: tuple[str, ...] | None = None,
    ) -> None:
        self.client = client
        self._tools: list[StructuredTool] = []
        self._secrets = (
            _configured_secrets()
            if secrets is None
            else tuple(sorted(set(secrets), key=len, reverse=True))
        )

    async def get_langchain_tools(self) -> list[StructuredTool]:
        """Discover MCP tools and return validated, server-qualified LangChain tools."""
        try:
            discovered = await self.client.discover_all_tools()
        except Exception as error:
            raise MCPToolAdapterError(
                f"MCP tool discovery failed ({type(error).__name__})."
            ) from None

        tools = []
        for qualified_name, metadata in discovered.items():
            server_name = metadata["server_name"]
            mcp_tool_name = metadata["tool_name"]
            description = _redact(
                metadata.get("description") or mcp_tool_name,
                self._secrets,
            )
            input_schema = metadata.get("input_schema", {})
            if not isinstance(input_schema, dict):
                raise MCPToolAdapterError(
                    f"MCP tool '{qualified_name}' has an invalid input schema."
                )
            input_schema = _redact(input_schema, self._secrets)
            args_schema = _input_model(qualified_name, input_schema)

            async def invoke_mcp_tool(
                _server_name: str = server_name,
                _tool_name: str = mcp_tool_name,
                _schema: type = args_schema,
                **arguments: Any,
            ) -> dict[str, Any]:
                try:
                    validated = _schema.model_validate(arguments)
                    result = await self.client.call_tool(
                        _server_name,
                        _tool_name,
                        validated.model_dump(exclude_unset=True),
                    )
                    if isinstance(result, dict) and result.get("is_error") is True:
                        raise MCPToolAdapterError(
                            f"MCP tool '{_tool_name}' reported an execution error "
                            f"on server '{_server_name}'."
                        )
                except MCPClientError as error:
                    raise MCPToolAdapterError(
                        f"MCP tool '{_tool_name}' failed on server "
                        f"'{_server_name}' ({type(error).__name__})."
                    ) from None
                except MCPToolAdapterError:
                    raise
                except Exception as error:
                    if error.__class__.__module__.startswith("pydantic"):
                        raise
                    raise MCPToolAdapterError(
                        f"MCP tool '{_tool_name}' failed on server "
                        f"'{_server_name}' ({type(error).__name__})."
                    ) from None
                return _redact(result, self._secrets)

            tool = StructuredTool(
                name=qualified_name,
                description=description,
                args_schema=args_schema,
                coroutine=invoke_mcp_tool,
                metadata={
                    "server_name": server_name,
                    "mcp_tool_name": mcp_tool_name,
                    "input_schema": input_schema,
                },
            )
            tools.append(tool)
        self._tools = tools
        return list(self._tools)


async def get_langchain_tools(
    client: MultiServerMCPClient,
) -> list[StructuredTool]:
    """Convenience function to dynamically adapt an existing MCP client."""
    return await LangChainMCPAdapter(client).get_langchain_tools()
