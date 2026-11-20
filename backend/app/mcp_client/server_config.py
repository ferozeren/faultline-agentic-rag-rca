"""Local stdio configuration for the project's standalone MCP servers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration needed to launch one local MCP server."""

    name: str
    module: str
    command: str
    cwd: Path
    args: tuple[str, ...]


def load_server_configs() -> dict[str, MCPServerConfig]:
    """Return all configured servers using this Python environment."""
    modules = {
        "log": "app.mcp_servers.log_mcp_server",
        "metrics": "app.mcp_servers.metrics_mcp_server",
        "prometheus": "app.mcp_servers.prometheus_mcp_server",
        "grafana": "app.mcp_servers.grafana_mcp_server",
    }
    return {
        name: MCPServerConfig(
            name=name,
            module=module,
            command=sys.executable,
            cwd=BACKEND_ROOT,
            args=("-m", module),
        )
        for name, module in modules.items()
    }
