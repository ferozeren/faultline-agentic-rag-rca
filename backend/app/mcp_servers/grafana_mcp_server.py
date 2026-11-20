"""MCP tools for discovering Grafana dashboards and alerting data."""

from __future__ import annotations

import os
import re
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from dotenv import dotenv_values
from mcp.server.fastmcp import FastMCP

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REQUEST_TIMEOUT_SECONDS = 10.0
GRAFANA_URL_ENV = "GRAFANA_URL"
GRAFANA_TOKEN_ENV = "GRAFANA_SERVICE_ACCOUNT_TOKEN"


def _load_grafana_environment() -> None:
    """Load only Grafana settings from .env without overriding exported values."""
    env_path = BACKEND_ROOT / ".env"
    if not env_path.exists():
        return

    setting_names = {GRAFANA_URL_ENV, GRAFANA_TOKEN_ENV}
    selected_lines = []
    with env_path.open(encoding="utf-8") as env_file:
        for line in env_file:
            assignment = re.match(
                r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=",
                line,
            )
            if assignment and assignment.group(1) in setting_names:
                selected_lines.append(line)
    values = dotenv_values(stream=StringIO("".join(selected_lines)))
    for name in setting_names:
        value = values.get(name)
        if value is not None:
            os.environ.setdefault(name, value)


_load_grafana_environment()
mcp = FastMCP("Grafana Server")


class GrafanaError(RuntimeError):
    """A safe, user-facing Grafana configuration or API error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code


def _configuration() -> tuple[str, str]:
    """Return a validated base URL and service-account token."""
    base_url = os.getenv(GRAFANA_URL_ENV, "").strip()
    if not base_url:
        raise GrafanaError("GRAFANA_URL is not configured.")
    try:
        parsed_url = urlsplit(base_url)
        parsed_url.port
    except ValueError:
        raise GrafanaError("GRAFANA_URL must be a valid HTTP or HTTPS URL.") from None
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or any(character.isspace() for character in base_url)
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise GrafanaError("GRAFANA_URL must be a valid HTTP or HTTPS URL.")

    token = os.getenv(GRAFANA_TOKEN_ENV, "").strip()
    if not token:
        raise GrafanaError("GRAFANA_SERVICE_ACCOUNT_TOKEN is not configured.")
    return base_url.rstrip("/"), token


def _redact(value: Any, token: str) -> Any:
    """Remove a configured token from API-provided values."""
    if isinstance(value, str):
        return value.replace(token, "[REDACTED]") if token else value
    if isinstance(value, list):
        return [_redact(item, token) for item in value]
    if isinstance(value, dict):
        return {_redact(key, token): _redact(item, token) for key, item in value.items()}
    return value


def _request_data(path: str, params: dict[str, str] | None = None) -> Any:
    """Call the Grafana API and return its JSON response."""
    base_url, token = _configuration()
    try:
        response = httpx.get(
            f"{base_url}{path}",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        raise GrafanaError("Grafana request timed out.") from None
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        if status == 401:
            message = "Grafana authentication failed (HTTP 401)."
        elif status == 403:
            message = "Grafana access was denied (HTTP 403)."
        elif status == 404:
            message = "Grafana resource or API endpoint was not found (HTTP 404)."
        elif status == 429:
            message = "Grafana rate limit exceeded (HTTP 429)."
        elif status >= 500:
            message = f"Grafana server error (HTTP {status})."
        else:
            message = f"Grafana returned HTTP {status}."
        raise GrafanaError(message, status_code=status) from None
    except httpx.RequestError:
        raise GrafanaError("Could not connect to the Grafana API.") from None

    try:
        payload = response.json()
    except (ValueError, TypeError):
        raise GrafanaError("Grafana returned an invalid JSON response.") from None

    return _redact(payload, token)


def _dashboard_metadata(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise GrafanaError("Grafana returned an invalid dashboard search response.")
    return {
        "uid": item.get("uid"),
        "title": item.get("title"),
        "url": item.get("url"),
        "folder_title": item.get("folderTitle"),
        "type": item.get("type"),
    }


def _search_dashboards(query: str | None = None) -> list[dict[str, Any]]:
    params = {"type": "dash-db"}
    if query is not None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty search keyword.")
        params["query"] = query.strip()
    data = _request_data("/api/search", params)
    if not isinstance(data, list):
        raise GrafanaError("Grafana returned an invalid dashboard search response.")
    return [_dashboard_metadata(item) for item in data]


@mcp.tool()
def list_dashboards() -> list[dict[str, Any]]:
    """List dashboards available in Grafana."""
    return _search_dashboards()


@mcp.tool()
def search_dashboards(query: str) -> list[dict[str, Any]]:
    """Search Grafana dashboards by title or keyword."""
    return _search_dashboards(query)


def _datasource_references(value: Any) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "datasource" and child is not None:
                if child not in found:
                    found.append(child)
            found.extend(_datasource_references(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_datasource_references(child))
    unique = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


@mcp.tool()
def get_dashboard(uid: str) -> dict[str, Any]:
    """Retrieve a dashboard's configuration by UID."""
    if not isinstance(uid, str) or not uid.strip():
        raise ValueError("uid must be a non-empty dashboard UID.")
    data = _request_data(f"/api/dashboards/uid/{quote(uid.strip(), safe='')}")
    if not isinstance(data, dict) or not isinstance(data.get("dashboard"), dict):
        raise GrafanaError("Grafana returned an invalid dashboard response.")
    dashboard = data["dashboard"]
    metadata = data.get("meta", {})
    if not isinstance(metadata, dict):
        raise GrafanaError("Grafana returned an invalid dashboard response.")
    panels = dashboard.get("panels", [])
    if not isinstance(panels, list):
        raise GrafanaError("Grafana returned an invalid dashboard response.")

    return {
        "title": dashboard.get("title"),
        "uid": dashboard.get("uid"),
        "description": dashboard.get("description"),
        "panels": panels,
        "datasource_references": _datasource_references(dashboard),
        "configuration_metadata": {
            key: dashboard[key]
            for key in (
                "id",
                "version",
                "schemaVersion",
                "timezone",
                "refresh",
                "time",
                "timepicker",
                "templating",
                "annotations",
                "links",
                "tags",
            )
            if key in dashboard
        },
        "dashboard_configuration": dashboard,
        "metadata": metadata,
    }


@mcp.tool()
def list_alert_rules() -> list[dict[str, Any]]:
    """List Grafana-managed alert rules."""
    data = _request_data("/api/v1/provisioning/alert-rules")
    if not isinstance(data, list):
        raise GrafanaError("Grafana returned an invalid alert-rules response.")
    rules = []
    for rule in data:
        if not isinstance(rule, dict):
            raise GrafanaError("Grafana returned an invalid alert-rules response.")
        rules.append(
            {
                "uid": rule.get("uid"),
                "title": rule.get("title"),
                "condition": rule.get("condition"),
                "labels": rule.get("labels", {}),
                "annotations": rule.get("annotations", {}),
                "folder_uid": rule.get("folderUID"),
                "folder_title": rule.get("folderTitle"),
                "evaluation_interval": rule.get("intervalSeconds"),
                "rule_group": rule.get("ruleGroup"),
            }
        )
    return rules


@mcp.tool()
def get_alert_instances() -> list[dict[str, Any]]:
    """Retrieve Grafana alert instances and their current states."""
    data = _request_data("/api/alertmanager/grafana/api/v2/alerts")
    if not isinstance(data, list):
        raise GrafanaError("Grafana returned an invalid alert-instances response.")
    alerts = []
    for alert in data:
        if not isinstance(alert, dict):
            raise GrafanaError("Grafana returned an invalid alert-instances response.")
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        if not isinstance(labels, dict) or not isinstance(annotations, dict):
            raise GrafanaError("Grafana returned an invalid alert-instances response.")
        status = alert.get("status", {})
        if not isinstance(status, dict):
            raise GrafanaError("Grafana returned an invalid alert-instances response.")
        alerts.append(
            {
                "alert_name": labels.get("alertname", ""),
                "state": status.get("state"),
                "labels": labels,
                "annotations": annotations,
                "active_at": alert.get("startsAt"),
                "fingerprint": alert.get("fingerprint"),
            }
        )
    return alerts


@mcp.tool()
def get_grafana_health() -> dict[str, Any]:
    """Check Grafana API connectivity and report health metadata."""
    try:
        data = _request_data("/api/health")
    except GrafanaError as error:
        message = str(error)
        if error.status_code == 404:
            message = (
                "Grafana GET /api/health returned HTTP 404; verify GRAFANA_URL "
                "is the active Grafana instance URL and includes any "
                "reverse-proxy subpath. For Grafana Cloud, use the URL shown "
                "for the active stack."
            )
        return {
            "connected": False,
            "database": None,
            "version": None,
            "commit": None,
            "error": message,
            "http_status": error.status_code,
        }
    if not isinstance(data, dict):
        return {
            "connected": False,
            "database": None,
            "version": None,
            "commit": None,
            "error": "Grafana returned an invalid health response.",
            "http_status": 200,
        }
    return {
        "connected": True,
        "database": data.get("database"),
        "version": data.get("version"),
        "commit": data.get("commit"),
        "error": None,
        "http_status": 200,
    }


@mcp.resource(
    "grafana://available",
    name="available_grafana",
    description="Grafana connection configuration and supported API operations.",
)
def available_grafana() -> dict[str, Any]:
    """Describe Grafana configuration without exposing credentials."""
    url = os.getenv(GRAFANA_URL_ENV, "").strip()
    try:
        _configuration()
        url_valid = True
    except GrafanaError:
        url_valid = False
    return {
        "configuration": {
            "url_configured": bool(url),
            "url_valid": url_valid,
            "service_account_token_configured": bool(
                os.getenv(GRAFANA_TOKEN_ENV, "").strip()
            ),
            "authentication_method": "bearer",
        },
        "tools": [
            "list_dashboards",
            "search_dashboards",
            "get_dashboard",
            "list_alert_rules",
            "get_alert_instances",
            "get_grafana_health",
        ],
        "supported_api_operations": [
            "GET /api/search?type=dash-db",
            "GET /api/dashboards/uid/{uid}",
            "GET /api/v1/provisioning/alert-rules",
            "GET /api/alertmanager/grafana/api/v2/alerts",
            "GET /api/health",
        ],
    }


def main() -> None:
    """Run the standalone MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
