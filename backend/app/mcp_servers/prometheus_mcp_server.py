"""MCP tools for querying a Prometheus HTTP API."""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values
from mcp.server.fastmcp import FastMCP

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REQUEST_TIMEOUT_SECONDS = 10.0
PROMETHEUS_URL_ENV = "PROMETHEUS_URL"
PROMETHEUS_TOKEN_ENV = "PROMETHEUS_BEARER_TOKEN"
PROMETHEUS_USERNAME_ENV = "PROMETHEUS_USERNAME"


def _load_prometheus_environment() -> None:
    """Load only Prometheus settings from .env without overriding the process."""
    env_path = BACKEND_ROOT / ".env"
    if not env_path.exists():
        return

    setting_names = {
        PROMETHEUS_URL_ENV,
        PROMETHEUS_TOKEN_ENV,
        PROMETHEUS_USERNAME_ENV,
    }
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


_load_prometheus_environment()
mcp = FastMCP("Prometheus Server")


class PrometheusError(RuntimeError):
    """A safe, user-facing Prometheus configuration or API error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        auth_schemes: list[str] | None = None,
        response_content_type: str | None = None,
        response_detail: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.auth_schemes = auth_schemes or []
        self.response_content_type = response_content_type
        self.response_detail = response_detail


def _configuration() -> tuple[str, str, str]:
    """Return a validated base URL, optional token, and optional username."""
    base_url = os.getenv(PROMETHEUS_URL_ENV, "").strip()
    if not base_url:
        raise PrometheusError("PROMETHEUS_URL is not configured.")

    try:
        parsed_url = urlsplit(base_url)
        parsed_url.port
    except ValueError:
        raise PrometheusError(
            "PROMETHEUS_URL must be a valid HTTP or HTTPS URL."
        ) from None
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or any(character.isspace() for character in base_url)
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise PrometheusError("PROMETHEUS_URL must be a valid HTTP or HTTPS URL.")

    token = os.getenv(PROMETHEUS_TOKEN_ENV, "").strip()
    username = os.getenv(PROMETHEUS_USERNAME_ENV, "").strip()
    if username and not token:
        raise PrometheusError(
            "PROMETHEUS_BEARER_TOKEN is required with PROMETHEUS_USERNAME."
        )
    return base_url.rstrip("/"), token, username


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    """Remove configured credentials from API-provided values."""
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            _redact(key, secrets): _redact(item, secrets)
            for key, item in value.items()
        }
    return value


def _auth_schemes(challenge: str) -> list[str]:
    return sorted(set(re.findall(r"(?:^|,)\s*([A-Za-z][A-Za-z0-9_-]*)", challenge)))


def _safe_response_detail(response: httpx.Response, secrets: tuple[str, ...]) -> str | None:
    """Extract only a short JSON error detail, with configured values redacted."""
    try:
        payload = response.json()
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    error = payload.get("error")
    error_type = payload.get("errorType")
    parts = []
    if isinstance(error_type, str) and error_type:
        parts.append(error_type)
    if isinstance(error, str) and error:
        parts.append(error)
    if not parts:
        return None
    return str(_redact(": ".join(parts), secrets))[:300]


def _request_data_with_status(
    path: str,
    params: dict[str, str] | None = None,
) -> tuple[Any, int]:
    """Call Prometheus and return validated data plus the HTTP status."""
    base_url, token, username = _configuration()
    auth = None
    headers = {}
    if username and token:
        auth = httpx.BasicAuth(username, token)
    elif token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = httpx.get(
            f"{base_url}{path}",
            params=params,
            auth=auth,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        raise PrometheusError("Prometheus request timed out.") from None
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        auth_schemes = _auth_schemes(
            error.response.headers.get("www-authenticate", "")
        )
        content_type = error.response.headers.get("content-type")
        response_detail = _safe_response_detail(error.response, (token, username))
        if status == 401:
            message = "Prometheus authentication failed (HTTP 401)."
        elif status == 403:
            message = "Prometheus access was denied (HTTP 403)."
        else:
            message = f"Prometheus returned HTTP {status}."
        raise PrometheusError(
            message,
            status_code=status,
            auth_schemes=auth_schemes,
            response_content_type=content_type,
            response_detail=response_detail,
        ) from None
    except httpx.RequestError:
        raise PrometheusError("Could not connect to the Prometheus API.") from None

    try:
        payload = response.json()
    except (ValueError, TypeError):
        raise PrometheusError("Prometheus returned an invalid JSON response.") from None

    if not isinstance(payload, dict):
        raise PrometheusError("Prometheus returned an invalid API response.")

    status = payload.get("status")
    if status == "error":
        error_type = payload.get("errorType")
        api_message = payload.get("error")
        detail = f": {api_message}" if isinstance(api_message, str) else ""
        if isinstance(error_type, str) and error_type:
            message = f"Prometheus API error ({error_type}){detail}."
        else:
            message = f"Prometheus API error{detail}."
        raise PrometheusError(
            _redact(message, (token, username)),
            status_code=response.status_code,
            response_content_type=response.headers.get("content-type"),
        )
    if status != "success" or "data" not in payload:
        raise PrometheusError("Prometheus returned an invalid API response.")

    return _redact(payload["data"], (token, username)), response.status_code


def _request_data(path: str, params: dict[str, str] | None = None) -> Any:
    """Call a Prometheus API endpoint and return its validated data value."""
    data, _ = _request_data_with_status(path, params)
    return data


def _authentication_status() -> str:
    token = os.getenv(PROMETHEUS_TOKEN_ENV, "").strip()
    username = os.getenv(PROMETHEUS_USERNAME_ENV, "").strip()
    if username and token:
        return "basic"
    if username:
        return "basic_incomplete"
    if token:
        return "bearer"
    return "none"


def _validate_query_data(
    data: Any,
    expected_result_type: str | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("resultType"), str)
        or not isinstance(data.get("result"), list)
        or (
            expected_result_type is not None
            and data["resultType"] != expected_result_type
        )
    ):
        raise PrometheusError("Prometheus returned an invalid query response.")
    return data


def _query_with_status(
    path: str,
    params: dict[str, str],
    expected_result_type: str,
) -> tuple[dict[str, Any], int]:
    data, status_code = _request_data_with_status(path, params)
    return _validate_query_data(data, expected_result_type), status_code


def _validate_timestamp(value: str, name: str) -> datetime | float:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty timestamp.")
    try:
        numeric_value = float(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"{name} must be a valid timestamp.") from None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite timestamp.")
    return numeric_value


def _timestamp_seconds(value: datetime | float) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    return value


def _validate_step(step: str) -> None:
    if not isinstance(step, str) or not step.strip():
        raise ValueError("step must be a positive number or Prometheus duration.")
    value = step.strip()
    try:
        numeric_step = float(value)
    except ValueError:
        parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h|d|w|y)", value)
        if parts and "".join(amount + unit for amount, unit in parts) == value:
            seconds_per_unit = {
                "ms": 0.001,
                "s": 1,
                "m": 60,
                "h": 3600,
                "d": 86400,
                "w": 604800,
                "y": 31536000,
            }
            duration_seconds = sum(
                float(amount) * seconds_per_unit[unit] for amount, unit in parts
            )
            if duration_seconds > 0:
                return
            raise ValueError("step must be greater than zero.")
        raise ValueError("step must be a positive number or Prometheus duration.")
    if not math.isfinite(numeric_step) or numeric_step <= 0:
        raise ValueError("step must be greater than zero.")


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty PromQL expression.")


@mcp.tool()
def query_prometheus(query: str, time: str | None = None) -> dict[str, Any]:
    """Execute an instant PromQL query."""
    _validate_query(query)
    params = {"query": query}
    if time is not None:
        _validate_timestamp(time, "time")
        params["time"] = time
    data, _ = _query_with_status("/api/v1/query", params, "vector")
    return data


@mcp.tool()
def query_prometheus_range(
    query: str,
    start: str,
    end: str,
    step: str,
) -> dict[str, Any]:
    """Retrieve Prometheus time-series data for a time range."""
    _validate_query(query)
    start_time = _validate_timestamp(start, "start")
    end_time = _validate_timestamp(end, "end")
    if _timestamp_seconds(start_time) >= _timestamp_seconds(end_time):
        raise ValueError("end must be later than start.")
    _validate_step(step)
    data, _ = _query_with_status(
        "/api/v1/query_range",
        {"query": query, "start": start, "end": end, "step": step},
        "matrix",
    )
    return data


def _targets_with_status() -> tuple[list[dict[str, Any]], int]:
    data, status_code = _request_data_with_status("/api/v1/targets")
    if not isinstance(data, dict) or not isinstance(data.get("activeTargets"), list):
        raise PrometheusError("Prometheus returned an invalid targets response.")

    fields = (
        "health",
        "scrapeUrl",
        "lastScrape",
        "lastScrapeDuration",
        "lastError",
        "labels",
    )
    targets = []
    for target in data["activeTargets"]:
        if not isinstance(target, dict):
            raise PrometheusError("Prometheus returned an invalid targets response.")
        targets.append(
            {field: target[field] for field in fields if field in target}
        )
    return targets, status_code


@mcp.tool()
def get_prometheus_targets() -> list[dict[str, Any]]:
    """Return health and scrape details for active Prometheus targets."""
    targets, _ = _targets_with_status()
    return targets


def _alerts_with_status() -> tuple[list[dict[str, Any]], int]:
    data, status_code = _request_data_with_status("/api/v1/alerts")
    if not isinstance(data, dict) or not isinstance(data.get("alerts"), list):
        raise PrometheusError("Prometheus returned an invalid alerts response.")

    alerts = []
    for alert in data["alerts"]:
        if not isinstance(alert, dict):
            raise PrometheusError("Prometheus returned an invalid alerts response.")
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        if not isinstance(labels, dict) or not isinstance(annotations, dict):
            raise PrometheusError("Prometheus returned an invalid alerts response.")
        alerts.append(
            {
                "alert_name": labels.get("alertname", ""),
                "state": alert.get("state", ""),
                "labels": labels,
                "annotations": annotations,
                "active_at": alert.get("activeAt"),
            }
        )
    return alerts, status_code


@mcp.tool()
def get_prometheus_alerts() -> list[dict[str, Any]]:
    """Return alerts currently known to Prometheus."""
    alerts, _ = _alerts_with_status()
    return alerts


def _health_with_status() -> tuple[dict[str, Any], int]:
    data, status_code = _request_data_with_status("/api/v1/status/buildinfo")
    if not isinstance(data, dict):
        raise PrometheusError(
            "Prometheus returned an invalid health response.",
            status_code=status_code,
        )
    version = data.get("version")
    return {
        "connected": True,
        "prometheus_version": version if isinstance(version, str) else None,
        "error": None,
    }, status_code


@mcp.tool()
def get_prometheus_health() -> dict[str, Any]:
    """Check Prometheus API connectivity and report its version if available."""
    try:
        health, _ = _health_with_status()
    except PrometheusError as error:
        return {"connected": False, "prometheus_version": None, "error": str(error)}
    return health


@mcp.resource(
    "prometheus://available",
    name="available_prometheus",
    description="Prometheus connection status and supported API operations.",
)
def available_prometheus() -> dict[str, Any]:
    """Describe configuration status and the supported Prometheus operations."""
    try:
        _configuration()
        url_configured = True
        url_valid = True
    except PrometheusError:
        url_configured = bool(os.getenv(PROMETHEUS_URL_ENV, "").strip())
        url_valid = False

    return {
        "connection_configuration": {
            "url_configured": url_configured,
            "url_valid": url_valid,
            "bearer_token_configured": bool(
                os.getenv(PROMETHEUS_TOKEN_ENV, "").strip()
            ),
            "username_configured": bool(
                os.getenv(PROMETHEUS_USERNAME_ENV, "").strip()
            ),
            "authentication_method": _authentication_status(),
        },
        "tools": [
            "query_prometheus",
            "query_prometheus_range",
            "get_prometheus_targets",
            "get_prometheus_alerts",
            "get_prometheus_health",
        ],
        "supported_api_operations": [
            "/api/v1/query",
            "/api/v1/query_range",
            "/api/v1/targets",
            "/api/v1/alerts",
            "/api/v1/status/buildinfo",
        ],
    }


def main() -> None:
    """Run the standalone MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
