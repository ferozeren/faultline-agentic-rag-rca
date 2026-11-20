"""Safely validate live Prometheus API connectivity and read-only operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

from app.mcp_servers import prometheus_mcp_server as prometheus


def _result(
    operation: str,
    status_code: int | None,
    *,
    structure_valid: bool,
    count: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "http_status": status_code,
        "success": status_code is not None
        and 200 <= status_code < 300
        and structure_valid,
        "response_structure_valid": structure_valid,
        "result_count": count,
        **(details or {}),
    }


def _failure(operation: str, error: Exception) -> dict[str, Any]:
    status_code = (
        error.status_code
        if isinstance(error, prometheus.PrometheusError)
        else None
    )
    details = {"error": str(error)}
    if isinstance(error, prometheus.PrometheusError):
        if error.auth_schemes:
            details["authentication_challenge_schemes"] = error.auth_schemes
        if error.response_content_type:
            details["response_content_type"] = error.response_content_type
        if error.response_detail:
            details["safe_response_detail"] = error.response_detail
    return _result(
        operation,
        status_code,
        structure_valid=False,
        details=details,
    )


def _up_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"up": 0, "down": 0, "other": 0}
    for series in results:
        sample = series.get("value")
        if not isinstance(sample, list) or len(sample) != 2:
            raise prometheus.PrometheusError(
                "Prometheus returned an invalid instant-query series."
            )
        value = sample[1]
        if value == "1":
            counts["up"] += 1
        elif value == "0":
            counts["down"] += 1
        else:
            counts["other"] += 1
    return counts


def _range_sample_count(results: list[dict[str, Any]]) -> int:
    sample_count = 0
    for series in results:
        values = series.get("values")
        if not isinstance(values, list):
            raise prometheus.PrometheusError(
                "Prometheus returned an invalid range-query series."
            )
        sample_count += len(values)
    return sample_count


def run_validation() -> dict[str, Any]:
    """Run each read-only API check, stopping after an authentication failure."""
    try:
        base_url, _, _ = prometheus._configuration()
        host = urlsplit(base_url).hostname or ""
        endpoint_kind = (
            "Grafana Cloud hosted Prometheus"
            if host.endswith("grafana.net")
            else "other Prometheus HTTP endpoint"
        )
        configuration = {
            "url_configured": True,
            "url_valid": True,
            "endpoint_kind": endpoint_kind,
            "authentication_method": prometheus._authentication_status(),
        }
    except prometheus.PrometheusError as error:
        return {
            "configuration": {
                "url_configured": bool(
                    prometheus.os.getenv(prometheus.PROMETHEUS_URL_ENV, "").strip()
                ),
                "url_valid": False,
                "authentication_method": prometheus._authentication_status(),
            },
            "operations": [],
            "final_status": "CONNECTION FAILED",
            "error": str(error),
        }

    operations: list[dict[str, Any]] = []
    try:
        health, status_code = prometheus._health_with_status()
        operations.append(
            _result(
                "/api/v1/status/buildinfo",
                status_code,
                structure_valid=health["connected"],
                details={"prometheus_version": health["prometheus_version"]},
            )
        )
    except prometheus.PrometheusError as error:
        operations.append(_failure("/api/v1/status/buildinfo", error))
        if error.status_code in {401, 403}:
            final_status = (
                "AUTHENTICATION FAILED" if error.status_code == 401 else "PARTIALLY VERIFIED"
            )
            return {
                "configuration": configuration,
                "operations": operations,
                "final_status": final_status,
            }
        return {
            "configuration": configuration,
            "operations": operations,
            "final_status": "CONNECTION FAILED",
        }

    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=5)
    end = now

    try:
        query_data, status_code = prometheus._query_with_status(
            "/api/v1/query",
            {"query": "up"},
            "vector",
        )
        counts = _up_counts(query_data["result"])
        operations.append(
            _result(
                "/api/v1/query",
                status_code,
                structure_valid=True,
                count=len(query_data["result"]),
                details={
                    "promql": "up",
                    "up_value_counts": counts,
                    "up_series_interpretation": (
                        "Each result value of 1 means the monitored target is up; "
                        "0 means it is down."
                    ),
                    "interpretation": {
                        "1": "monitored target is up",
                        "0": "monitored target is down",
                    },
                },
            )
        )
    except prometheus.PrometheusError as error:
        operations.append(_failure("/api/v1/query", error))
        if error.status_code == 401:
            return {
                "configuration": configuration,
                "operations": operations,
                "final_status": "AUTHENTICATION FAILED",
            }

    try:
        range_data, status_code = prometheus._query_with_status(
            "/api/v1/query_range",
            {
                "query": "up",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "step": "30s",
            },
            "matrix",
        )
        samples = _range_sample_count(range_data["result"])
        operations.append(
            _result(
                "/api/v1/query_range",
                status_code,
                structure_valid=True,
                count=len(range_data["result"]),
                details={
                    "series_count": len(range_data["result"]),
                    "sample_count": samples,
                },
            )
        )
    except prometheus.PrometheusError as error:
        operations.append(_failure("/api/v1/query_range", error))
        if error.status_code == 401:
            return {
                "configuration": configuration,
                "operations": operations,
                "final_status": "AUTHENTICATION FAILED",
            }

    try:
        targets, status_code = prometheus._targets_with_status()
        operations.append(
            _result(
                "/api/v1/targets",
                status_code,
                structure_valid=True,
                count=len(targets),
            )
        )
    except prometheus.PrometheusError as error:
        operations.append(_failure("/api/v1/targets", error))
        if error.status_code == 401:
            return {
                "configuration": configuration,
                "operations": operations,
                "final_status": "AUTHENTICATION FAILED",
            }

    try:
        alerts, status_code = prometheus._alerts_with_status()
        operations.append(
            _result(
                "/api/v1/alerts",
                status_code,
                structure_valid=True,
                count=len(alerts),
            )
        )
    except prometheus.PrometheusError as error:
        operations.append(_failure("/api/v1/alerts", error))
        if error.status_code == 401:
            return {
                "configuration": configuration,
                "operations": operations,
                "final_status": "AUTHENTICATION FAILED",
            }

    all_succeeded = all(operation["success"] for operation in operations)
    some_succeeded = any(operation["success"] for operation in operations)
    final_status = (
        "LIVE CONNECTION SUCCESSFUL"
        if all_succeeded
        else "PARTIALLY VERIFIED"
        if some_succeeded
        else "CONNECTION FAILED"
    )
    return {
        "configuration": configuration,
        "operations": operations,
        "final_status": final_status,
    }


def main() -> int:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    report = run_validation()
    print(json.dumps(report, indent=2))
    return 0 if report["final_status"] == "LIVE CONNECTION SUCCESSFUL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
