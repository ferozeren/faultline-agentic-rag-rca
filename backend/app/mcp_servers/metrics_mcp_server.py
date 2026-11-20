"""MCP tools for retrieving operational service metrics."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

BACKEND_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = BACKEND_ROOT / "data" / "metrics" / "processed" / "metrics.csv"
OPERATIONAL_FIELDS = (
    "timestamp",
    "service_name",
    "cpu_usage_percent",
    "memory_usage_percent",
    "request_count",
    "error_count",
    "avg_latency_ms",
    "active_connections",
    "db_connections",
    "disk_usage_percent",
)

mcp = FastMCP("Incident Metrics Server")


@lru_cache(maxsize=1)
def _load_metrics() -> list[dict[str, str]]:
    """Load operational columns from the local metrics CSV."""
    with METRICS_PATH.open(encoding="utf-8-sig", newline="") as input_file:
        return [
            {field: row[field] for field in OPERATIONAL_FIELDS}
            for row in csv.DictReader(input_file)
        ]


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer.")


def _matching_metrics(
    service_name: str | None,
    limit: int,
) -> list[dict[str, str]]:
    _validate_limit(limit)
    records = _load_metrics()
    if service_name is not None:
        requested_service = service_name.casefold()
        records = [
            record
            for record in records
            if record["service_name"].casefold() == requested_service
        ]
    return sorted(records, key=lambda record: record["timestamp"], reverse=True)[
        :limit
    ]


@mcp.tool()
def get_latest_metrics(limit: int = 20) -> list[dict[str, str]]:
    """Return the most recent operational metrics records."""
    return _matching_metrics(None, limit)


@mcp.tool()
def get_metrics_by_service(
    service_name: str,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Return the most recent metrics records for one service."""
    return _matching_metrics(service_name, limit)


def _usage_records(
    metric: str,
    service_name: str | None,
    limit: int,
) -> list[dict[str, str]]:
    return [
        {
            "timestamp": record["timestamp"],
            "service_name": record["service_name"],
            metric: record[metric],
        }
        for record in _matching_metrics(service_name, limit)
    ]


@mcp.tool()
def get_cpu_usage(
    service_name: str | None = None,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Return timestamped CPU usage, optionally for one service."""
    return _usage_records("cpu_usage_percent", service_name, limit)


@mcp.tool()
def get_memory_usage(
    service_name: str | None = None,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Return timestamped memory usage, optionally for one service."""
    return _usage_records("memory_usage_percent", service_name, limit)


@mcp.tool()
def get_latency_metrics(
    service_name: str | None = None,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Return timestamped average latency, optionally for one service."""
    return _usage_records("avg_latency_ms", service_name, limit)


@mcp.tool()
def get_error_metrics(
    service_name: str | None = None,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Return timestamped request and error counts, optionally for one service."""
    return [
        {
            "timestamp": record["timestamp"],
            "service_name": record["service_name"],
            "request_count": record["request_count"],
            "error_count": record["error_count"],
        }
        for record in _matching_metrics(service_name, limit)
    ]


@mcp.tool()
def get_database_connections(
    service_name: str | None = None,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Return timestamped database connection counts, optionally by service."""
    return _usage_records("db_connections", service_name, limit)


@mcp.tool()
def get_service_summary(service_name: str) -> dict[str, Any]:
    """Summarize supported operational metrics for one service."""
    records = _matching_metrics(service_name, len(_load_metrics()))
    if not records:
        return {"service_name": service_name, "record_count": 0}

    def values(field: str) -> list[float]:
        return [float(record[field]) for record in records]

    cpu = values("cpu_usage_percent")
    memory = values("memory_usage_percent")
    latency = values("avg_latency_ms")

    return {
        "service_name": service_name,
        "record_count": len(records),
        "average_cpu_usage_percent": round(sum(cpu) / len(cpu), 2),
        "maximum_cpu_usage_percent": max(cpu),
        "average_memory_usage_percent": round(sum(memory) / len(memory), 2),
        "maximum_memory_usage_percent": max(memory),
        "average_latency_ms": round(sum(latency) / len(latency), 2),
        "maximum_latency_ms": max(latency),
        "total_requests": sum(int(record["request_count"]) for record in records),
        "total_errors": sum(int(record["error_count"]) for record in records),
        "maximum_db_connections": max(
            int(record["db_connections"]) for record in records
        ),
    }


@lru_cache(maxsize=1)
def _dataset_catalog() -> dict[str, Any]:
    """Describe available metrics without including evaluation labels."""
    records = _load_metrics()
    timestamps = [record["timestamp"] for record in records]
    return {
        "dataset_name": METRICS_PATH.name,
        "available_services": sorted(
            {record["service_name"] for record in records}
        ),
        "fields": list(OPERATIONAL_FIELDS),
        "record_count": len(records),
        "timestamp_range": {
            "start": min(timestamps) if timestamps else None,
            "end": max(timestamps) if timestamps else None,
        },
        "supported_metrics": list(OPERATIONAL_FIELDS[2:]),
    }


@mcp.resource(
    "metrics://available",
    name="available_metrics",
    description="Available operational metrics, services, fields, and time range.",
)
def available_metrics() -> dict[str, Any]:
    """Describe the operational metrics available to MCP tools."""
    return _dataset_catalog()


def main() -> None:
    """Run the standalone MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
