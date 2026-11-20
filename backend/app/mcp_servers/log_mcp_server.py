"""MCP tools for searching the existing application log datasets."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

BACKEND_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = BACKEND_ROOT / "data" / "logs"
PROCESSED_ROOT = LOG_ROOT / "processed"
DATASET_FILES = {
    "Apache_log.csv": PROCESSED_ROOT / "Apache_log.csv",
    "payment_api_logs.csv": PROCESSED_ROOT / "payment_api_logs.csv",
}
LEVEL_FIELDS = ("level", "Level")

mcp = FastMCP("Incident Log Server")


@lru_cache(maxsize=1)
def _load_datasets() -> dict[str, list[dict[str, str]]]:
    """Load processed CSV datasets once for the lifetime of this server."""
    datasets = {}
    for name, path in DATASET_FILES.items():
        with path.open(encoding="utf-8-sig", newline="") as input_file:
            datasets[name] = list(csv.DictReader(input_file))
    return datasets


def _records() -> list[tuple[str, dict[str, str]]]:
    return [
        (dataset_name, record)
        for dataset_name, dataset in _load_datasets().items()
        for record in dataset
    ]


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer.")


def _record_time(record: dict[str, str]) -> float:
    parsed = _parse_timestamp(record.get("timestamp") or record.get("Time") or "")
    return parsed.timestamp() if parsed is not None else float("-inf")


def _parse_timestamp(value: str) -> datetime | None:
    if not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(value.strip(), "%a %b %d %H:%M:%S %Y")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(timezone.utc)


def _record_service(dataset_name: str, record: dict[str, str]) -> str | None:
    for key in ("service", "service_name", "serviceName"):
        value = record.get(key)
        if value:
            return value
    if dataset_name == "payment_api_logs.csv":
        return "payment-api"
    return None


def _parse_filter_timestamp(value: str | None, name: str) -> datetime | None:
    if value is None:
        return None
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise ValueError(f"{name} must be a valid ISO 8601 timestamp.")
    return parsed


def _with_dataset(dataset_name: str, record: dict[str, str]) -> dict[str, str]:
    return {"dataset": dataset_name, **record}


def _level_value(record: dict[str, str]) -> str:
    for field in LEVEL_FIELDS:
        if field in record:
            return record[field].strip().upper()
    return ""


def _available_levels() -> list[str]:
    return sorted(
        {
            level
            for _, record in _records()
            if (level := _level_value(record))
        }
    )


@mcp.tool()
def get_recent_logs(limit: int = 20) -> list[dict[str, str]]:
    """Return the most recent records across the processed log datasets."""
    _validate_limit(limit)
    recent = sorted(
        _records(),
        key=lambda item: _record_time(item[1]),
        reverse=True,
    )[:limit]
    return [_with_dataset(name, record) for name, record in recent]


@mcp.tool()
def query_logs(
    service_name: str | None = None,
    level: str | None = None,
    keyword: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Filter and paginate log records without assigning unknown services."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000.")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer.")
    if service_name is not None and (
        not isinstance(service_name, str) or not service_name.strip()
    ):
        raise ValueError("service_name must be a non-empty string.")
    if level is not None and (not isinstance(level, str) or not level.strip()):
        raise ValueError("level must be a non-empty string.")
    if keyword is not None and (not isinstance(keyword, str) or not keyword.strip()):
        raise ValueError("keyword must be a non-empty string.")

    start = _parse_filter_timestamp(start_time, "start_time")
    end = _parse_filter_timestamp(end_time, "end_time")
    if start is not None and end is not None and start > end:
        raise ValueError("end_time must be at or after start_time.")

    matching = []
    excluded_untimed = 0
    excluded_outside_range = 0
    for dataset_name, record in _records():
        record_service = _record_service(dataset_name, record)
        if service_name and (
            record_service is None
            or record_service.casefold() != service_name.strip().casefold()
        ):
            continue
        if level and _level_value(record) != level.strip().upper():
            continue
        if keyword and keyword.strip().casefold() not in " ".join(
            (dataset_name, *record.values())
        ).casefold():
            continue

        record_timestamp = _parse_timestamp(
            record.get("timestamp") or record.get("Time") or ""
        )
        if start is not None or end is not None:
            if record_timestamp is None:
                excluded_untimed += 1
                continue
            if (
                (start is not None and record_timestamp < start)
                or (end is not None and record_timestamp > end)
            ):
                excluded_outside_range += 1
                continue
        matching.append((dataset_name, record))

    matching.sort(key=lambda item: _record_time(item[1]), reverse=True)
    total = len(matching)
    page = matching[offset : offset + limit]
    return {
        "records": [_with_dataset(name, record) for name, record in page],
        "total": total,
        "limit": limit,
        "offset": offset,
        "excluded_untimed_count": excluded_untimed,
        "excluded_outside_range_count": excluded_outside_range,
    }


@mcp.tool()
def search_logs(keyword: str, limit: int = 20) -> list[dict[str, str]]:
    """Search all fields in the processed log datasets, case-insensitively."""
    _validate_limit(limit)
    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError("keyword must be a non-empty string.")
    normalized_keyword = keyword.casefold()
    matches = [
        _with_dataset(name, record)
        for name, record in _records()
        if any(normalized_keyword in value.casefold() for value in record.values())
    ]
    return matches[:limit]


@mcp.tool()
def filter_logs_by_level(level: str, limit: int = 20) -> list[dict[str, str]]:
    """Return log records matching an available severity level."""
    _validate_limit(limit)
    if not isinstance(level, str) or not level.strip():
        raise ValueError("level must be a non-empty string.")

    requested_level = level.strip().upper()
    available_levels = _available_levels()
    if requested_level not in available_levels:
        choices = ", ".join(available_levels) or "none"
        raise ValueError(
            f"Unsupported level '{level}'. Available levels: {choices}."
        )

    matches = [
        _with_dataset(name, record)
        for name, record in _records()
        if _level_value(record) == requested_level
    ]
    return matches[:limit]


@mcp.tool()
def get_error_logs(limit: int = 20) -> list[dict[str, str]]:
    """Return records marked as errors or indicating failed requests."""
    _validate_limit(limit)
    error_records = []
    for name, record in _records():
        level = _level_value(record)
        outcome = record.get("outcome", "").casefold()
        error_code = record.get("error_code", "").strip()
        error_message = record.get("error_message", "").strip()
        content = record.get("Content", "").casefold()
        try:
            server_error = int(record.get("http_status", "")) >= 500
        except ValueError:
            server_error = False

        if (
            level in {"ERROR", "ERR", "CRITICAL", "FATAL"}
            or outcome in {"error", "failed", "failure"}
            or bool(error_code or error_message)
            or server_error
            or "error" in content
        ):
            error_records.append(_with_dataset(name, record))
    return error_records[:limit]


def _raw_jsonl_info(path: Path) -> tuple[int, list[str]]:
    count = 0
    fields: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                count += 1
                fields.update(json.loads(line).keys())
    return count, sorted(fields)


def _raw_text_count(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as input_file:
        return sum(1 for line in input_file if line.strip())


@lru_cache(maxsize=1)
def _dataset_catalog() -> dict[str, Any]:
    """Describe processed searchable datasets and their raw source files."""
    catalog = []
    for name, path in DATASET_FILES.items():
        records = _load_datasets()[name]
        raw_name = "Apache.log" if name == "Apache_log.csv" else "payment_api_logs.jsonl"
        raw_path = LOG_ROOT / "raw" / raw_name
        if raw_name.endswith(".log"):
            raw_format = "text"
            raw_count = _raw_text_count(raw_path)
            raw_fields = []
            raw_record_format = "one log entry per line"
        else:
            raw_format = "JSONL"
            raw_count, raw_fields = _raw_jsonl_info(raw_path)
            raw_record_format = "one JSON object per line"
        catalog.append(
            {
                "name": name,
                "path": str(path.relative_to(BACKEND_ROOT)),
                "format": "CSV",
                "record_count": len(records),
                "fields": list(records[0]) if records else [],
                "used_by_tools": True,
                "raw_source": {
                    "name": raw_name,
                    "format": raw_format,
                    "record_count": raw_count,
                    "fields": raw_fields,
                    "record_format": raw_record_format,
                },
            }
        )
    return {"datasets": catalog}


@mcp.resource(
    "logs://available",
    name="available_logs",
    description="Available log datasets, fields, formats, and record counts.",
)
def available_logs() -> dict[str, Any]:
    """Describe available processed logs and their raw source datasets."""
    return _dataset_catalog()


def main() -> None:
    """Run the standalone MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()