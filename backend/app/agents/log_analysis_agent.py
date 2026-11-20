"""Log analysis using the discovered Log MCP tools."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from time import perf_counter

from app.agents.agent_utils import (
    evidence_from_record,
    evidence_gap,
    execution_metadata,
    failed_agent_update,
    log_record_reference,
    log_service_from_record,
    mapping_records,
    meaningful_keywords,
    observation_for_evidence,
    parse_timestamp,
    stable_id,
)
from app.agents.mcp_tool_access import ToolArguments, call_server_tools
from app.graph.state import InvestigationState
from app.schemas.evidence import Evidence, EvidenceSourceType
from app.schemas.findings import AgentObservation

AGENT_NAME = "log_analysis"
SOURCE_TYPE = EvidenceSourceType.APPLICATION_LOGS
_LOG_LIMIT = 50
_SAFE_LOG_TEXT = re.compile(r"[A-Za-z0-9_ .:/-]{1,100}")
_SAFE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _record_value(record: dict[str, object], name: str) -> object:
    return next(
        (value for key, value in record.items() if key.casefold() == name),
        None,
    )


def _safe_endpoint(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.split("?", 1)[0].strip()
    if not path.startswith("/") or len(path) > 160:
        return None
    segments = []
    for segment in path.split("/"):
        if re.fullmatch(r"\d{3,}|[0-9a-fA-F-]{16,}|[A-Za-z0-9_-]{24,}", segment):
            segments.append("{id}")
        elif not segment or re.fullmatch(r"[A-Za-z0-9._{}-]{1,60}", segment):
            segments.append(segment)
        else:
            return None
    return "/".join(segments)


def _log_description(
    record: dict[str, object],
    *,
    level: str | None,
    outcome: object,
    observed_at: datetime | None,
) -> str:
    parts = ["Log MCP operational record"]
    if level and _SAFE_LOG_TEXT.fullmatch(level):
        parts.append(f"level={level}")
    if isinstance(outcome, str) and _SAFE_LOG_TEXT.fullmatch(outcome):
        parts.append(f"outcome={outcome}")
    if observed_at:
        parts.append(f"timestamp={observed_at.isoformat()}")

    for field in ("event", "error_code", "gateway", "environment"):
        value = _record_value(record, field)
        if isinstance(value, str) and _SAFE_LOG_TEXT.fullmatch(value):
            parts.append(f"{field}={value}")

    endpoint = _safe_endpoint(_record_value(record, "endpoint"))
    if endpoint:
        parts.append(f"endpoint={endpoint}")

    for field in ("http_status", "latency_ms", "retry_count"):
        value = _record_value(record, field)
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float, str))
            and _SAFE_NUMBER.fullmatch(str(value))
        ):
            parts.append(f"{field}={value}")
    return "; ".join(parts)


async def _analyze_logs(state: InvestigationState) -> dict[str, object]:
    """Collect relevant log records and return a source-specific state update."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    description = state.get("incident_description", "")
    service = state.get("affected_service")
    requests: list[tuple[str, ToolArguments]] = [
        ("get_recent_logs", {"limit": _LOG_LIMIT}),
        ("get_error_logs", {"limit": _LOG_LIMIT}),
        ("filter_logs_by_level", {"level": "ERROR", "limit": _LOG_LIMIT}),
        ("filter_logs_by_level", {"level": "WARN", "limit": _LOG_LIMIT}),
    ]
    requests.extend(
        ("search_logs", {"keyword": keyword, "limit": _LOG_LIMIT})
        for keyword in meaningful_keywords(description, service)
    )
    start_time = state.get("start_time")
    end_time = state.get("end_time")
    if start_time is not None or end_time is not None:
        requests.append(
            (
                "query_logs",
                {
                    "service_name": service,
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None,
                    "limit": 1000,
                    "offset": 0,
                },
            )
        )
    batch = await call_server_tools("log", requests)

    evidence_by_id: dict[str, Evidence] = {}
    observations_by_id: dict[str, AgentObservation] = {}
    excluded_evidence_ids: set[str] = set()
    excluded_environment_ids: set[str] = set()
    unscoped_environment_ids: set[str] = set()
    query_logs_omitted = 0
    for invocation in batch.outputs:
        if invocation.tool_name == "query_logs" and isinstance(invocation.output, dict):
            records = mapping_records(invocation.output.get("records", []))
            total = invocation.output.get("total")
            if isinstance(total, int) and total > len(records):
                query_logs_omitted = total - len(records)
        else:
            records = mapping_records(invocation.output)
        for record in records:
            evidence_id = stable_id(SOURCE_TYPE.value, record)
            requested_environment = state.get("environment")
            record_environment = _record_value(record, "environment")
            if (
                isinstance(requested_environment, str)
                and requested_environment.strip()
            ):
                if isinstance(record_environment, str) and record_environment.strip():
                    if (
                        record_environment.strip().casefold()
                        != requested_environment.strip().casefold()
                    ):
                        excluded_environment_ids.add(evidence_id)
                        continue
                else:
                    unscoped_environment_ids.add(evidence_id)
            level = next(
                (
                    value
                    for key, value in record.items()
                    if key.casefold() in {"level", "severity", "log_level"}
                    and isinstance(value, str)
                ),
                None,
            )
            timestamp = next(
                (
                    value
                    for key, value in record.items()
                    if key.casefold() in {"timestamp", "time", "observed_at"}
                ),
                None,
            )
            outcome = _record_value(record, "outcome")
            observed_at = parse_timestamp(timestamp)
            if start_time is not None or end_time is not None:
                if observed_at is None or (
                    start_time is not None and observed_at < start_time
                ) or (
                    end_time is not None and observed_at > end_time
                ):
                    excluded_evidence_ids.add(evidence_id)
                    continue
            item = evidence_by_id.get(evidence_id)
            if item is None:
                item = evidence_from_record(
                    source_type=SOURCE_TYPE,
                    source_name="Log MCP",
                    description=_log_description(
                        record,
                        level=level if isinstance(level, str) else None,
                        outcome=outcome,
                        observed_at=observed_at,
                    ),
                    record=record,
                    service=log_service_from_record(record),
                    severity=level if isinstance(level, str) else None,
                    observed_at=observed_at,
                    raw_reference=log_record_reference(record),
                    evidence_id=evidence_id,
                )
                evidence_by_id[evidence_id] = item
                observation = observation_for_evidence(
                    agent_name=AGENT_NAME,
                    summary=f"Observed log record: {item.description}.",
                    evidence=[item],
                    source_type=SOURCE_TYPE,
                )
                observations_by_id[observation.observation_id] = observation

    evidence = list(evidence_by_id.values())
    observations = list(observations_by_id.values())
    errors = []
    warnings = []
    if excluded_evidence_ids:
        warnings.append(
            f"Excluded {len(excluded_evidence_ids)} log record(s) that were outside "
            "or could not be matched to the incident time window."
        )
    if excluded_environment_ids:
        warnings.append(
            f"Excluded {len(excluded_environment_ids)} log record(s) whose "
            "environment did not match the incident."
        )
    if unscoped_environment_ids:
        warnings.append(
            f"Retained {len(unscoped_environment_ids)} log record(s) with no "
            "environment field; their environment could not be verified."
        )
    if query_logs_omitted:
        warnings.append(
            f"The Log MCP returned only 1000 of {query_logs_omitted + 1000} "
            "matching time-scoped records."
        )
    if batch.connection_error:
        errors.append(f"{AGENT_NAME}: {batch.connection_error}")
    for tool_name in batch.unavailable_tools:
        warnings.append(f"Log MCP tool is unavailable: {tool_name}.")
    if batch.errors:
        errors.extend(
            f"Log MCP {failure.tool_name}: {failure.message}"
            for failure in batch.errors
        )
    if not batch.connected and not errors:
        errors.append(f"{AGENT_NAME} could not connect to the Log MCP server.")

    gaps = []
    if not evidence:
        description = (
            "The Log MCP server was unavailable; no log evidence was collected."
            if not batch.connected
            else "No log records matching the available retrievals were returned."
        )
        gaps.append(
            evidence_gap(
                agent_name=AGENT_NAME,
                description=description,
                source_type=SOURCE_TYPE,
                impact="Log patterns could not be assessed.",
            )
        )
        observations.append(
            observation_for_evidence(
                agent_name=AGENT_NAME,
                summary="No matching application log records were returned.",
                evidence=[],
                source_type=SOURCE_TYPE,
            )
        )

    if errors:
        status = "partial_success" if batch.outputs and batch.connected else "failed"
    elif evidence:
        status = "partial_success" if warnings else "success"
    else:
        status = "no_evidence"

    return {
        "log_evidence": evidence,
        "observations": observations,
        "evidence_gaps": gaps,
        "errors": errors,
        "warnings": warnings,
        "execution_metadata": execution_metadata(
            agent_name=AGENT_NAME,
            started_at=started_at,
            started_clock=started_clock,
            status=status,
            evidence_count=len(evidence),
            errors=len(errors),
            warnings=len(warnings),
        ),
    }


async def analyze_logs(state: InvestigationState) -> dict[str, object]:
    """Run log analysis without allowing an unexpected branch error to escape."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    try:
        return await _analyze_logs(state)
    except Exception as error:  # noqa: BLE001 - isolate malformed MCP payloads.
        return failed_agent_update(
            agent_name=AGENT_NAME,
            source_type=SOURCE_TYPE,
            evidence_field="log_evidence",
            error=error,
            started_at=started_at,
            started_clock=started_clock,
        )
