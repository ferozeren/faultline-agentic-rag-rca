"""Service metrics analysis using the discovered Metrics MCP tools."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from time import perf_counter

from pydantic import JsonValue

from app.agents.agent_utils import (
    evidence_from_record,
    evidence_gap,
    execution_metadata,
    failed_agent_update,
    mapping_records,
    observation_for_evidence,
    parse_timestamp,
    service_from_record,
    stable_id,
)
from app.agents.mcp_tool_access import ToolArguments, call_server_tools
from app.graph.state import InvestigationState
from app.schemas.evidence import Evidence, EvidenceSourceType
from app.schemas.findings import AgentObservation

AGENT_NAME = "metrics_analysis"
SOURCE_TYPE = EvidenceSourceType.SERVICE_METRICS
_METRIC_FIELDS = (
    "cpu_usage_percent",
    "memory_usage_percent",
    "avg_latency_ms",
    "request_count",
    "error_count",
    "db_connections",
)
_LIMIT = 500


def _zero_value(value: JsonValue) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return float(value) == 0
    except (TypeError, ValueError):
        return False


async def _analyze_service_metrics(state: InvestigationState) -> dict[str, object]:
    """Collect service metrics, preserving missing and zero values distinctly."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    service = state.get("affected_service")
    start_time = state.get("start_time")
    end_time = state.get("end_time")
    if service:
        requests: list[tuple[str, ToolArguments]] = [
            ("get_metrics_by_service", {"service_name": service, "limit": _LIMIT})
        ]
        if start_time is None and end_time is None:
            requests.append(("get_service_summary", {"service_name": service}))
    else:
        requests = [("get_latest_metrics", {"limit": _LIMIT})]
    batch = await call_server_tools("metrics", requests)

    evidence_by_id: dict[str, Evidence] = {}
    observations_by_id: dict[str, AgentObservation] = {}
    warnings: list[str] = []
    excluded_evidence_ids: set[str] = set()
    excluded_environment_ids: set[str] = set()
    unscoped_environment_ids: set[str] = set()
    for invocation in batch.outputs:
        for record in mapping_records(invocation.output):
            if invocation.tool_name == "get_service_summary":
                record_count = record.get("record_count")
                if record_count == 0 or record_count == "0":
                    continue
            evidence_id = stable_id(SOURCE_TYPE.value, record)
            requested_environment = state.get("environment")
            record_environment = next(
                (
                    value
                    for key, value in record.items()
                    if key.casefold() == "environment"
                ),
                None,
            )
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
            timestamp = record.get("timestamp")
            observed_at = parse_timestamp(timestamp)
            if start_time is not None or end_time is not None:
                if observed_at is None or (
                    start_time is not None and observed_at < start_time
                ) or (
                    end_time is not None and observed_at > end_time
                ):
                    excluded_evidence_ids.add(evidence_id)
                    continue
            record_service = service_from_record(record, service)
            missing_fields = [
                field
                for field in _METRIC_FIELDS
                if field not in record or record[field] is None or record[field] == ""
            ]
            zero_fields = [
                field for field in _METRIC_FIELDS if _zero_value(record.get(field))
            ]
            if missing_fields:
                warnings.append(
                    "Metric data omitted fields: " + ", ".join(missing_fields) + "."
                )
            description_parts = [
                "Metrics MCP result",
                f"service={record_service or 'not specified'}",
            ]
            if isinstance(timestamp, str) and timestamp:
                description_parts.append(f"timestamp={timestamp}")
            if isinstance(record_environment, str) and record_environment.strip():
                description_parts.append(f"environment={record_environment.strip()}")
            elif (
                isinstance(requested_environment, str)
                and requested_environment.strip()
            ):
                description_parts.append("environment=not specified")
            present_fields = [
                f"{field}={record[field]}"
                for field in _METRIC_FIELDS
                if field in record and record[field] is not None and record[field] != ""
            ]
            if present_fields:
                description_parts.extend(present_fields)
            if missing_fields:
                description_parts.append("missing fields=" + ", ".join(missing_fields))
            if zero_fields:
                description_parts.append(
                    "zero-valued metrics=" + ", ".join(zero_fields)
                )
            derived_error_rate = None
            request_count = record.get("request_count")
            error_count = record.get("error_count")
            if request_count is not None and error_count is not None:
                try:
                    requests_value = float(request_count)
                    errors_value = float(error_count)
                    if (
                        math.isfinite(requests_value)
                        and math.isfinite(errors_value)
                        and requests_value > 0
                        and errors_value >= 0
                    ):
                        derived_error_rate = errors_value / requests_value * 100
                except (TypeError, ValueError):
                    warnings.append(
                        "Could not derive an error rate because returned request "
                        "or error counts were not numeric."
                    )
            if derived_error_rate is not None:
                description_parts.append(
                    f"derived error rate={derived_error_rate:.2f}% "
                    "(error_count/request_count)"
                )

            item = evidence_by_id.get(evidence_id)
            if item is None:
                item = evidence_from_record(
                    source_type=SOURCE_TYPE,
                    source_name="Metrics MCP",
                    description="; ".join(description_parts),
                    record=record,
                    service=record_service,
                    observed_at=observed_at,
                    raw_reference=(
                        f"{record_service}:{timestamp}"
                        if isinstance(record_service, str)
                        and isinstance(timestamp, str)
                        else None
                    ),
                    evidence_id=evidence_id,
                )
                evidence_by_id[evidence_id] = item
                observations_by_id[item.evidence_id] = observation_for_evidence(
                    agent_name=AGENT_NAME,
                    summary=(
                        f"Observed operational metrics: {item.description}. "
                        "A missing field is unavailable data; a returned zero is "
                        "a measured zero."
                    ),
                    evidence=[item],
                    source_type=SOURCE_TYPE,
                )

    evidence = list(evidence_by_id.values())
    observations = list(observations_by_id.values())
    if excluded_evidence_ids:
        warnings.append(
            f"Excluded {len(excluded_evidence_ids)} metric record(s) outside or "
            "without timestamps matching the incident time window."
        )
    if excluded_environment_ids:
        warnings.append(
            f"Excluded {len(excluded_environment_ids)} metric record(s) whose "
            "environment did not match the incident."
        )
    if unscoped_environment_ids:
        warnings.append(
            f"Retained {len(unscoped_environment_ids)} metric record(s) with no "
            "environment field; their environment could not be verified."
        )
    errors: list[str] = []
    if batch.connection_error:
        errors.append(f"{AGENT_NAME}: {batch.connection_error}")
    if batch.errors:
        errors.extend(
            f"Metrics MCP {failure.tool_name}: {failure.message}"
            for failure in batch.errors
        )
    warnings.extend(
        f"Metrics MCP tool is unavailable: {tool_name}."
        for tool_name in batch.unavailable_tools
    )
    gaps = []
    if not evidence:
        description = (
            "The Metrics MCP server was unavailable; metric values are unknown."
            if not batch.connected
            else "No service metric records were returned; unavailable values are not zero."
        )
        gaps.append(
            evidence_gap(
                agent_name=AGENT_NAME,
                description=description,
                source_type=SOURCE_TYPE,
                impact="CPU, memory, latency, error counts, and database connections could not be assessed.",
            )
        )
        observations.append(
            observation_for_evidence(
                agent_name=AGENT_NAME,
                summary=(
                    "No service metric records were returned. Missing data must not "
                    "be interpreted as zero."
                ),
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
        "metrics_evidence": evidence,
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


async def analyze_service_metrics(
    state: InvestigationState,
) -> dict[str, object]:
    """Run metrics analysis without propagating source/data conversion failures."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    try:
        return await _analyze_service_metrics(state)
    except Exception as error:  # noqa: BLE001 - isolate malformed MCP payloads.
        return failed_agent_update(
            agent_name=AGENT_NAME,
            source_type=SOURCE_TYPE,
            evidence_field="metrics_evidence",
            error=error,
            started_at=started_at,
            started_clock=started_clock,
        )
