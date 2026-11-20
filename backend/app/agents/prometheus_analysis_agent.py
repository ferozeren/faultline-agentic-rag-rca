"""Prometheus query and alert analysis using discovered MCP tools."""

from __future__ import annotations

import json
import re
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
    stable_id,
)
from app.agents.mcp_tool_access import (
    ToolArguments,
    ToolInvocation,
    call_server_tools,
)
from app.graph.state import InvestigationState
from app.schemas.evidence import Evidence, EvidenceSourceType
from app.schemas.findings import AgentObservation

AGENT_NAME = "prometheus_analysis"
SOURCE_TYPE = EvidenceSourceType.PROMETHEUS


def _prometheus_result(
    payload: JsonValue,
) -> tuple[str | None, str | None, list[dict[str, JsonValue]]]:
    if not isinstance(payload, dict):
        return None, None, []
    status = payload.get("status")
    data = payload.get("data")
    if isinstance(data, dict):
        result_type = data.get("resultType")
        results = data.get("result")
    else:
        result_type = payload.get("resultType")
        results = payload.get("result")
    if not isinstance(result_type, str) or not isinstance(results, list):
        return (
            status if isinstance(status, str) else None,
            result_type if isinstance(result_type, str) else None,
            [],
        )
    return (
        status if isinstance(status, str) else None,
        result_type,
        [item for item in results if isinstance(item, dict)],
    )


def _series_observed_at(series: dict[str, JsonValue]) -> datetime | None:
    for field in ("value", "values"):
        value = series.get(field)
        samples = value if field == "values" and isinstance(value, list) else [value]
        for sample in samples:
            if isinstance(sample, list) and sample:
                try:
                    return datetime.fromtimestamp(float(sample[0]), tz=UTC)
                except (TypeError, ValueError, OverflowError, OSError):
                    continue
    return None


def _service_label(
    series: dict[str, JsonValue],
    requested_service: str | None,
) -> str | None:
    labels = series.get("metric")
    if isinstance(labels, dict):
        for value in labels.values():
            if (
                isinstance(value, str)
                and requested_service
                and value.casefold() == requested_service.casefold()
            ):
                return requested_service
        for key in ("service", "service_name", "serviceName"):
            value = labels.get(key)
            if isinstance(value, str):
                return value
    return None


def _query_plan(
    results: tuple[ToolInvocation, ...],
    *,
    service: str | None,
    start: datetime | None,
    end: datetime | None,
) -> tuple[str, list[tuple[str, ToolArguments]]]:
    """Use only target labels observed in Prometheus to scope the known `up` query."""
    selector = None
    if service:
        for invocation in results:
            if invocation.tool_name != "get_prometheus_targets":
                continue
            for target in mapping_records(invocation.output):
                labels = target.get("labels")
                if not isinstance(labels, dict):
                    continue
                matching_labels = sorted(
                    key
                    for key, value in labels.items()
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                    and isinstance(value, str)
                    and value.casefold() == service.casefold()
                )
                if matching_labels:
                    selector = f"{{{matching_labels[0]}={json.dumps(service)}}}"
                    break
            if selector:
                break

    query = "up" + (selector or "")
    if start is not None and end is not None:
        return query, [
            (
                "query_prometheus_range",
                {
                    "query": query,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "step": "60s",
                },
            )
        ]
    return query, [("query_prometheus", {"query": query})]


async def _analyze_prometheus(state: InvestigationState) -> dict[str, object]:
    """Query the known Prometheus `up` metric and collect targets and alerts."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    start = state.get("start_time")
    end = state.get("end_time")
    query = "up"
    service = state.get("affected_service")

    def follow_up(
        outputs: tuple[ToolInvocation, ...],
    ) -> list[tuple[str, ToolArguments]]:
        nonlocal query
        query, plan = _query_plan(
            outputs,
            service=service,
            start=start,
            end=end,
        )
        return plan

    batch = await call_server_tools(
        "prometheus",
        [
            ("get_prometheus_targets", {}),
            ("get_prometheus_alerts", {}),
        ],
        follow_up=follow_up,
    )

    evidence: list[Evidence] = []
    observations: list[AgentObservation] = []
    warnings: list[str] = []
    errors: list[str] = []
    data_record_count = 0
    for invocation in batch.outputs:
        if invocation.tool_name in {"query_prometheus", "query_prometheus_range"}:
            status, result_type, series_results = _prometheus_result(invocation.output)
            if status is not None and status != "success":
                errors.append(
                    "Prometheus returned an unsuccessful query status; the query "
                    "was not interpreted as an empty result."
                )
                continue
            if result_type is None:
                errors.append(
                    "Prometheus query response did not contain a result type and "
                    "result list."
                )
                continue
            data_record_count += len(series_results)
            if not series_results:
                empty_result: dict[str, JsonValue] = {
                    "query": query,
                    "result_type": result_type,
                    "result_count": 0,
                    "status": status or "success",
                    "incident_service": state.get("affected_service"),
                }
                item = evidence_from_record(
                    source_type=SOURCE_TYPE,
                    source_name=invocation.tool_name,
                    description=(
                        f"Prometheus query {query!r} completed successfully with "
                        "zero returned series; this does not prove zero errors."
                    ),
                    record=empty_result,
                    service=None,
                    raw_reference=query,
                    evidence_id=stable_id(SOURCE_TYPE.value, empty_result),
                )
                evidence.append(item)
                observations.append(
                    observation_for_evidence(
                        agent_name=AGENT_NAME,
                        summary=(
                            f"Prometheus query {query!r} returned zero series. "
                            "An empty query result is not proof of zero errors."
                        ),
                        evidence=[item],
                        source_type=SOURCE_TYPE,
                    )
                )
            for series in series_results:
                record: dict[str, JsonValue] = {
                    "query": query,
                    "result_type": result_type,
                    "series": series,
                }
                matched_service = _service_label(
                    series,
                    state.get("affected_service"),
                )
                item = evidence_from_record(
                    source_type=SOURCE_TYPE,
                    source_name=invocation.tool_name,
                    description=(
                        f"Prometheus returned a series for query {query!r}."
                        + (
                            f" A returned label matches service {matched_service!r}."
                            if matched_service
                            else " The returned series was not confirmed as belonging "
                            "to the affected service."
                        )
                    ),
                    record=record,
                    service=matched_service,
                    observed_at=_series_observed_at(series),
                    raw_reference=query,
                )
                evidence.append(item)
                observations.append(
                    observation_for_evidence(
                        agent_name=AGENT_NAME,
                        summary=item.description,
                        evidence=[item],
                        source_type=SOURCE_TYPE,
                    )
                )
            continue

        records = mapping_records(invocation.output)
        if invocation.tool_name == "get_prometheus_alerts" and not records:
            warnings.append("Prometheus returned no alert records.")
            observations.append(
                observation_for_evidence(
                    agent_name=AGENT_NAME,
                    summary="Prometheus alert retrieval succeeded with no alert records.",
                    evidence=[],
                    source_type=SOURCE_TYPE,
                )
            )
        for record in records:
            data_record_count += 1
            labels = record.get("labels")
            service = _service_label(
                {"metric": labels} if isinstance(labels, dict) else record,
                state.get("affected_service"),
            )
            timestamp_value = record.get("lastScrape") or record.get("active_at")
            item = evidence_from_record(
                source_type=SOURCE_TYPE,
                source_name=invocation.tool_name,
                description=f"Prometheus {invocation.tool_name} record: {record}",
                record=record,
                service=service,
                severity=(
                    record.get("state")
                    if isinstance(record.get("state"), str)
                    else None
                ),
                observed_at=parse_timestamp(timestamp_value),
                raw_reference=(
                    record.get("alert_name")
                    if isinstance(record.get("alert_name"), str)
                    else invocation.tool_name
                ),
            )
            evidence.append(item)
            observations.append(
                observation_for_evidence(
                    agent_name=AGENT_NAME,
                    summary=item.description,
                    evidence=[item],
                    source_type=SOURCE_TYPE,
                )
            )

    if batch.connection_error:
        errors.append(f"{AGENT_NAME}: {batch.connection_error}")
    if batch.errors:
        errors.extend(
            f"Prometheus MCP {failure.tool_name}: {failure.message}"
            for failure in batch.errors
        )
    warnings.extend(
        f"Prometheus MCP tool is unavailable: {tool_name}."
        for tool_name in batch.unavailable_tools
    )
    gaps = []
    if data_record_count == 0:
        description = (
            "The Prometheus MCP server was unavailable; query and alert state are unknown."
            if not batch.connected
            else "Prometheus returned no target series or alert records."
        )
        gaps.append(
            evidence_gap(
                agent_name=AGENT_NAME,
                description=description,
                source_type=SOURCE_TYPE,
                impact="Prometheus did not provide operational series or alert records.",
            )
        )
        if not observations:
            observations.append(
                observation_for_evidence(
                    agent_name=AGENT_NAME,
                    summary=(
                        "No Prometheus operational records were available; an empty "
                        "result is not evidence that error rates are zero."
                    ),
                    evidence=[],
                    source_type=SOURCE_TYPE,
                )
            )

    if errors:
        status = "partial_success" if batch.outputs and batch.connected else "failed"
    elif data_record_count:
        status = "partial_success" if warnings else "success"
    else:
        status = "no_evidence"

    return {
        "prometheus_evidence": evidence,
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


async def analyze_prometheus(state: InvestigationState) -> dict[str, object]:
    """Run Prometheus analysis with failures contained to this specialist branch."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    try:
        return await _analyze_prometheus(state)
    except Exception as error:  # noqa: BLE001 - isolate source/query parsing failures.
        return failed_agent_update(
            agent_name=AGENT_NAME,
            source_type=SOURCE_TYPE,
            evidence_field="prometheus_evidence",
            error=error,
            started_at=started_at,
            started_clock=started_clock,
        )
