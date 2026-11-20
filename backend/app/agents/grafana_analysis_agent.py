"""Grafana dashboard and alert analysis using discovered MCP tools."""

from __future__ import annotations

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

AGENT_NAME = "grafana_analysis"
SOURCE_TYPE = EvidenceSourceType.GRAFANA
_MAX_DASHBOARDS = 5


def _dashboard_search(
    results: tuple[ToolInvocation, ...],
    search_tool: str,
) -> list[tuple[str, ToolArguments]]:
    dashboard_uids = []
    seen: set[str] = set()
    for invocation in results:
        if invocation.tool_name != search_tool:
            continue
        for dashboard in mapping_records(invocation.output):
            uid = dashboard.get("uid")
            if (
                isinstance(uid, str)
                and uid
                and uid not in seen
                and len(dashboard_uids) < _MAX_DASHBOARDS
            ):
                seen.add(uid)
                dashboard_uids.append(uid)
    return [("get_dashboard", {"uid": uid}) for uid in dashboard_uids]


def _panel_context(
    dashboard: dict[str, JsonValue],
    uid: str,
) -> list[dict[str, JsonValue]]:
    panels = dashboard.get("panels")
    if not isinstance(panels, list):
        return []
    references = []
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        panel_id = panel.get("id")
        reference: dict[str, JsonValue] = {
            "reference": (
                f"grafana://dashboard/{uid}/panel/{panel_id}"
                if isinstance(panel_id, (str, int))
                else f"grafana://dashboard/{uid}/panel/unknown"
            )
        }
        for key in ("id", "title", "type", "description", "targets"):
            if key in panel:
                reference[key] = panel[key]
        references.append(reference)
    return references


async def _analyze_grafana(state: InvestigationState) -> dict[str, object]:
    """Search dashboards, retrieve their panel configuration, and collect alerts."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    service = state.get("affected_service")
    search_tool = "search_dashboards" if service else "list_dashboards"
    search_arguments: ToolArguments = {"query": service} if service else {}
    requests: list[tuple[str, ToolArguments]] = [
        (search_tool, search_arguments),
        ("list_alert_rules", {}),
        ("get_alert_instances", {}),
    ]
    batch = await call_server_tools(
        "grafana",
        requests,
        follow_up=lambda outputs: _dashboard_search(outputs, search_tool),
    )

    dashboard_search_results: list[dict[str, JsonValue]] = []
    dashboard_details: dict[str, dict[str, JsonValue]] = {}
    alert_records: list[tuple[str, dict[str, JsonValue]]] = []
    for invocation in batch.outputs:
        if invocation.tool_name == search_tool:
            dashboard_search_results.extend(mapping_records(invocation.output))
        elif invocation.tool_name == "get_dashboard":
            details = mapping_records(invocation.output)
            for record in details:
                uid = record.get("uid")
                if isinstance(uid, str):
                    dashboard_details[uid] = record
        elif invocation.tool_name in {"list_alert_rules", "get_alert_instances"}:
            alert_records.extend(
                (invocation.tool_name, record)
                for record in mapping_records(invocation.output)
            )

    evidence_by_id: dict[str, Evidence] = {}
    observations_by_id: dict[str, AgentObservation] = {}
    warnings: list[str] = []
    for dashboard in dashboard_search_results:
        uid = dashboard.get("uid")
        if not isinstance(uid, str) or not uid:
            warnings.append("Grafana returned a dashboard without a UID.")
            continue
        details = dashboard_details.get(uid)
        panels = _panel_context(details or {}, uid)
        title = dashboard.get("title")
        title_text = title if isinstance(title, str) and title else uid
        record: dict[str, JsonValue] = {
            "dashboard": dashboard,
            "details": details,
            "panel_references": panels,
        }
        item = evidence_from_record(
            source_type=SOURCE_TYPE,
            source_name="Grafana dashboard",
            description=(
                f"Grafana dashboard {title_text!r} (UID {uid}) was found. "
                f"{len(panels)} panel configurations were retrieved; dashboard "
                "presence does not establish that its panels contain live data."
                if details is not None
                else f"Grafana dashboard {title_text!r} (UID {uid}) was found, "
                "but its panel configuration could not be retrieved."
            ),
            record=record,
            service=service,
            raw_reference=f"grafana://dashboard/{uid}",
            evidence_id=stable_id("grafana:dashboard", record),
        )
        evidence_by_id[item.evidence_id] = item
        observation = observation_for_evidence(
            agent_name=AGENT_NAME,
            summary=item.description,
            evidence=[item],
            source_type=SOURCE_TYPE,
        )
        observations_by_id[observation.observation_id] = observation

    for tool_name, record in alert_records:
        if tool_name == "get_alert_instances":
            identifier = record.get("fingerprint") or record.get("alert_name")
            id_prefix = "grafana:alert-instance"
            description = "Grafana alert instance"
            observed_at = parse_timestamp(record.get("active_at"))
        else:
            identifier = record.get("uid") or record.get("title")
            id_prefix = "grafana:alert-rule"
            description = "Grafana alert rule"
            observed_at = None
        if not isinstance(identifier, str) or not identifier:
            identifier = stable_id(id_prefix, record)
        name = record.get("alert_name") or record.get("title")
        if isinstance(name, str) and name:
            description += f" {name!r}"
        item = evidence_from_record(
            source_type=SOURCE_TYPE,
            source_name=tool_name,
            description=f"{description}: {record}",
            record=record,
            service=service,
            severity=(
                record.get("state") if isinstance(record.get("state"), str) else None
            ),
            observed_at=observed_at,
            raw_reference=f"{id_prefix}:{identifier}",
            evidence_id=stable_id(
                id_prefix,
                {"reference": identifier, "record": record},
            ),
        )
        evidence_by_id[item.evidence_id] = item
        observation = observation_for_evidence(
            agent_name=AGENT_NAME,
            summary=item.description,
            evidence=[item],
            source_type=SOURCE_TYPE,
        )
        observations_by_id[observation.observation_id] = observation

    no_alert_instances = "get_alert_instances" in {
        item.tool_name for item in batch.outputs
    } and not any(name == "get_alert_instances" for name, _ in alert_records)
    if no_alert_instances:
        warnings.append("Grafana alert retrieval succeeded with no alert instances.")
        observation = observation_for_evidence(
            agent_name=AGENT_NAME,
            summary="Grafana alert retrieval succeeded with no alert instances.",
            evidence=[],
            source_type=SOURCE_TYPE,
        )
        observations_by_id[observation.observation_id] = observation

    errors: list[str] = []
    if batch.connection_error:
        errors.append(f"{AGENT_NAME}: {batch.connection_error}")
    if batch.errors:
        errors.extend(
            f"Grafana MCP {failure.tool_name}: {failure.message}"
            for failure in batch.errors
        )
    warnings.extend(
        f"Grafana MCP tool is unavailable: {tool_name}."
        for tool_name in batch.unavailable_tools
    )
    evidence = list(evidence_by_id.values())
    observations = list(observations_by_id.values())
    gaps = []
    if not evidence:
        description = (
            "The Grafana MCP server was unavailable; dashboards and alerts are unknown."
            if not batch.connected
            else "Grafana returned no matching dashboards, alert rules, or alert instances."
        )
        gaps.append(
            evidence_gap(
                agent_name=AGENT_NAME,
                description=description,
                source_type=SOURCE_TYPE,
                impact="Grafana dashboard and alert context could not be assessed.",
            )
        )
        if not observations:
            observations.append(
                observation_for_evidence(
                    agent_name=AGENT_NAME,
                    summary="No Grafana dashboard or alert records were returned.",
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
        "grafana_evidence": evidence,
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


async def analyze_grafana(state: InvestigationState) -> dict[str, object]:
    """Run Grafana analysis without propagating external source failures."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    try:
        return await _analyze_grafana(state)
    except Exception as error:  # noqa: BLE001 - isolate API/data conversion failures.
        return failed_agent_update(
            agent_name=AGENT_NAME,
            source_type=SOURCE_TYPE,
            evidence_field="grafana_evidence",
            error=error,
            started_at=started_at,
            started_clock=started_clock,
        )
