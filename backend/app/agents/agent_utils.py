"""Shared evidence, observation, and execution metadata helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from pydantic import JsonValue

from app.schemas.evidence import Evidence, EvidenceClassification, EvidenceSourceType
from app.schemas.findings import AgentObservation, EvidenceGap

EVALUATION_FIELDS = frozenset({"scenario_label", "is_anomaly"})


def configured_nvidia_model(*, error_message: str) -> Any:
    """Create the configured NVIDIA NIM client without embedding credentials."""
    from app.rag.test_nvidia_api import NVIDIA_ENDPOINT, _read_setting

    model_name = _read_setting("NVIDIA_MODEL")
    api_key = _read_setting("NVIDIA_API_KEY")
    if not model_name or not api_key:
        raise RuntimeError(error_message)

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(
        model=model_name,
        api_key=api_key,
        base_url=NVIDIA_ENDPOINT,
        temperature=0.1,
        max_completion_tokens=2048,
    )


def remove_evaluation_fields(value: JsonValue) -> JsonValue:
    """Recursively exclude offline ground-truth labels from operational payloads."""
    if isinstance(value, dict):
        return {
            key: remove_evaluation_fields(item)
            for key, item in value.items()
            if key.casefold() not in EVALUATION_FIELDS
        }
    if isinstance(value, list):
        return [remove_evaluation_fields(item) for item in value]
    return value


def mapping_records(value: JsonValue) -> list[dict[str, JsonValue]]:
    """Extract record mappings from standard MCP list or result envelopes."""
    if isinstance(value, dict):
        result = value.get("result")
        if result is not None:
            return mapping_records(result)
        data = value.get("data")
        if isinstance(data, dict) and "result" in data:
            return mapping_records(data["result"])
        cleaned = remove_evaluation_fields(value)
        return [cleaned] if isinstance(cleaned, dict) else []
    if isinstance(value, list):
        records = []
        for item in value:
            if isinstance(item, dict):
                cleaned = remove_evaluation_fields(item)
                if isinstance(cleaned, dict):
                    records.append(cleaned)
        return records
    return []


def stable_id(namespace: str, value: JsonValue) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return f"{namespace}:{digest}"


def evidence_from_record(
    *,
    source_type: EvidenceSourceType,
    source_name: str,
    description: str,
    record: dict[str, JsonValue],
    service: str | None = None,
    severity: str | None = None,
    observed_at: datetime | None = None,
    raw_reference: str | None = None,
    evidence_id: str | None = None,
) -> Evidence:
    classification = (
        EvidenceClassification.HISTORICAL
        if source_type is EvidenceSourceType.HISTORICAL_RAG
        else EvidenceClassification.CURRENT
    )
    reference = raw_reference or record.get("chunk_id")
    return Evidence(
        evidence_id=evidence_id or stable_id(source_type.value, record),
        source_type=source_type,
        source_name=source_name,
        description=description,
        observed_at=observed_at,
        service=service,
        severity=severity,
        raw_reference=reference if isinstance(reference, str) else None,
        metadata=record,
        classification=classification,
    )


def evidence_for_model(evidence: Evidence) -> dict[str, JsonValue]:
    """Serialize evidence without raw metadata or source-specific identifiers."""
    return evidence.model_dump(
        mode="json",
        exclude={"metadata", "raw_reference"},
    )


def observation_for_evidence(
    *,
    agent_name: str,
    summary: str,
    evidence: Iterable[Evidence],
    source_type: EvidenceSourceType,
) -> AgentObservation:
    evidence_ids = sorted({item.evidence_id for item in evidence})
    classification = (
        EvidenceClassification.HISTORICAL
        if source_type is EvidenceSourceType.HISTORICAL_RAG
        else EvidenceClassification.CURRENT
    )
    return AgentObservation(
        observation_id=stable_id(
            f"observation:{agent_name}",
            {"evidence_ids": evidence_ids, "summary": summary},
        ),
        agent_name=agent_name,
        summary=summary,
        evidence_ids=evidence_ids,
        source_types=[source_type],
        classification=classification,
    )


def evidence_gap(
    *,
    agent_name: str,
    description: str,
    source_type: EvidenceSourceType,
    impact: str | None = None,
) -> EvidenceGap:
    return EvidenceGap(
        gap_id=stable_id(
            f"gap:{agent_name}",
            {"description": description, "source_type": source_type.value},
        ),
        description=description,
        missing_source_types=[source_type],
        impact=impact,
        recommended_data_sources=[source_type.value],
    )


def parse_timestamp(value: JsonValue) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(normalized, "%a %b %d %H:%M:%S %Y")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def log_service_from_record(record: dict[str, JsonValue]) -> str | None:
    """Return only explicit or dataset-backed log service identity."""
    for key in ("service", "service_name", "serviceName"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if record.get("dataset") == "payment_api_logs.csv":
        return "payment-api"
    return None


def service_from_record(
    record: dict[str, JsonValue],
    requested_service: str | None,
) -> str | None:
    for key in ("service", "service_name", "serviceName"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return requested_service


def concise_error(agent_name: str, error: Exception) -> str:
    """Return exception type only; exception text may contain credentials."""
    return f"{agent_name} failed ({type(error).__name__})."


def failed_agent_update(
    *,
    agent_name: str,
    source_type: EvidenceSourceType,
    evidence_field: str,
    error: Exception,
    historical: bool = False,
    started_at: datetime | None = None,
    started_clock: float | None = None,
) -> dict[str, object]:
    """Create an explicit failed-branch update without propagating plugin errors."""
    started_at = started_at or datetime.now(UTC)
    started_clock = started_clock if started_clock is not None else perf_counter()
    message = concise_error(agent_name, error)
    observation = observation_for_evidence(
        agent_name=agent_name,
        summary=(
            "Historical retrieval failed; no current root-cause conclusion can be "
            "drawn from prior incidents."
            if historical
            else f"{agent_name} could not complete its analysis; no source conclusion "
            "was reached."
        ),
        evidence=[],
        source_type=source_type,
    )
    update: dict[str, object] = {
        evidence_field: [],
        "evidence_gaps": [
            evidence_gap(
                agent_name=agent_name,
                description=f"{agent_name} failed; no evidence was collected.",
                source_type=source_type,
                impact="This source could not contribute to the investigation.",
            )
        ],
        "errors": [message],
        "warnings": [],
        "execution_metadata": execution_metadata(
            agent_name=agent_name,
            started_at=started_at,
            started_clock=started_clock,
            status="failed",
            evidence_count=0,
            errors=1,
            warnings=0,
        ),
    }
    update["rag_observations" if historical else "observations"] = [observation]
    return update


def execution_metadata(
    *,
    agent_name: str,
    started_at: datetime,
    started_clock: float,
    status: str,
    evidence_count: int,
    errors: int,
    warnings: int,
) -> dict[str, JsonValue]:
    completed_at = datetime.now(UTC)
    run_key = f"{agent_name}:{started_at.isoformat()}"
    return {
        run_key: {
            "agent_name": agent_name,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round(perf_counter() - started_clock, 6),
            "status": status,
            "evidence_count": evidence_count,
            "error_count": errors,
            "warning_count": warnings,
        }
    }


def log_record_reference(record: dict[str, JsonValue]) -> str | None:
    dataset = record.get("dataset")
    identifier = next(
        (
            record.get(field)
            for field in ("trace_id", "request_id", "id", "timestamp")
            if isinstance(record.get(field), str) and record[field]
        ),
        None,
    )
    if isinstance(dataset, str) and isinstance(identifier, str):
        return f"{dataset}:{identifier}"
    return None


def meaningful_keywords(description: str, service: str | None) -> list[str]:
    candidates = [service or ""]
    candidates.extend(re.findall(r"[A-Za-z0-9_.-]{3,}", description))
    result: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = item.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
        if len(result) == 4:
            break
    return result
