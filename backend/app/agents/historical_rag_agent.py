"""Historical incident retrieval using the existing RAG answer chain."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from pydantic import JsonValue, TypeAdapter

from app.agents.agent_utils import (
    concise_error,
    evidence_from_record,
    evidence_gap,
    execution_metadata,
    failed_agent_update,
    observation_for_evidence,
)
from app.graph.state import InvestigationState
from app.rag import rag_chain
from app.schemas.evidence import Evidence, EvidenceSourceType
from app.schemas.findings import AgentObservation

_JSON_VALUE = TypeAdapter(JsonValue)


def _analyze_historical_incidents(
    state: InvestigationState,
    *,
    llm: Any | None = None,
) -> dict[str, object]:
    """Return historical evidence, source matches, and a grounded observation."""
    agent_name = "historical_rag"
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    query = state.get("incident_description", "").strip()
    service = state.get("affected_service")
    if service:
        query = f"{query}\nAffected service: {service}"

    evidence: list[Evidence] = []
    observations: list[AgentObservation] = []
    matches: list[dict[str, JsonValue]] = []
    errors: list[str] = []
    warnings: list[str] = []
    gaps = []

    try:
        if not query.strip():
            raise ValueError("incident description is empty.")
        result = rag_chain.answer_question(query, llm=llm)
        retrieved = result.get("retrieved_documents", [])
        if not isinstance(retrieved, list):
            raise TypeError("RAG returned an invalid retrieved_documents value.")

        for document in retrieved:
            if not isinstance(document, dict):
                warnings.append("RAG returned a document with an invalid shape.")
                continue
            clean_document = _JSON_VALUE.validate_python(document)
            if not isinstance(clean_document, dict):
                warnings.append("RAG returned a document with an invalid shape.")
                continue
            chunk_id = clean_document.get("chunk_id")
            text = clean_document.get("text")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                warnings.append("A retrieved historical document had no chunk ID.")
                continue
            if not isinstance(text, str) or not text.strip():
                warnings.append(f"Historical chunk {chunk_id!r} had no text.")
                continue

            metadata = clean_document.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            incident_id = metadata.get("incident_id")
            description = f"Historical incident chunk {chunk_id}: {text}"
            if isinstance(incident_id, str):
                description = (
                    f"Historical incident {incident_id}, chunk {chunk_id}: {text}"
                )
            doc_service = metadata.get("service") or metadata.get("service_name")
            evidence.append(
                evidence_from_record(
                    source_type=EvidenceSourceType.HISTORICAL_RAG,
                    source_name="historical incident RAG",
                    description=description,
                    record=clean_document,
                    service=doc_service if isinstance(doc_service, str) else service,
                    raw_reference=chunk_id,
                    evidence_id=f"historical_rag:{chunk_id}",
                )
            )
            matches.append(clean_document)

        answer = result.get("answer", "")
        answer_text = answer if isinstance(answer, str) else str(answer)
        if evidence:
            summary = (
                "Historical reference only; it is not proof of the current root "
                f"cause. RAG response: {answer_text}"
            )
        else:
            summary = (
                "No historical incident documents were retrieved; historical "
                "evidence cannot establish the current incident's cause."
            )
            gaps.append(
                evidence_gap(
                    agent_name=agent_name,
                    description="No historical incident documents were retrieved.",
                    source_type=EvidenceSourceType.HISTORICAL_RAG,
                    impact="Historical comparison is unavailable.",
                )
            )
        observations.append(
            observation_for_evidence(
                agent_name=agent_name,
                summary=summary,
                evidence=evidence,
                source_type=EvidenceSourceType.HISTORICAL_RAG,
            )
        )
        status = "success" if evidence else "no_evidence"
    except Exception as error:  # noqa: BLE001 - isolate retrieval/model failures to this branch.
        errors.append(concise_error(agent_name, error))
        gaps.append(
            evidence_gap(
                agent_name=agent_name,
                description="Historical retrieval failed; no historical evidence was collected.",
                source_type=EvidenceSourceType.HISTORICAL_RAG,
                impact="Historical comparison is unavailable.",
            )
        )
        observations.append(
            observation_for_evidence(
                agent_name=agent_name,
                summary=(
                    "Historical retrieval failed. No conclusion about the current "
                    "incident can be drawn from historical evidence."
                ),
                evidence=[],
                source_type=EvidenceSourceType.HISTORICAL_RAG,
            )
        )
        status = "partial_success" if evidence else "failed"

    metadata = execution_metadata(
        agent_name=agent_name,
        started_at=started_at,
        started_clock=started_clock,
        status=status,
        evidence_count=len(evidence),
        errors=len(errors),
        warnings=len(warnings),
    )
    return {
        "historical_evidence": evidence,
        "historical_incident_matches": matches,
        "rag_observations": observations,
        "evidence_gaps": gaps,
        "errors": errors,
        "warnings": warnings,
        "execution_metadata": metadata,
    }


def analyze_historical_incidents(
    state: InvestigationState,
    *,
    llm: Any | None = None,
) -> dict[str, object]:
    """Run historical retrieval and contain unexpected specialist failures."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    try:
        return _analyze_historical_incidents(state, llm=llm)
    except Exception as error:  # noqa: BLE001 - isolate retrieval/state failures.
        return failed_agent_update(
            agent_name="historical_rag",
            source_type=EvidenceSourceType.HISTORICAL_RAG,
            evidence_field="historical_evidence",
            error=error,
            historical=True,
            started_at=started_at,
            started_clock=started_clock,
        )
