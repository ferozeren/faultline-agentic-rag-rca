"""Validation tests for evidence-referenced RCA output models."""

from datetime import UTC, datetime, timedelta

import pytest
from app.schemas import (
    Evidence,
    EvidenceClassification,
    EvidenceSourceType,
    RCAEvidenceClaim,
    RCAResult,
    RCAStatus,
    RCATimelineEvent,
)
from app.schemas.rca import RCAEventType
from pydantic import ValidationError

_START = datetime(2026, 11, 6, 10, 0, tzinfo=UTC)


def _evidence(
    evidence_id="logs:1",
    *,
    source_type=EvidenceSourceType.APPLICATION_LOGS,
    classification=EvidenceClassification.CURRENT,
    observed_at=_START,
    service="payment-api",
):
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        source_name=source_type.value,
        description="Database connection errors increased.",
        observed_at=observed_at,
        service=service,
        classification=classification,
    )


def _context(*evidence):
    return {
        "evidence_by_id": {item.evidence_id: item for item in evidence},
        "incident_service": "payment-api",
    }


def _payload(**updates):
    payload = {
        "status": RCAStatus.ANALYZED,
        "incident_summary": "Payment API experienced connection errors.",
        "primary_finding": {
            "finding_id": "finding:pool",
            "summary": "Connection pool saturation is the probable cause.",
            "root_cause": "The connection pool became saturated.",
            "affected_services": ["payment-api"],
            "supporting_evidence_ids": ["logs:1", "metrics:1"],
            "confidence": 0.72,
            "confidence_explanation": "Two current source types corroborate the signal.",
            "causal_reasoning": "The evidence links pool waits with increased errors.",
        },
        "timeline": [
            {
                "event_id": "event:logs",
                "timestamp": _START,
                "service": "payment-api",
                "source": EvidenceSourceType.APPLICATION_LOGS,
                "description": "Database connection errors increased.",
                "evidence_ids": ["logs:1"],
                "event_type": RCAEventType.ERROR,
            }
        ],
        "verified_facts": [
            {
                "description": "Database connection errors increased.",
                "evidence_ids": ["logs:1"],
                "classification": EvidenceClassification.CURRENT,
            }
        ],
        "overall_confidence": 0.72,
        "confidence_explanation": "This is a qualitative evidence-strength estimate.",
        "analysis_summary": "Correlated current evidence supports a probable cause.",
    }
    payload.update(updates)
    return payload


def test_valid_rca_result_with_contextual_evidence():
    logs = _evidence()
    metrics = _evidence(
        "metrics:1",
        source_type=EvidenceSourceType.SERVICE_METRICS,
        observed_at=_START + timedelta(minutes=2),
    )

    result = RCAResult.model_validate(_payload(), context=_context(logs, metrics))

    assert result.status is RCAStatus.ANALYZED
    assert result.primary_finding.supporting_evidence_ids == ["logs:1", "metrics:1"]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, True])
def test_invalid_confidence_is_rejected(confidence):
    payload = _payload()
    payload["overall_confidence"] = confidence

    with pytest.raises(ValidationError, match="confidence"):
        RCAResult.model_validate(
            payload,
            context=_context(
                _evidence(),
                _evidence("metrics:1", source_type=EvidenceSourceType.SERVICE_METRICS),
            ),
        )


def test_unknown_evidence_ids_are_rejected():
    payload = _payload()
    payload["primary_finding"]["supporting_evidence_ids"] = ["logs:unknown"]

    with pytest.raises(ValidationError, match="unknown evidence IDs"):
        RCAResult.model_validate(
            payload,
            context=_context(
                _evidence(),
                _evidence("metrics:1", source_type=EvidenceSourceType.SERVICE_METRICS),
            ),
        )


def test_analyzed_result_requires_primary_root_cause():
    payload = _payload(primary_finding=None)

    with pytest.raises(ValidationError, match="require a primary root-cause"):
        RCAResult.model_validate(payload)


def test_historical_evidence_cannot_be_primary_cause_support():
    history = _evidence(
        "history:1",
        source_type=EvidenceSourceType.HISTORICAL_RAG,
        classification=EvidenceClassification.HISTORICAL,
    )
    payload = _payload()
    payload["primary_finding"]["supporting_evidence_ids"] = ["history:1"]

    with pytest.raises(ValidationError, match="current operational evidence only"):
        RCAResult.model_validate(
            payload,
            context={
                "evidence_by_id": {
                    "history:1": history,
                    "logs:1": _evidence(),
                    "metrics:1": _evidence(
                        "metrics:1",
                        source_type=EvidenceSourceType.SERVICE_METRICS,
                    ),
                },
                "incident_service": "payment-api",
            },
        )


def test_invalid_timeline_evidence_and_timestamp_are_rejected():
    logs = _evidence()
    metrics = _evidence("metrics:1", source_type=EvidenceSourceType.SERVICE_METRICS)
    payload = _payload()
    payload["timeline"][0]["evidence_ids"] = ["logs:missing"]

    with pytest.raises(ValidationError, match="unknown evidence IDs"):
        RCAResult.model_validate(payload, context=_context(logs, metrics))

    payload = _payload()
    payload["timeline"][0]["timestamp"] = _START + timedelta(hours=4)
    with pytest.raises(ValidationError, match="timestamps must come"):
        RCAResult.model_validate(payload, context=_context(logs, metrics))


def test_claim_classification_must_match_referenced_evidence():
    logs = _evidence()
    payload = _payload(
        verified_facts=[
            RCAEvidenceClaim(
                description="Prior incident showed the same symptom.",
                evidence_ids=["logs:1"],
                classification=EvidenceClassification.HISTORICAL,
            ).model_dump()
        ]
    )

    with pytest.raises(ValidationError, match="Claim classification"):
        RCAResult.model_validate(
            payload,
            context=_context(
                logs,
                _evidence("metrics:1", source_type=EvidenceSourceType.SERVICE_METRICS),
            ),
        )


def test_timeline_source_and_service_must_match_referenced_evidence():
    logs = _evidence()
    metrics = _evidence("metrics:1", source_type=EvidenceSourceType.SERVICE_METRICS)
    event = RCATimelineEvent(
        event_id="event:bad",
        timestamp=_START,
        service="elsewhere",
        source=EvidenceSourceType.APPLICATION_LOGS,
        description="Evidence event.",
        evidence_ids=["logs:1"],
        event_type=RCAEventType.ERROR,
    )
    payload = _payload(timeline=[event.model_dump()])

    with pytest.raises(ValidationError, match="Timeline service"):
        RCAResult.model_validate(payload, context=_context(logs, metrics))
