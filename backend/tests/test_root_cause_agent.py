"""Deterministic tests for evidence-grounded root-cause analysis."""

import asyncio
from datetime import UTC, datetime, timedelta

from app.agents.root_cause_agent import _model_context, analyze_root_cause
from app.graph.state import create_initial_state
from app.schemas import (
    AgentObservation,
    Evidence,
    EvidenceClassification,
    EvidenceSourceType,
    IncidentRequest,
    RCAEventType,
    RCAHypothesisStatus,
    RCAStatus,
)
from app.schemas.rca import RCAResult
from app.schemas.supervisor import (
    DecisionStatus,
    EvidenceConflict,
    EvidenceSufficiency,
    HistoricalCurrentAlignment,
    InvestigationDecision,
    InvestigationEvidenceSummary,
    RecommendedNextStage,
    SourceAssessment,
    SourceAssessmentStatus,
)

_START = datetime(2026, 11, 6, 10, 0, tzinfo=UTC)
_INCIDENT_SERVICE = "payment-api"


def _evidence(
    evidence_id,
    source_type,
    description,
    *,
    classification=None,
    observed_at=_START,
    service=_INCIDENT_SERVICE,
):
    if classification is None:
        classification = (
            EvidenceClassification.HISTORICAL
            if source_type is EvidenceSourceType.HISTORICAL_RAG
            else EvidenceClassification.CURRENT
        )
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        source_name=source_type.value,
        description=description,
        observed_at=observed_at,
        service=service,
        classification=classification,
    )


def _state(evidence, *, conflicts=(), incident_service=_INCIDENT_SERVICE):
    state = create_initial_state(
        IncidentRequest(
            incident_id="INC-RCA",
            description="Elevated payment latency with database errors.",
            affected_service=incident_service,
            start_time="2026-11-06T09:45:00Z",
            end_time="2026-11-06T10:30:00Z",
        )
    )
    state["consolidated_evidence"] = list(evidence)
    state["observations"] = [
        AgentObservation(
            observation_id=f"observation:{item.evidence_id}",
            agent_name=item.source_type.value,
            summary=item.description,
            evidence_ids=[item.evidence_id],
            source_types=[item.source_type],
            classification=item.classification,
        )
        for item in evidence
    ]
    grouped = {}
    for item in evidence:
        grouped.setdefault(item.source_type, []).append(item)
    source_assessments = [
        SourceAssessment(
            source_type=source_type,
            classification=(
                EvidenceClassification.HISTORICAL
                if source_type is EvidenceSourceType.HISTORICAL_RAG
                else EvidenceClassification.CURRENT
            ),
            status=SourceAssessmentStatus.AVAILABLE,
            evidence_count=len(items),
            evidence_ids=[item.evidence_id for item in items],
            summary=f"{len(items)} record(s) available.",
        )
        for source_type, items in grouped.items()
    ]
    current_ids = [
        item.evidence_id
        for item in evidence
        if item.classification is EvidenceClassification.CURRENT
    ]
    state["supervisor_decision"] = InvestigationDecision(
        incident_id=state["incident_id"],
        decision_status=DecisionStatus.READY_FOR_RCA,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        evidence_summary=InvestigationEvidenceSummary(),
        source_assessments=source_assessments,
        conflicts=list(conflicts),
        supporting_evidence_ids=current_ids,
        historical_current_alignment=(
            HistoricalCurrentAlignment.NOT_COMPARABLE
            if any(
                item.classification is EvidenceClassification.HISTORICAL
                for item in evidence
            )
            else HistoricalCurrentAlignment.NO_HISTORICAL
        ),
        confidence=0.8,
        decision_rationale="Independent current sources are available for RCA.",
        recommended_next_stage=RecommendedNextStage.ROOT_CAUSE_ANALYSIS,
    )
    return state


def _result_payload(evidence, *, timeline_reverse=False):
    current = [
        item
        for item in evidence
        if item.classification is EvidenceClassification.CURRENT
    ]
    support_ids = [item.evidence_id for item in current]
    timelines = [
        {
            "event_id": f"event:{item.evidence_id}",
            "timestamp": item.observed_at,
            "service": item.service,
            "source": item.source_type,
            "description": item.description,
            "evidence_ids": [item.evidence_id],
            "event_type": RCAEventType.ERROR,
        }
        for item in current
    ]
    if timeline_reverse:
        timelines.reverse()
    facts = [
        {
            "description": item.description,
            "evidence_ids": [item.evidence_id],
            "classification": item.classification,
        }
        for item in evidence
    ]
    return {
        "status": RCAStatus.ANALYZED,
        "incident_summary": "Payment API experienced elevated latency.",
        "primary_finding": {
            "finding_id": "finding:connections",
            "summary": "Database connection pressure is the probable cause.",
            "root_cause": "Connection acquisition failures increased under load.",
            "affected_services": list(
                dict.fromkeys(item.service for item in current if item.service)
            ),
            "contributing_factors": [],
            "supporting_evidence_ids": support_ids,
            "contradicting_evidence_ids": [],
            "confidence": 0.72,
            "confidence_explanation": "Multiple current sources support the same pattern.",
            "causal_reasoning": "Connection errors and latency rise in the same interval; timing alone is not treated as proof.",
        },
        "alternative_hypotheses": [
            {
                "hypothesis_id": "hypothesis:network",
                "description": "A network interruption may have contributed.",
                "supporting_evidence_ids": support_ids[:1],
                "contradicting_evidence_ids": [],
                "confidence": 0.25,
                "status": RCAHypothesisStatus.PLAUSIBLE,
            }
        ],
        "timeline": timelines,
        "verified_facts": facts,
        "hypotheses": [],
        "evidence_gaps": [],
        "unresolved_conflicts": [],
        "overall_confidence": 0.72,
        "confidence_explanation": "Qualitative strength assessment, not a calibrated probability.",
        "analysis_summary": "Current evidence supports a probable connection-related cause.",
    }


class FakeStructuredModel:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.schema = None
        self.prompt = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, prompt):
        self.prompt = prompt
        if self.error:
            raise self.error
        return self.response


def _corroborated_evidence():
    return [
        _evidence(
            "logs:1",
            EvidenceSourceType.APPLICATION_LOGS,
            "Database connection acquisition failures increased.",
        ),
        _evidence(
            "metrics:1",
            EvidenceSourceType.SERVICE_METRICS,
            "Connection wait duration and request latency increased.",
            observed_at=_START + timedelta(minutes=3),
        ),
    ]


def test_current_evidence_produces_a_valid_evidence_grounded_rca():
    evidence = _corroborated_evidence()
    state = _state(evidence)
    model = FakeStructuredModel(_result_payload(evidence))

    agent_result = asyncio.run(analyze_root_cause(state, llm=model))

    assert model.schema is RCAResult
    assert agent_result.rca_result.status is RCAStatus.ANALYZED
    assert agent_result.rca_result.primary_finding.supporting_evidence_ids == [
        "logs:1",
        "metrics:1",
    ]
    assert agent_result.rca_result.verified_facts[0].evidence_ids == ["logs:1"]


def test_historical_evidence_is_separated_from_current_context():
    evidence = [
        *_corroborated_evidence(),
        _evidence(
            "history:1",
            EvidenceSourceType.HISTORICAL_RAG,
            "A prior incident also recorded connection acquisition errors.",
        ),
    ]
    model = FakeStructuredModel(_result_payload(evidence))

    result = asyncio.run(analyze_root_cause(_state(evidence), llm=model))
    payload = model.prompt.to_string()

    assert "current_operational_evidence" in payload
    assert "historical_context_only" in payload
    assert result.rca_result.primary_finding.supporting_evidence_ids == [
        "logs:1",
        "metrics:1",
    ]
    history_fact = next(
        item
        for item in result.rca_result.verified_facts
        if item.evidence_ids == ["history:1"]
    )
    assert history_fact.classification is EvidenceClassification.HISTORICAL


def test_model_context_lists_exact_allowed_evidence_ids():
    evidence = [
        *_corroborated_evidence(),
        _evidence(
            "history:1",
            EvidenceSourceType.HISTORICAL_RAG,
            "A prior incident also recorded connection acquisition errors.",
        ),
    ]

    context = _model_context(
        _state(evidence),
        {item.evidence_id: item for item in evidence},
    )

    assert context["valid_current_evidence_ids"] == ["logs:1", "metrics:1"]
    assert context["valid_evidence_ids"] == ["history:1", "logs:1", "metrics:1"]


def test_model_context_excludes_raw_log_metadata_and_source_references():
    evidence = _corroborated_evidence()
    logs = evidence[0].model_copy(
        update={
            "metadata": {
                "user_id": "private-user",
                "source_ip": "192.0.2.1",
                "http_status": "504",
            },
            "raw_reference": "private-request-id",
        }
    )
    evidence[0] = logs

    context = _model_context(
        _state(evidence),
        {item.evidence_id: item for item in evidence},
    )

    assert "private-user" not in repr(context)
    assert "192.0.2.1" not in repr(context)
    assert "private-request-id" not in repr(context)
    assert all(
        "metadata" not in item and "raw_reference" not in item
        for item in context["current_operational_evidence"]
    )


def test_evidence_outside_supervisor_relevance_assessment_is_not_used():
    evidence = _corroborated_evidence()
    state = _state(evidence)
    decision = state["supervisor_decision"]
    logs_assessment = next(
        assessment
        for assessment in decision.source_assessments
        if assessment.source_type is EvidenceSourceType.APPLICATION_LOGS
    )
    state["supervisor_decision"] = decision.model_copy(
        update={
            "source_assessments": [logs_assessment],
            "supporting_evidence_ids": ["logs:1"],
        }
    )
    model = FakeStructuredModel(_result_payload([evidence[0]]))

    result = asyncio.run(analyze_root_cause(state, llm=model))

    assert "metrics:1" not in model.prompt.to_string()
    assert {event.evidence_ids[0] for event in result.rca_result.timeline} == {"logs:1"}
    assert any("excluded from RCA" in item for item in result.warnings)


def test_historical_only_support_is_rejected_as_primary_cause():
    evidence = [
        *_corroborated_evidence(),
        _evidence(
            "history:1",
            EvidenceSourceType.HISTORICAL_RAG,
            "Prior incident involved database connection errors.",
        ),
    ]
    payload = _result_payload(evidence)
    payload["status"] = RCAStatus.ANALYZED
    payload["primary_finding"] = {
        **payload["primary_finding"],
        "supporting_evidence_ids": ["history:1"],
    }

    result = asyncio.run(
        analyze_root_cause(_state(evidence), llm=FakeStructuredModel(payload))
    )

    assert result.rca_result.status is RCAStatus.PARTIAL_SUCCESS
    assert result.rca_result.primary_finding is None
    assert result.errors


def test_verified_facts_and_hypotheses_are_separate():
    evidence = _corroborated_evidence()

    result = asyncio.run(
        analyze_root_cause(
            _state(evidence),
            llm=FakeStructuredModel(_result_payload(evidence)),
        )
    ).rca_result

    assert result.verified_facts
    assert result.hypotheses == []
    assert result.alternative_hypotheses[0].status is RCAHypothesisStatus.PLAUSIBLE


def test_insufficient_model_result_preserves_evidence_linked_hypothesis():
    evidence = _corroborated_evidence()
    payload = _result_payload(evidence)
    payload["status"] = RCAStatus.INSUFFICIENT_EVIDENCE
    payload["primary_finding"] = None
    payload["alternative_hypotheses"] = []
    payload["hypotheses"] = [
        {
            "hypothesis_id": "hypothesis:connection-pressure",
            "description": (
                "Connection pressure may contribute to the observed latency, but "
                "the available evidence does not establish a connection limit."
            ),
            "supporting_evidence_ids": ["metrics:1"],
            "contradicting_evidence_ids": [],
            "confidence": 0.3,
            "status": RCAHypothesisStatus.UNRESOLVED,
        }
    ]

    result = asyncio.run(
        analyze_root_cause(
            _state(evidence),
            llm=FakeStructuredModel(payload),
        )
    ).rca_result

    assert result.status is RCAStatus.INSUFFICIENT_EVIDENCE
    assert result.primary_finding is None
    assert result.hypotheses[0].hypothesis_id == "hypothesis:connection-pressure"
    assert result.hypotheses[0].supporting_evidence_ids == ["metrics:1"]


def test_timeline_is_sorted_and_uses_evidence_timestamps():
    evidence = _corroborated_evidence()
    payload = _result_payload(evidence, timeline_reverse=True)

    result = asyncio.run(
        analyze_root_cause(_state(evidence), llm=FakeStructuredModel(payload))
    ).rca_result

    assert [event.timestamp for event in result.timeline] == [
        _START,
        _START + timedelta(minutes=3),
    ]


def test_missing_timestamps_are_explicitly_limited_without_invention():
    evidence = [
        _evidence(
            "logs:no-time",
            EvidenceSourceType.APPLICATION_LOGS,
            "A request error was observed.",
            observed_at=None,
        )
    ]
    result = asyncio.run(analyze_root_cause(_state(evidence))).rca_result

    assert result.timeline[0].timestamp is None
    assert any("no timestamp" in item for item in result.limitations)
    assert result.primary_finding is None


def test_supervisor_conflicts_are_retained_even_if_model_omits_them():
    evidence = _corroborated_evidence()
    conflict = EvidenceConflict(
        conflict_type="current_current",
        evidence_ids=["logs:1", "metrics:1"],
        description="The sources disagree about connection availability.",
    )

    result = asyncio.run(
        analyze_root_cause(
            _state(evidence, conflicts=[conflict]),
            llm=FakeStructuredModel(_result_payload(evidence)),
        )
    ).rca_result

    assert result.unresolved_conflicts == [conflict]


def test_multiple_services_remain_attributed_to_their_evidence():
    evidence = [
        _evidence(
            "logs:api",
            EvidenceSourceType.APPLICATION_LOGS,
            "Payment API request timed out.",
            service="payment-api",
        ),
        _evidence(
            "metrics:db",
            EvidenceSourceType.SERVICE_METRICS,
            "Database connection wait increased.",
            service="payments-db",
            observed_at=_START + timedelta(minutes=2),
        ),
    ]

    result = asyncio.run(
        analyze_root_cause(
            _state(evidence),
            llm=FakeStructuredModel(_result_payload(evidence)),
        )
    ).rca_result

    assert result.primary_finding.affected_services == ["payment-api", "payments-db"]
    assert {event.service for event in result.timeline} == {
        "payment-api",
        "payments-db",
    }


def test_invalid_or_unknown_model_references_return_safe_partial_result():
    evidence = _corroborated_evidence()
    invalid_payload = _result_payload(evidence)
    invalid_payload["primary_finding"]["supporting_evidence_ids"] = ["unknown:id"]

    result = asyncio.run(
        analyze_root_cause(
            _state(evidence),
            llm=FakeStructuredModel(invalid_payload),
        )
    )

    assert result.rca_result.status is RCAStatus.PARTIAL_SUCCESS
    assert result.rca_result.primary_finding is None
    assert result.errors
    assert "unknown:id" not in repr(result.rca_result)


def test_model_exception_returns_safe_result_and_preserves_input_evidence():
    evidence = _corroborated_evidence()
    state = _state(evidence)
    before = list(state["consolidated_evidence"])

    result = asyncio.run(
        analyze_root_cause(
            state,
            llm=FakeStructuredModel(error=RuntimeError("private response")),
        )
    )

    assert result.rca_result.status is RCAStatus.PARTIAL_SUCCESS
    assert result.rca_result.primary_finding is None
    assert result.rca_result.verified_facts
    assert state["consolidated_evidence"] == before
    assert "private response" not in repr(result)


def test_model_timeout_returns_a_safe_partial_result():
    class HangingModel(FakeStructuredModel):
        async def ainvoke(self, prompt):
            await asyncio.Event().wait()

    evidence = _corroborated_evidence()
    result = asyncio.run(
        analyze_root_cause(
            _state(evidence),
            llm=HangingModel(),
            timeout_seconds=0.01,
        )
    )

    assert result.rca_result.status is RCAStatus.PARTIAL_SUCCESS
    assert result.rca_result.primary_finding is None
    assert result.errors == ("RCA model analysis failed (TimeoutError).",)


def test_historical_only_or_empty_evidence_cannot_fabricate_a_cause():
    historical = _evidence(
        "history:1",
        EvidenceSourceType.HISTORICAL_RAG,
        "Previous incident showed a service timeout.",
    )
    result = asyncio.run(analyze_root_cause(_state([historical]))).rca_result

    assert result.status is RCAStatus.INSUFFICIENT_EVIDENCE
    assert result.primary_finding is None
    assert result.verified_facts[0].classification is EvidenceClassification.HISTORICAL

    empty = asyncio.run(analyze_root_cause(_state([]))).rca_result
    assert empty.status is RCAStatus.INSUFFICIENT_EVIDENCE
    assert empty.primary_finding is None


def test_upstream_evidence_gaps_are_preserved():
    evidence = _corroborated_evidence()
    state = _state(evidence)
    state["evidence_gaps"] = [
        {
            "gap_id": "gap:prometheus",
            "description": "Prometheus metrics were unavailable.",
            "missing_source_types": [EvidenceSourceType.PROMETHEUS],
            "recommended_data_sources": ["prometheus"],
        }
    ]

    result = asyncio.run(
        analyze_root_cause(
            state,
            llm=FakeStructuredModel(_result_payload(evidence)),
        )
    ).rca_result

    assert [gap.gap_id for gap in result.evidence_gaps] == ["gap:prometheus"]


def test_no_supervisor_decision_skips_without_model_invocation():
    state = _state(_corroborated_evidence())
    state["supervisor_decision"] = None
    model = FakeStructuredModel(_result_payload(_corroborated_evidence()))

    result = asyncio.run(analyze_root_cause(state, llm=model))

    assert result.rca_result.status is RCAStatus.SKIPPED
    assert model.prompt is None
    assert result.warnings


def test_unknown_evidence_claim_in_model_output_is_rejected():
    evidence = _corroborated_evidence()
    payload = _result_payload(evidence)
    payload["verified_facts"][0] = {
        **payload["verified_facts"][0],
        "evidence_ids": ["nonexistent"],
    }

    result = asyncio.run(
        analyze_root_cause(_state(evidence), llm=FakeStructuredModel(payload))
    )

    assert result.rca_result.primary_finding is None
    assert result.errors
