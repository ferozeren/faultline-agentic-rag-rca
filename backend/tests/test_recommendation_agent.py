"""Tests for validated, root-cause-specific recommendation generation."""

import asyncio
import copy

import pytest
from app.agents.recommendation_agent import (
    _approval_required,
    generate_recommendations,
)
from app.graph.state import create_initial_state
from app.schemas import (
    Evidence,
    EvidenceClassification,
    EvidenceSourceType,
    IncidentRequest,
)
from app.schemas.rca import RCAFinding, RCAResult, RCAStatus
from app.schemas.recommendations import (
    Recommendation,
    RecommendationBatch,
    RecommendationCategory,
    RecommendationStatus,
)
from app.schemas.supervisor import ConflictType, EvidenceConflict

_CASES = (
    (
        "Database connection pool saturation",
        "Database connection acquisition failures and pool saturation increased.",
        "database connection-pool saturation",
    ),
    (
        "CPU saturation under request load",
        "CPU utilization reached saturation in the payment service under workload.",
        "CPU-intensive workloads, resource contention, and service-level utilization",
    ),
    (
        "Memory growth caused by object retention",
        "Memory usage increased and retained objects were observed.",
        "memory growth and object retention",
    ),
    (
        "API request timeouts from downstream latency",
        "API requests timed out as downstream database latency increased.",
        "downstream latency, timeout behavior, and dependency availability",
    ),
)


def _state(
    cause: str = _CASES[0][0],
    evidence_description: str = _CASES[0][1],
    *,
    historical: bool = False,
    incident_id: str = "INC-REC",
) -> dict:
    request = IncidentRequest(
        incident_id=incident_id,
        description="Elevated payment latency",
        affected_service="payment-api",
        start_time="2026-11-06T09:45:00Z",
        end_time="2026-11-06T10:30:00Z",
    )
    state = create_initial_state(request)
    evidence = Evidence(
        evidence_id="logs:current",
        source_type=EvidenceSourceType.APPLICATION_LOGS,
        source_name="application logs",
        description=evidence_description,
        service="payment-api",
        classification=EvidenceClassification.CURRENT,
    )
    state["consolidated_evidence"] = [evidence]
    state["rca_result"] = RCAResult(
        status=RCAStatus.ANALYZED,
        incident_summary="Payment API experienced elevated latency.",
        primary_finding=RCAFinding(
            finding_id="finding:primary",
            summary=cause,
            root_cause=cause,
            affected_services=["payment-api"],
            supporting_evidence_ids=[evidence.evidence_id],
            contradicting_evidence_ids=[],
            confidence=0.8,
            confidence_explanation="Current evidence supports this finding.",
            causal_reasoning=f"The current evidence supports {cause}.",
        ),
        overall_confidence=0.8,
        confidence_explanation="The RCA is evidence-grounded.",
        analysis_summary="The RCA identified a supported primary finding.",
    )
    if historical:
        state["historical_evidence"] = [
            Evidence(
                evidence_id="history:prior",
                source_type=EvidenceSourceType.HISTORICAL_RAG,
                source_name="incident archive",
                description=f"A prior incident also mentioned {cause}.",
                service="payment-api",
                classification=EvidenceClassification.HISTORICAL,
            )
        ]
    return state


def _batch_payload(state: dict) -> dict:
    finding = state["rca_result"].primary_finding
    cause = finding.root_cause
    shared = {
        "related_finding_ids": [finding.finding_id],
        "supporting_evidence_ids": finding.supporting_evidence_ids,
        "affected_services": ["payment-api"],
        "confidence": 0.8,
        "verification_steps": [f"Verify the current evidence for {cause}."],
    }
    return {
        "recommendations": [
            {
                **shared,
                "recommendation_id": "rec:corrective",
                "title": f"Mitigate {cause}",
                "description": f"Mitigate the validated {cause} and verify service recovery.",
                "category": "corrective",
                "priority": "high",
                "urgency": "immediate",
                "rationale": f"This addresses {cause} identified in the validated RCA.",
                "expected_impact": f"Reduce impact associated with {cause}.",
                "requires_human_approval": False,
            },
            {
                **shared,
                "recommendation_id": "rec:preventive",
                "title": f"Monitor recurrence of {cause}",
                "description": f"Monitor recurrence of the validated {cause}.",
                "category": "preventive",
                "priority": "medium",
                "urgency": "planned",
                "rationale": f"Recurrence monitoring is relevant to {cause}.",
                "expected_impact": f"Detect recurrence of {cause} earlier.",
                "requires_human_approval": False,
            },
            {
                **shared,
                "recommendation_id": "rec:diagnostic",
                "title": f"Verify {cause}",
                "description": f"Read-only inspect the evidence for {cause}.",
                "category": "diagnostic",
                "priority": "medium",
                "urgency": "planned",
                "rationale": f"Verification can resolve uncertainty about {cause}.",
                "expected_impact": f"Clarify the evidence for {cause}.",
                "requires_human_approval": False,
            },
        ]
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
        if callable(self.response):
            response = self.response()
            if asyncio.iscoroutine(response):
                return await response
            return response
        return self.response


@pytest.mark.parametrize(("cause", "evidence", "focus"), _CASES)
def test_model_receives_root_cause_specific_recommendation_focus(
    cause, evidence, focus
):
    state = _state(cause, evidence)
    model = FakeStructuredModel(_batch_payload(state))

    result = asyncio.run(generate_recommendations(state, llm=model))

    assert model.schema is RecommendationBatch
    assert result.status is RecommendationStatus.COMPLETED
    assert len(result.recommendations) == 3
    assert all(
        cause.casefold() in item.description.casefold()
        for item in result.recommendations
    )
    assert focus.casefold() in model.prompt.to_string().casefold()


def test_recommendation_model_receives_no_raw_log_metadata():
    state = _state()
    evidence = state["consolidated_evidence"][0].model_copy(
        update={
            "metadata": {
                "user_id": "private-user",
                "source_ip": "192.0.2.1",
                "error_code": "GATEWAY_TIMEOUT",
            },
            "raw_reference": "private-request-id",
        }
    )
    state["consolidated_evidence"] = [evidence]
    model = FakeStructuredModel(_batch_payload(state))

    result = asyncio.run(generate_recommendations(state, llm=model))

    prompt = model.prompt.to_string()
    assert result.status is RecommendationStatus.COMPLETED
    assert "private-user" not in prompt
    assert "192.0.2.1" not in prompt
    assert "private-request-id" not in prompt
    assert "GATEWAY_TIMEOUT" not in prompt


def test_actions_are_separated_and_reference_only_the_validated_finding_and_evidence():
    state = _state()
    model = FakeStructuredModel(_batch_payload(state))

    result = asyncio.run(generate_recommendations(state, llm=model))

    assert result.corrective_action_count == 1
    assert result.preventive_action_count == 1
    assert result.diagnostic_action_count == 1
    assert result.source_rca_finding_ids == ["finding:primary"]
    assert all(
        item.related_finding_ids == ["finding:primary"]
        and item.supporting_evidence_ids == ["logs:current"]
        for item in result.recommendations
    )
    corrective, preventive, diagnostic = result.recommendations
    assert corrective.requires_human_approval is True
    assert preventive.requires_human_approval is False
    assert diagnostic.requires_human_approval is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("supporting_evidence_ids", ["logs:unknown"]),
        ("related_finding_ids", ["finding:unknown"]),
    ],
)
def test_unknown_evidence_or_finding_references_reject_all_model_actions(field, value):
    state = _state()
    payload = _batch_payload(state)
    payload["recommendations"][0][field] = value
    original_rca = state["rca_result"]
    original_evidence = list(state["consolidated_evidence"])

    result = asyncio.run(
        generate_recommendations(state, llm=FakeStructuredModel(payload))
    )

    assert result.status is RecommendationStatus.FAILED
    assert result.recommendations == []
    assert result.errors
    assert state["rca_result"] is original_rca
    assert state["consolidated_evidence"] == original_evidence


def test_unsupported_claims_are_rejected():
    state = _state()
    payload = _batch_payload(state)
    payload["recommendations"][0]["description"] += " Restart the Kubernetes cluster."

    result = asyncio.run(
        generate_recommendations(state, llm=FakeStructuredModel(payload))
    )

    assert result.status is RecommendationStatus.FAILED
    assert result.recommendations == []
    assert "ValueError" in result.errors[0]


def test_historical_evidence_is_context_only_and_cannot_support_an_action():
    state = _state(historical=True)
    payload = _batch_payload(state)
    payload["recommendations"][0]["supporting_evidence_ids"] = ["history:prior"]
    model = FakeStructuredModel(payload)

    result = asyncio.run(generate_recommendations(state, llm=model))

    prompt = model.prompt.to_string()
    assert "historical_context_only" in prompt
    assert "history:prior" in prompt
    assert result.status is RecommendationStatus.FAILED
    assert result.recommendations == []


def test_contradictions_are_passed_as_uncertainty_not_as_supporting_evidence():
    state = _state()
    contradiction = Evidence(
        evidence_id="metrics:conflict",
        source_type=EvidenceSourceType.SERVICE_METRICS,
        source_name="service metrics",
        description="A network interruption may have affected the same interval.",
        service="payment-api",
        classification=EvidenceClassification.CURRENT,
    )
    state["consolidated_evidence"].append(contradiction)
    original = state["rca_result"]
    finding = original.primary_finding.model_copy(
        update={"contradicting_evidence_ids": [contradiction.evidence_id]}
    )
    state["rca_result"] = original.model_copy(
        update={
            "primary_finding": finding,
            "unresolved_conflicts": [
                EvidenceConflict(
                    conflict_type=ConflictType.CURRENT_CURRENT,
                    evidence_ids=["logs:current", contradiction.evidence_id],
                    description="Current evidence differs on the contributing network condition.",
                )
            ],
        }
    )
    model = FakeStructuredModel(_batch_payload(state))

    result = asyncio.run(generate_recommendations(state, llm=model))

    prompt = model.prompt.to_string()
    assert "contradicting_current_evidence" in prompt
    assert "unresolved_conflicts" in prompt
    assert result.status is RecommendationStatus.COMPLETED
    assert result.unresolved_conflicts
    assert all(
        item.supporting_evidence_ids == ["logs:current"]
        for item in result.recommendations
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["recommendations"][0].update({"confidence": 1.5}),
        lambda payload: payload["recommendations"][0].update(
            {"verification_steps": []}
        ),
    ],
)
def test_invalid_confidence_or_empty_verification_steps_are_rejected(mutate):
    state = _state()
    payload = _batch_payload(state)
    mutate(payload)

    result = asyncio.run(
        generate_recommendations(state, llm=FakeStructuredModel(payload))
    )

    assert result.status is RecommendationStatus.FAILED
    assert result.recommendations == []
    assert result.errors


def test_invented_numeric_thresholds_are_rejected():
    state = _state()
    payload = _batch_payload(state)
    payload["recommendations"][0]["description"] += (
        " Increase the maximum database connections to 800."
    )

    result = asyncio.run(
        generate_recommendations(state, llm=FakeStructuredModel(payload))
    )

    assert result.status is RecommendationStatus.FAILED
    assert result.recommendations == []


def test_model_failure_returns_typed_failure_without_fabricated_actions():
    state = _state()
    original_rca = state["rca_result"]
    original_evidence = copy.deepcopy(state["consolidated_evidence"])

    result = asyncio.run(
        generate_recommendations(
            state,
            llm=FakeStructuredModel(error=RuntimeError("private provider response")),
        )
    )

    assert result.status is RecommendationStatus.FAILED
    assert result.recommendations == []
    assert result.errors == [
        "Recommendation generation failed (RuntimeError); no recommendations were returned."
    ]
    assert "private provider response" not in repr(result)
    assert state["rca_result"] is original_rca
    assert state["consolidated_evidence"] == original_evidence


def test_model_timeout_returns_typed_failure():
    async def wait_forever(_prompt=None):
        await asyncio.sleep(10)

    state = _state()
    model = FakeStructuredModel(response=wait_forever)

    result = asyncio.run(
        generate_recommendations(state, llm=model, timeout_seconds=0.001)
    )

    assert result.status is RecommendationStatus.FAILED
    assert result.recommendations == []
    assert "TimeoutError" in result.errors[0]


def test_missing_nvidia_configuration_returns_safe_failure_without_fallback(
    monkeypatch,
):
    state = _state()

    def fail_configuration():
        raise RuntimeError("NVIDIA recommendation model configuration is incomplete.")

    monkeypatch.setattr(
        "app.agents.recommendation_agent._nvidia_model",
        fail_configuration,
    )
    result = asyncio.run(generate_recommendations(state))

    assert result.status is RecommendationStatus.FAILED
    assert result.recommendations == []
    assert "RuntimeError" in result.errors[0]


def test_insufficient_or_skipped_rca_does_not_generate_actions():
    state = _state()
    state["rca_result"] = RCAResult(
        status=RCAStatus.INSUFFICIENT_EVIDENCE,
        incident_summary="No cause was established.",
        overall_confidence=0,
        confidence_explanation="No root cause is available.",
        analysis_summary="Evidence was insufficient.",
    )

    result = asyncio.run(generate_recommendations(state))

    assert result.status is RecommendationStatus.INSUFFICIENT_EVIDENCE
    assert result.recommendations == []


@pytest.mark.parametrize(
    "action",
    [
        "Restart the service after approval.",
        "Scale the service after approval.",
        "Change the database configuration after approval.",
        "Deploy the change after approval.",
    ],
)
def test_disruptive_actions_require_human_approval(action):
    recommendation = Recommendation(
        recommendation_id="rec:approval",
        title=action,
        description=action,
        category=RecommendationCategory.DIAGNOSTIC,
        rationale="A reviewer must approve this action.",
        supporting_evidence_ids=["logs:current"],
        related_finding_ids=["finding:primary"],
        verification_steps=["Verify the action outcome."],
        confidence=0.5,
        requires_human_approval=False,
    )

    assert _approval_required(recommendation) is True


def test_read_only_diagnostics_do_not_require_human_approval():
    recommendation = Recommendation(
        recommendation_id="rec:read-only",
        title="Inspect current evidence",
        description="Compare current latency and errors.",
        category=RecommendationCategory.DIAGNOSTIC,
        rationale="Read-only evidence review.",
        supporting_evidence_ids=["logs:current"],
        related_finding_ids=["finding:primary"],
        verification_steps=["Verify the evidence comparison."],
        confidence=0.5,
        requires_human_approval=False,
    )

    assert _approval_required(recommendation) is False
