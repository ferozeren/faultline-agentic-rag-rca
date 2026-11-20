"""Validation tests for the recommendation output models."""

from datetime import UTC, datetime

import pytest
from app.schemas.recommendations import (
    Recommendation,
    RecommendationBatch,
    RecommendationCategory,
    RecommendationPriority,
    RecommendationResult,
    RecommendationStatus,
    RecommendationUrgency,
)
from pydantic import ValidationError


def _recommendation(category: str, *, recommendation_id: str | None = None) -> dict:
    return {
        "recommendation_id": recommendation_id or f"rec:{category}",
        "title": f"Review the {category} action",
        "description": "Review the validated failure mode.",
        "category": category,
        "priority": "high",
        "urgency": "planned",
        "rationale": "This follows the validated RCA.",
        "supporting_evidence_ids": ["logs:1"],
        "related_finding_ids": ["finding:1"],
        "affected_services": ["checkout"],
        "expected_impact": "Improve service health.",
        "implementation_risk": "Review the operational impact.",
        "implementation_effort": "low",
        "prerequisites": ["Confirm the incident context."],
        "verification_steps": ["Verify the outcome against current evidence."],
        "rollback_considerations": ["Use the approved rollback process if needed."],
        "confidence": 0.75,
        "requires_human_approval": False,
    }


def test_recommendation_validates_required_references_and_output_fields():
    recommendation = Recommendation.model_validate(_recommendation("diagnostic"))

    assert recommendation.category is RecommendationCategory.DIAGNOSTIC
    assert recommendation.priority is RecommendationPriority.HIGH
    assert recommendation.urgency is RecommendationUrgency.PLANNED
    assert recommendation.supporting_evidence_ids == ["logs:1"]
    assert recommendation.related_finding_ids == ["finding:1"]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, True])
def test_recommendation_rejects_invalid_confidence(confidence):
    payload = _recommendation("diagnostic")
    payload["confidence"] = confidence

    with pytest.raises(ValidationError, match="confidence"):
        Recommendation.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verification_steps", []),
        ("verification_steps", ["  "]),
        ("supporting_evidence_ids", []),
        ("related_finding_ids", [""]),
    ],
)
def test_recommendation_rejects_missing_references_or_verification(field, value):
    payload = _recommendation("diagnostic")
    payload[field] = value

    with pytest.raises(ValidationError):
        Recommendation.model_validate(payload)


def test_batch_requires_corrective_preventive_and_diagnostic_actions():
    batch = RecommendationBatch(
        recommendations=[
            _recommendation("corrective"),
            _recommendation("preventive"),
            _recommendation("diagnostic"),
        ]
    )

    assert {item.category for item in batch.recommendations} == set(
        RecommendationCategory
    )


def test_batch_rejects_missing_category_and_duplicate_ids():
    with pytest.raises(ValidationError, match="must include corrective"):
        RecommendationBatch(
            recommendations=[
                _recommendation("corrective", recommendation_id="rec:one"),
                _recommendation("corrective", recommendation_id="rec:two"),
                _recommendation("diagnostic"),
            ]
        )

    with pytest.raises(ValidationError, match="identifiers must be unique"):
        RecommendationBatch(
            recommendations=[
                _recommendation("corrective", recommendation_id="rec:duplicate"),
                _recommendation("preventive", recommendation_id="rec:duplicate"),
                _recommendation("diagnostic"),
            ]
        )


def test_result_validates_action_counts_and_timestamp():
    result = RecommendationResult(
        status=RecommendationStatus.COMPLETED,
        recommendations=[
            Recommendation.model_validate(_recommendation("corrective")),
            Recommendation.model_validate(_recommendation("preventive")),
            Recommendation.model_validate(_recommendation("diagnostic")),
        ],
        corrective_action_count=1,
        preventive_action_count=1,
        diagnostic_action_count=1,
        overall_summary="All recommendation categories are present.",
        source_rca_finding_ids=["finding:1"],
        generated_at=datetime(2026, 11, 6, tzinfo=UTC),
    )

    assert result.generated_at.tzinfo is not None
    assert result.status is RecommendationStatus.COMPLETED


def test_result_rejects_incorrect_action_counts():
    with pytest.raises(ValidationError, match="counts must match"):
        RecommendationResult(
            status=RecommendationStatus.COMPLETED,
            recommendations=[
                Recommendation.model_validate(_recommendation("corrective"))
            ],
            corrective_action_count=0,
            overall_summary="Count mismatch.",
        )
