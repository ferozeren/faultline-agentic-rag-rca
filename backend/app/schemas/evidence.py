"""Evidence schemas shared by historical retrieval and operational sources."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from app.schemas._validators import validate_time_range


class EvidenceSourceType(str, Enum):
    """Supported historical and operational evidence sources."""

    HISTORICAL_RAG = "historical_rag"
    APPLICATION_LOGS = "application_logs"
    SERVICE_METRICS = "service_metrics"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"


class EvidenceClassification(str, Enum):
    """Distinguish prior incident records from present observations."""

    HISTORICAL = "historical"
    CURRENT = "current"


class Evidence(BaseModel):
    """A traceable observation from a historical or operational source."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    source_type: EvidenceSourceType
    source_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    observed_at: datetime | None = None
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None
    service: str | None = None
    severity: str | None = None
    raw_reference: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    classification: EvidenceClassification

    @model_validator(mode="before")
    @classmethod
    def infer_classification(cls, value: object) -> object:
        if not isinstance(value, dict) or "classification" in value:
            return value

        source_type = value.get("source_type")
        if source_type == EvidenceSourceType.HISTORICAL_RAG:
            classification = EvidenceClassification.HISTORICAL
        elif source_type in {
            EvidenceSourceType.APPLICATION_LOGS,
            EvidenceSourceType.SERVICE_METRICS,
            EvidenceSourceType.PROMETHEUS,
            EvidenceSourceType.GRAFANA,
        }:
            classification = EvidenceClassification.CURRENT
        else:
            return value
        return {**value, "classification": classification}

    @model_validator(mode="after")
    def validate_classification_and_time_range(self) -> "Evidence":
        expected = (
            EvidenceClassification.HISTORICAL
            if self.source_type is EvidenceSourceType.HISTORICAL_RAG
            else EvidenceClassification.CURRENT
        )
        if self.classification is not expected:
            raise ValueError(
                f"{self.source_type.value} evidence must be classified as "
                f"{expected.value}."
            )
        validate_time_range(self.time_range_start, self.time_range_end)
        return self
