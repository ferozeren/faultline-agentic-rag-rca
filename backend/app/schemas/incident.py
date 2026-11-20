"""Validated incident input schema."""

from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from app.schemas._validators import validate_time_range


class IncidentRequest(BaseModel):
    """User-provided incident scope and context."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str | None = None
    description: str = Field(min_length=1)
    affected_service: str | None = None
    environment: str | None = None
    severity: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    additional_context: dict[str, JsonValue] | None = None

    @field_validator("description", mode="before")
    @classmethod
    def strip_description(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_time_range(self) -> "IncidentRequest":
        validate_time_range(self.start_time, self.end_time)
        return self
