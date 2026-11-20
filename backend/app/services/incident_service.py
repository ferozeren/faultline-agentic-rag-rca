"""Compatibility exports for the investigation service."""

from app.services.investigation_service import (
    InvestigationService,
    InvestigationServiceError,
    InvestigationServiceResult,
)

__all__ = [
    "InvestigationService",
    "InvestigationServiceError",
    "InvestigationServiceResult",
]
