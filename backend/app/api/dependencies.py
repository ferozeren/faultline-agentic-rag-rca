"""Dependencies shared by FastAPI routes."""

from app.services.investigation_service import InvestigationService
from app.services.operational_data_service import OperationalDataService


def get_investigation_service() -> InvestigationService:
    """Return a stateless service for one request."""
    return InvestigationService()


def get_operational_data_service() -> OperationalDataService:
    """Return a stateless service for one operational-data request."""
    return OperationalDataService()
