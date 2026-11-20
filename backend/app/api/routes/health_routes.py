"""Liveness and readiness endpoints."""

from typing import Literal

from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.config import API_VERSION, APPLICATION_NAME
from app.database.session import database_is_available

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["healthy", "ready"]
    application: str
    api_version: str
    external_dependencies: Literal["not_checked"] = "not_checked"


class ReadinessResponse(HealthResponse):
    status: Literal["ready", "not_ready"]
    database_status: Literal["connected", "unavailable"]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report application liveness without probing external services."""
    return HealthResponse(
        status="healthy",
        application=APPLICATION_NAME,
        api_version=API_VERSION,
    )


@router.get("/ready", response_model=ReadinessResponse)
def ready(response: Response) -> ReadinessResponse:
    """Report API and PostgreSQL readiness without probing other dependencies."""
    database_available = database_is_available()
    if not database_available:
        response.status_code = 503
    return ReadinessResponse(
        status="ready" if database_available else "not_ready",
        application=APPLICATION_NAME,
        api_version=API_VERSION,
        database_status="connected" if database_available else "unavailable",
    )
