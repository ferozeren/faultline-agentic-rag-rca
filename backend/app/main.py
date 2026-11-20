"""FastAPI application for incident investigations."""

from __future__ import annotations

import logging
import re
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.health_routes import router as health_router
from app.api.routes.investigation_routes import router as investigation_router
from app.api.routes.operational_routes import router as operational_router
from app.config import API_VERSION, APPLICATION_NAME, CORS_ALLOWED_ORIGINS
from app.schemas.api import APIErrorResponse
from app.services.operational_data_service import OperationalDataServiceError
from app.services.investigation_service import (
    InvestigationPersistenceError,
    InvestigationServiceError,
)

logger = logging.getLogger(__name__)
_CORRELATION_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _error_response(
    request: Request,
    *,
    status_code: int,
    error: str,
    message: str,
) -> JSONResponse:
    correlation_id = getattr(request.state, "correlation_id", str(uuid4()))
    return JSONResponse(
        status_code=status_code,
        content=APIErrorResponse(
            error=error,
            message=message,
            correlation_id=correlation_id,
        ).model_dump(mode="json"),
        headers={"X-Correlation-ID": correlation_id},
    )


def create_app() -> FastAPI:
    """Create the API without starting MCP clients or contacting external services."""
    application = FastAPI(
        title=APPLICATION_NAME,
        version=API_VERSION,
        description="HTTP API for evidence-grounded IT incident investigations.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ALLOWED_ORIGINS),
        allow_credentials=bool(CORS_ALLOWED_ORIGINS)
        and "*" not in CORS_ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Correlation-ID"],
        expose_headers=["X-Correlation-ID"],
    )

    @application.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        requested_id = request.headers.get("X-Correlation-ID", "")
        correlation_id = (
            requested_id if _CORRELATION_ID.fullmatch(requested_id) else str(uuid4())
        )
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            error="invalid_request",
            message="The request does not match the required API schema.",
        )

    @application.exception_handler(ResponseValidationError)
    async def response_validation_handler(
        request: Request,
        error: ResponseValidationError,
    ) -> JSONResponse:
        logger.error(
            "API response validation failed",
            extra={
                "correlation_id": getattr(request.state, "correlation_id", None),
                "failure_kind": type(error).__name__,
            },
        )
        return _error_response(
            request,
            status_code=500,
            error="invalid_response",
            message="The investigation response could not be validated.",
        )

    @application.exception_handler(InvestigationServiceError)
    async def investigation_service_error_handler(
        request: Request,
        error: InvestigationServiceError,
    ) -> JSONResponse:
        logger.error(
            "Investigation service failed",
            extra={
                "correlation_id": getattr(request.state, "correlation_id", None),
                "failure_kind": error.code,
            },
        )
        return _error_response(
            request,
            status_code=500,
            error=error.code,
            message=error.public_message,
        )

    @application.exception_handler(InvestigationPersistenceError)
    async def investigation_persistence_error_handler(
        request: Request,
        error: InvestigationPersistenceError,
    ) -> JSONResponse:
        logger.error(
            "Investigation persistence failed",
            extra={
                "correlation_id": getattr(request.state, "correlation_id", None),
                "failure_kind": error.code,
            },
        )
        return _error_response(
            request,
            status_code=503,
            error=error.code,
            message=error.public_message,
        )

    @application.exception_handler(OperationalDataServiceError)
    async def operational_data_service_error_handler(
        request: Request,
        error: OperationalDataServiceError,
    ) -> JSONResponse:
        logger.warning(
            "Operational data request failed",
            extra={
                "correlation_id": getattr(request.state, "correlation_id", None),
                "failure_kind": error.code,
            },
        )
        return _error_response(
            request,
            status_code=error.status_code,
            error=error.code,
            message=error.public_message,
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        logger.error(
            "Unexpected API failure",
            extra={
                "correlation_id": getattr(request.state, "correlation_id", None),
                "failure_kind": type(error).__name__,
            },
        )
        return _error_response(
            request,
            status_code=500,
            error="internal_server_error",
            message="The request could not be completed.",
        )

    application.include_router(health_router, prefix="/api/v1")
    application.include_router(investigation_router, prefix="/api/v1")
    application.include_router(operational_router, prefix="/api/v1")
    return application


app = create_app()
