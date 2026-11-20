"""Read-only operational data and source-status endpoints."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_operational_data_service
from app.schemas.operational import (
    KnowledgeSearchResponse,
    LogQueryResponse,
    MCPServerStatus,
    MetricsQueryResponse,
    OperationOverview,
    VectorStoreStatus,
)
from app.services.operational_data_service import (
    OperationalDataService,
    OperationalDataServiceError,
)

router = APIRouter(tags=["operations"])


def _validate_time_range(
    start_time: datetime | None,
    end_time: datetime | None,
) -> None:
    if start_time is not None and end_time is not None:
        start = (
            start_time.replace(tzinfo=UTC)
            if start_time.tzinfo is None
            else start_time.astimezone(UTC)
        )
        end = (
            end_time.replace(tzinfo=UTC)
            if end_time.tzinfo is None
            else end_time.astimezone(UTC)
        )
        if start > end:
            raise OperationalDataServiceError(
                "invalid_time_range",
                "end_time must be at or after start_time.",
                status_code=422,
            )


def _validate_filter(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise OperationalDataServiceError(
            "invalid_filter",
            f"{name} must not be empty.",
            status_code=422,
        )
    return normalized


@router.get("/overview", response_model=OperationOverview)
async def operation_overview(
    service: Annotated[
        OperationalDataService,
        Depends(get_operational_data_service),
    ],
) -> OperationOverview:
    """Report actual local data and upstream dependency health."""
    return await service.overview()


@router.get("/mcp/status", response_model=list[MCPServerStatus])
async def mcp_status(
    service: Annotated[
        OperationalDataService,
        Depends(get_operational_data_service),
    ],
) -> list[MCPServerStatus]:
    """Discover configured MCP tools and probe each server's real health."""
    return await service.mcp_status()


@router.get("/data/logs", response_model=LogQueryResponse)
async def query_logs(
    service: Annotated[
        OperationalDataService,
        Depends(get_operational_data_service),
    ],
    service_name: Annotated[str | None, Query(min_length=1)] = None,
    level: Annotated[str | None, Query(min_length=1)] = None,
    search: Annotated[str | None, Query(min_length=1, max_length=300)] = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> LogQueryResponse:
    """Filter and paginate logs returned by the existing Log MCP server."""
    _validate_time_range(start_time, end_time)
    return await service.query_logs(
        service=_validate_filter(service_name, "service_name"),
        level=_validate_filter(level, "level"),
        search=_validate_filter(search, "search"),
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )


@router.get("/data/metrics", response_model=MetricsQueryResponse)
async def query_metrics(
    service: Annotated[
        OperationalDataService,
        Depends(get_operational_data_service),
    ],
    service_name: Annotated[str | None, Query(min_length=1)] = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> MetricsQueryResponse:
    """Return time-filtered records from the existing Metrics MCP server."""
    _validate_time_range(start_time, end_time)
    return await service.query_metrics(
        service=service_name,
        start_time=start_time,
        end_time=end_time,
    )


@router.get("/data/knowledge/status", response_model=VectorStoreStatus)
async def knowledge_status(
    service: Annotated[
        OperationalDataService,
        Depends(get_operational_data_service),
    ],
) -> VectorStoreStatus:
    """Check the local Chroma collection and return a small document preview."""
    return await service.knowledge_status()


@router.get("/data/knowledge/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    query: Annotated[str, Query(min_length=1, max_length=500)],
    service: Annotated[
        OperationalDataService,
        Depends(get_operational_data_service),
    ],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> KnowledgeSearchResponse:
    """Search stored incidents without an LLM call; results remain historical context."""
    return await service.search_knowledge(query.strip(), limit=limit)
