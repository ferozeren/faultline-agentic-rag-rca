"""HTTP contracts and validation for operational data endpoints."""

from fastapi.testclient import TestClient

from app.api.dependencies import get_operational_data_service
from app.main import create_app
from app.schemas.operational import LogQueryResponse
from app.services.operational_data_service import OperationalDataServiceError


class FakeOperationalDataService:
    def __init__(self):
        self.log_query = None

    async def query_logs(self, **kwargs):
        self.log_query = kwargs
        return LogQueryResponse(
            logs=[],
            total=0,
            offset=kwargs["offset"],
            limit=kwargs["limit"],
        )


def test_operational_endpoints_are_published_in_openapi():
    response = TestClient(create_app()).get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/overview" in paths
    assert "/api/v1/mcp/status" in paths
    assert "/api/v1/data/logs" in paths
    assert "/api/v1/data/metrics" in paths
    assert "/api/v1/data/knowledge/status" in paths
    assert "/api/v1/data/knowledge/search" in paths


def test_log_route_validates_mixed_timezone_range_without_type_error():
    app = create_app()
    service = FakeOperationalDataService()
    app.dependency_overrides[get_operational_data_service] = lambda: service
    client = TestClient(app)

    response = client.get(
        "/api/v1/data/logs",
        params={
            "start_time": "2026-10-01T02:00:00+02:00",
            "end_time": "2026-10-01T00:00:00Z",
        },
    )

    assert response.status_code == 200
    assert service.log_query is not None
    assert service.log_query["start_time"].isoformat() == "2026-10-01T02:00:00+02:00"


def test_log_route_rejects_reversed_mixed_timezone_range():
    app = create_app()
    app.dependency_overrides[get_operational_data_service] = FakeOperationalDataService

    response = TestClient(app).get(
        "/api/v1/data/logs",
        params={
            "start_time": "2026-10-01T02:00:00+02:00",
            "end_time": "2026-09-30T23:59:00Z",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_time_range"


def test_log_route_rejects_whitespace_filters():
    app = create_app()
    app.dependency_overrides[get_operational_data_service] = FakeOperationalDataService

    response = TestClient(app).get(
        "/api/v1/data/logs",
        params={"service_name": "   "},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_filter"


def test_operational_error_response_does_not_expose_internal_exception():
    class FailingService:
        async def query_logs(self, **_kwargs):
            raise OperationalDataServiceError(
                "logs_unavailable",
                "The local Log MCP data source is unavailable.",
            )

    app = create_app()
    app.dependency_overrides[get_operational_data_service] = FailingService
    response = TestClient(app).get("/api/v1/data/logs")

    assert response.status_code == 503
    assert response.json()["error"] == "logs_unavailable"
    assert "traceback" not in response.text.casefold()
