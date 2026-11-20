"""HTTP health and OpenAPI behavior."""

from app.config import API_VERSION, APPLICATION_NAME
from app.api.routes import health_routes
from app.main import create_app
from fastapi.testclient import TestClient


def test_health_reports_application_liveness():
    response = TestClient(create_app()).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "application": APPLICATION_NAME,
        "api_version": API_VERSION,
        "external_dependencies": "not_checked",
    }


def test_readiness_does_not_claim_external_dependency_health(monkeypatch):
    client = TestClient(create_app())
    monkeypatch.setattr(health_routes, "database_is_available", lambda: True)
    response = client.get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["external_dependencies"] == "not_checked"
    assert response.json()["database_status"] == "connected"


def test_readiness_reports_unavailable_database(monkeypatch):
    monkeypatch.setattr(health_routes, "database_is_available", lambda: False)

    response = TestClient(create_app()).get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["database_status"] == "unavailable"


def test_docs_and_openapi_are_available():
    client = TestClient(create_app())

    assert client.get("/docs").status_code == 200
    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    assert openapi_response.json()["info"]["version"] == API_VERSION
    assert "/api/v1/investigations" in openapi_response.json()["paths"]


def test_invalid_correlation_id_is_replaced():
    response = TestClient(create_app()).get(
        "/api/v1/health",
        headers={"X-Correlation-ID": "bad id\nsecret"},
    )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] != "bad id\nsecret"
    assert len(response.headers["X-Correlation-ID"]) == 36
