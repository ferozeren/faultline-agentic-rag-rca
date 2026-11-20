"""Tests for the read-only Prometheus live-connection diagnostic."""

from unittest.mock import patch

from app.mcp_servers import prometheus_mcp_server as server
from app.mcp_servers import validate_prometheus_connection as diagnostic


def _configure(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_URL", "https://prometheus.example/api/prom")
    monkeypatch.setenv("PROMETHEUS_BEARER_TOKEN", "test-token")
    monkeypatch.setenv("PROMETHEUS_USERNAME", "test-user")


def test_diagnostic_reports_valid_empty_live_api_responses(monkeypatch):
    _configure(monkeypatch)
    with (
        patch.object(
            server,
            "_health_with_status",
            return_value=(
                {
                    "connected": True,
                    "prometheus_version": "test-version",
                    "error": None,
                },
                200,
            ),
        ),
        patch.object(
            server,
            "_query_with_status",
            side_effect=[
                ({"resultType": "vector", "result": []}, 200),
                ({"resultType": "matrix", "result": []}, 200),
            ],
        ),
        patch.object(server, "_targets_with_status", return_value=([], 200)),
        patch.object(server, "_alerts_with_status", return_value=([], 200)),
    ):
        report = diagnostic.run_validation()

    assert report["final_status"] == "LIVE CONNECTION SUCCESSFUL"
    assert all(operation["success"] for operation in report["operations"])
    assert [operation["result_count"] for operation in report["operations"]] == [
        None,
        0,
        0,
        0,
        0,
    ]


def test_diagnostic_stops_after_authentication_failure(monkeypatch):
    _configure(monkeypatch)
    auth_error = server.PrometheusError(
        "Prometheus authentication failed (HTTP 401).",
        status_code=401,
        auth_schemes=["Basic"],
    )
    with (
        patch.object(server, "_health_with_status", side_effect=auth_error),
        patch.object(server, "_query_with_status") as query,
    ):
        report = diagnostic.run_validation()

    assert report["final_status"] == "AUTHENTICATION FAILED"
    assert len(report["operations"]) == 1
    assert report["operations"][0]["http_status"] == 401
    assert report["operations"][0]["authentication_challenge_schemes"] == ["Basic"]
    query.assert_not_called()


def test_diagnostic_does_not_report_unsupported_targets_as_zero(monkeypatch):
    _configure(monkeypatch)
    with (
        patch.object(
            server,
            "_health_with_status",
            return_value=(
                {"connected": True, "prometheus_version": "test", "error": None},
                200,
            ),
        ),
        patch.object(
            server,
            "_query_with_status",
            side_effect=[
                ({"resultType": "vector", "result": []}, 200),
                ({"resultType": "matrix", "result": []}, 200),
            ],
        ),
        patch.object(
            server,
            "_targets_with_status",
            side_effect=server.PrometheusError(
                "Prometheus returned HTTP 404.",
                status_code=404,
                response_content_type="application/json",
                response_detail="not found",
            ),
        ),
        patch.object(server, "_alerts_with_status", return_value=([], 200)),
    ):
        report = diagnostic.run_validation()

    target_result = report["operations"][3]
    assert report["final_status"] == "PARTIALLY VERIFIED"
    assert target_result["http_status"] == 404
    assert target_result["result_count"] is None
    assert target_result["safe_response_detail"] == "not found"
