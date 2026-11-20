"""Tests for the Prometheus MCP server using mocked HTTP responses."""

from unittest.mock import Mock, patch

import httpx
import pytest

from app.mcp_servers import prometheus_mcp_server as server


TOKEN = "test-prometheus-secret"


@pytest.fixture
def configured_environment(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example")
    monkeypatch.setenv("PROMETHEUS_BEARER_TOKEN", TOKEN)
    monkeypatch.setenv("PROMETHEUS_USERNAME", "")


def _response(payload, status_code=200, headers=None):
    response = Mock()
    response.status_code = status_code
    response.headers = headers or {}
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "HTTP error",
            request=httpx.Request("GET", "http://prometheus.example"),
            response=httpx.Response(
                status_code,
                headers=headers,
                json=payload,
                request=httpx.Request("GET", "http://prometheus.example"),
            ),
        )
    return response


def test_successful_instant_query(configured_environment):
    payload = {
        "status": "success",
        "data": {"resultType": "vector", "result": [{"value": [1, "2"]}]},
    }
    with patch.object(server.httpx, "get", return_value=_response(payload)) as get:
        result = server.query_prometheus("up", time="123")

    assert result == payload["data"]
    assert get.call_args.args[0] == "http://prometheus.example/api/v1/query"
    assert get.call_args.kwargs["params"] == {"query": "up", "time": "123"}
    assert get.call_args.kwargs["headers"] == {"Authorization": f"Bearer {TOKEN}"}


def test_empty_bearer_token_uses_no_authentication_header(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example")
    monkeypatch.setenv("PROMETHEUS_BEARER_TOKEN", "")
    monkeypatch.setenv("PROMETHEUS_USERNAME", "")
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            {"status": "success", "data": {"resultType": "vector", "result": []}}
        ),
    ) as get:
        server.query_prometheus("up")

    assert get.call_args.kwargs["headers"] == {}


def test_username_and_token_use_http_basic_auth(monkeypatch):
    username = "prometheus-instance"
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example")
    monkeypatch.setenv("PROMETHEUS_BEARER_TOKEN", TOKEN)
    monkeypatch.setenv("PROMETHEUS_USERNAME", username)
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            {"status": "success", "data": {"resultType": "vector", "result": []}}
        ),
    ) as get:
        server.query_prometheus("up")

    auth = get.call_args.kwargs["auth"]
    assert isinstance(auth, httpx.BasicAuth)
    request = next(auth.auth_flow(httpx.Request("GET", "http://prometheus.example")))
    assert request.headers["Authorization"].startswith("Basic ")
    assert TOKEN not in request.headers["Authorization"]
    assert username not in repr(server.available_prometheus())


def test_successful_range_query(configured_environment):
    payload = {
        "status": "success",
        "data": {"resultType": "matrix", "result": [{"values": [[1, "2"]]}]},
    }
    with patch.object(server.httpx, "get", return_value=_response(payload)) as get:
        result = server.query_prometheus_range(
            "up", "100", "200", "15s"
        )

    assert result == payload["data"]
    assert get.call_args.args[0] == (
        "http://prometheus.example/api/v1/query_range"
    )
    assert get.call_args.kwargs["params"] == {
        "query": "up",
        "start": "100",
        "end": "200",
        "step": "15s",
    }


def test_successful_target_retrieval(configured_environment):
    target = {
        "health": "up",
        "scrapeUrl": "http://node:9100/metrics",
        "lastScrape": "2026-11-05T10:00:00Z",
        "lastScrapeDuration": 0.02,
        "lastError": "",
        "labels": {"job": "node"},
    }
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            {"status": "success", "data": {"activeTargets": [target]}}
        ),
    ):
        result = server.get_prometheus_targets()

    assert result == [target]


def test_successful_alert_retrieval(configured_environment):
    alert = {
        "labels": {"alertname": "HighCPU", "severity": "warning"},
        "annotations": {"summary": "CPU load is high"},
        "state": "firing",
        "activeAt": "2026-11-05T10:00:00Z",
    }
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            {"status": "success", "data": {"alerts": [alert]}}
        ),
    ):
        result = server.get_prometheus_alerts()

    assert result == [
        {
            "alert_name": "HighCPU",
            "state": "firing",
            "labels": alert["labels"],
            "annotations": alert["annotations"],
            "active_at": alert["activeAt"],
        }
    ]


def test_successful_health_check(configured_environment):
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            {"status": "success", "data": {"version": "2.54.1"}}
        ),
    ):
        result = server.get_prometheus_health()

    assert result == {
        "connected": True,
        "prometheus_version": "2.54.1",
        "error": None,
    }


def test_missing_url_is_reported_without_request(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_URL", raising=False)
    monkeypatch.setenv("PROMETHEUS_BEARER_TOKEN", TOKEN)
    with patch.object(server.httpx, "get") as get:
        with pytest.raises(server.PrometheusError, match="PROMETHEUS_URL"):
            server.query_prometheus("up")
        health = server.get_prometheus_health()

    assert health["connected"] is False
    assert "PROMETHEUS_URL" in health["error"]
    get.assert_not_called()


def test_authentication_failure_does_not_expose_token(configured_environment):
    with patch.object(
        server.httpx,
        "get",
        return_value=_response({}, status_code=401),
    ):
        with pytest.raises(server.PrometheusError) as error:
            server.query_prometheus("up")

    assert "authentication failed" in str(error.value)
    assert TOKEN not in str(error.value)


def test_http_error_details_redact_configured_credentials(monkeypatch):
    username = "private-prometheus-user"
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example")
    monkeypatch.setenv("PROMETHEUS_BEARER_TOKEN", TOKEN)
    monkeypatch.setenv("PROMETHEUS_USERNAME", username)
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            {"errorType": "not_found", "error": f"missing {TOKEN} {username}"},
            status_code=404,
            headers={
                "content-type": "application/json",
                "www-authenticate": 'Basic realm="metrics"',
            },
        ),
    ):
        with pytest.raises(server.PrometheusError) as error:
            server.query_prometheus("up")

    assert error.value.status_code == 404
    assert error.value.auth_schemes == ["Basic"]
    assert error.value.response_content_type == "application/json"
    assert TOKEN not in error.value.response_detail
    assert username not in error.value.response_detail


def test_connection_timeout_is_reported_safely(configured_environment):
    with patch.object(
        server.httpx,
        "get",
        side_effect=httpx.ReadTimeout("request timed out"),
    ):
        with pytest.raises(server.PrometheusError, match="timed out") as error:
            server.query_prometheus("up")

    assert TOKEN not in str(error.value)


def test_prometheus_api_error_handles_invalid_promql_and_redacts_token(
    configured_environment,
):
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            {
                "status": "error",
                "errorType": "bad_data",
                "error": f"invalid query; {TOKEN}",
            }
        ),
    ):
        with pytest.raises(server.PrometheusError) as error:
            server.query_prometheus("invalid{")

    assert "bad_data" in str(error.value)
    assert TOKEN not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"status": "success"},
        {"status": "success", "data": {"resultType": "vector"}},
    ],
)
def test_invalid_query_response_is_rejected(configured_environment, payload):
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(payload),
    ):
        with pytest.raises(server.PrometheusError, match="invalid"):
            server.query_prometheus("up")


def test_invalid_json_response_is_rejected(configured_environment):
    response = _response({})
    response.json.side_effect = ValueError("invalid JSON")
    with patch.object(server.httpx, "get", return_value=response):
        with pytest.raises(server.PrometheusError, match="invalid JSON"):
            server.query_prometheus("up")


def test_token_in_success_response_is_redacted(configured_environment):
    payload = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [{"metric": {"instance": TOKEN}, "value": [1, "2"]}],
        },
    }
    with patch.object(server.httpx, "get", return_value=_response(payload)):
        result = server.query_prometheus("up")

    assert TOKEN not in repr(result)


def test_tools_and_resource_are_registered(configured_environment):
    assert server.mcp.name == "Prometheus Server"
    assert set(server.mcp._tool_manager._tools) == {
        "query_prometheus",
        "query_prometheus_range",
        "get_prometheus_targets",
        "get_prometheus_alerts",
        "get_prometheus_health",
    }
    assert set(server.mcp._resource_manager._resources) == {
        "prometheus://available"
    }
    resource = server.available_prometheus()
    assert resource["connection_configuration"] == {
        "url_configured": True,
        "url_valid": True,
        "bearer_token_configured": True,
        "username_configured": False,
        "authentication_method": "bearer",
    }
    assert TOKEN not in repr(resource)


def test_range_query_validates_time_and_step(configured_environment):
    with pytest.raises(ValueError, match="later than start"):
        server.query_prometheus_range("up", "200", "100", "15s")
    with pytest.raises(ValueError, match="step"):
        server.query_prometheus_range("up", "100", "200", "0s")
    with pytest.raises(ValueError, match="timestamp"):
        server.query_prometheus_range("up", "not-a-time", "200", "15s")
    with pytest.raises(ValueError, match="later than start"):
        server.query_prometheus_range(
            "up",
            "2026-11-05T00:00:00Z",
            "1793836799",
            "15s",
        )
