"""Tests for the standalone Grafana MCP server using mocked HTTP responses."""

from unittest.mock import Mock, patch

import httpx
import pytest

from app.mcp_servers import grafana_mcp_server as server


TOKEN = "test-grafana-service-token"


@pytest.fixture
def configured_environment(monkeypatch):
    monkeypatch.setenv("GRAFANA_URL", "https://grafana.example")
    monkeypatch.setenv("GRAFANA_SERVICE_ACCOUNT_TOKEN", TOKEN)


def _response(payload, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "HTTP error",
            request=httpx.Request("GET", "https://grafana.example"),
            response=httpx.Response(
                status_code,
                request=httpx.Request("GET", "https://grafana.example"),
            ),
        )
    return response


def test_successful_dashboard_listing(configured_environment):
    dashboard = {
        "uid": "abc123",
        "title": "Service Overview",
        "url": "/d/abc123/service-overview",
        "folderTitle": "Operations",
        "type": "dash-db",
    }
    with patch.object(
        server.httpx,
        "get",
        return_value=_response([dashboard]),
    ) as get:
        result = server.list_dashboards()

    assert result == [
        {
            "uid": "abc123",
            "title": "Service Overview",
            "url": "/d/abc123/service-overview",
            "folder_title": "Operations",
            "type": "dash-db",
        }
    ]
    assert get.call_args.args[0] == "https://grafana.example/api/search"
    assert get.call_args.kwargs["params"] == {"type": "dash-db"}
    assert get.call_args.kwargs["headers"] == {"Authorization": f"Bearer {TOKEN}"}


def test_successful_dashboard_search(configured_environment):
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            [
                {
                    "uid": "payments",
                    "title": "Payments",
                    "url": "/d/payments/payments",
                    "folderTitle": "Services",
                    "type": "dash-db",
                }
            ]
        ),
    ) as get:
        result = server.search_dashboards("payments")

    assert len(result) == 1
    assert result[0]["title"] == "Payments"
    assert get.call_args.kwargs["params"] == {
        "type": "dash-db",
        "query": "payments",
    }


def test_successful_dashboard_retrieval(configured_environment):
    dashboard = {
        "uid": "abc123",
        "title": "Service Overview",
        "description": "Service status",
        "version": 4,
        "panels": [
            {
                "id": 1,
                "datasource": {"type": "prometheus", "uid": "prom-main"},
                "targets": [{"expr": "up"}],
            }
        ],
    }
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            {
                "dashboard": dashboard,
                "meta": {"folderTitle": "Operations", "folderUid": "folder1"},
            }
        ),
    ) as get:
        result = server.get_dashboard("abc123")

    assert result["title"] == "Service Overview"
    assert result["uid"] == "abc123"
    assert result["panels"] == dashboard["panels"]
    assert result["datasource_references"] == [
        {"type": "prometheus", "uid": "prom-main"}
    ]
    assert result["dashboard_configuration"] == dashboard
    assert get.call_args.args[0] == "https://grafana.example/api/dashboards/uid/abc123"


def test_successful_alert_rule_retrieval(configured_environment):
    rule = {
        "uid": "rule1",
        "title": "High CPU",
        "condition": "A",
        "labels": {"severity": "warning"},
        "annotations": {"summary": "CPU high"},
        "folderUID": "folder1",
        "folderTitle": "Operations",
        "intervalSeconds": 60,
        "ruleGroup": "system",
    }
    with patch.object(
        server.httpx,
        "get",
        return_value=_response([rule]),
    ):
        result = server.list_alert_rules()

    assert result == [
        {
            "uid": "rule1",
            "title": "High CPU",
            "condition": "A",
            "labels": {"severity": "warning"},
            "annotations": {"summary": "CPU high"},
            "folder_uid": "folder1",
            "folder_title": "Operations",
            "evaluation_interval": 60,
            "rule_group": "system",
        }
    ]


def test_successful_alert_instance_retrieval(configured_environment):
    alert = {
        "labels": {"alertname": "HighCPU"},
        "annotations": {"summary": "CPU is high"},
        "status": {"state": "active"},
        "startsAt": "2026-11-05T10:00:00Z",
        "fingerprint": "fingerprint-1",
    }
    with patch.object(server.httpx, "get", return_value=_response([alert])):
        result = server.get_alert_instances()

    assert result == [
        {
            "alert_name": "HighCPU",
            "state": "active",
            "labels": alert["labels"],
            "annotations": alert["annotations"],
            "active_at": alert["startsAt"],
            "fingerprint": "fingerprint-1",
        }
    ]


def test_successful_grafana_health(configured_environment):
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            {
                "database": "ok",
                "version": "11.2.0",
                "commit": "abc123",
            }
        ),
    ):
        result = server.get_grafana_health()

    assert result == {
        "connected": True,
        "database": "ok",
        "version": "11.2.0",
        "commit": "abc123",
        "error": None,
        "http_status": 200,
    }


def test_grafana_health_404_points_to_base_url_or_proxy_path(
    configured_environment,
):
    with patch.object(
        server.httpx,
        "get",
        return_value=_response({}, status_code=404),
    ):
        result = server.get_grafana_health()

    assert result["connected"] is False
    assert result["http_status"] == 404
    assert "GET /api/health" in result["error"]
    assert "GRAFANA_URL" in result["error"]
    assert "reverse-proxy subpath" in result["error"]
    assert "active stack" in result["error"]


def test_missing_configuration_is_reported_without_request(monkeypatch):
    monkeypatch.delenv("GRAFANA_URL", raising=False)
    monkeypatch.delenv("GRAFANA_SERVICE_ACCOUNT_TOKEN", raising=False)
    with patch.object(server.httpx, "get") as get:
        with pytest.raises(server.GrafanaError, match="GRAFANA_URL"):
            server.list_dashboards()
        result = server.get_grafana_health()

    assert result["connected"] is False
    assert "GRAFANA_URL" in result["error"]
    get.assert_not_called()


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "authentication failed"),
        (403, "access was denied"),
        (404, "not found"),
        (429, "rate limit"),
        (500, "server error"),
    ],
)
def test_http_errors_are_safe_and_status_specific(
    configured_environment,
    status_code,
    message,
):
    with patch.object(
        server.httpx,
        "get",
        return_value=_response({}, status_code=status_code),
    ):
        with pytest.raises(server.GrafanaError, match=message) as error:
            server.list_dashboards()

    assert error.value.status_code == status_code
    assert TOKEN not in str(error.value)


def test_dashboard_not_found_is_reported(configured_environment):
    with patch.object(
        server.httpx,
        "get",
        return_value=_response({}, status_code=404),
    ):
        with pytest.raises(server.GrafanaError, match="not found"):
            server.get_dashboard("missing")


def test_connection_timeout_is_reported_safely(configured_environment):
    with patch.object(
        server.httpx,
        "get",
        side_effect=httpx.ReadTimeout("timed out"),
    ):
        with pytest.raises(server.GrafanaError, match="timed out") as error:
            server.list_dashboards()

    assert TOKEN not in str(error.value)


def test_malformed_api_response_is_rejected(configured_environment):
    with patch.object(server.httpx, "get", return_value=_response({"invalid": True})):
        with pytest.raises(server.GrafanaError, match="invalid dashboard search"):
            server.list_dashboards()


def test_token_is_redacted_from_api_values(configured_environment):
    with patch.object(
        server.httpx,
        "get",
        return_value=_response(
            [
                {
                    "uid": "safe",
                    "title": TOKEN,
                    "url": "/dashboard",
                    "folderTitle": "Ops",
                    "type": "dash-db",
                }
            ]
        ),
    ):
        result = server.list_dashboards()

    assert TOKEN not in repr(result)


def test_tool_and_resource_registration(configured_environment):
    assert server.mcp.name == "Grafana Server"
    assert set(server.mcp._tool_manager._tools) == {
        "list_dashboards",
        "search_dashboards",
        "get_dashboard",
        "list_alert_rules",
        "get_alert_instances",
        "get_grafana_health",
    }
    assert set(server.mcp._resource_manager._resources) == {"grafana://available"}
    resource = server.available_grafana()
    assert resource["configuration"]["service_account_token_configured"] is True
    assert resource["configuration"]["authentication_method"] == "bearer"
    assert TOKEN not in repr(resource)


def test_empty_results_and_invalid_inputs(configured_environment):
    with patch.object(server.httpx, "get", return_value=_response([])):
        assert server.list_dashboards() == []
        assert server.search_dashboards("no matches") == []
        assert server.list_alert_rules() == []
        assert server.get_alert_instances() == []

    with pytest.raises(ValueError, match="query"):
        server.search_dashboards(" ")
    with pytest.raises(ValueError, match="uid"):
        server.get_dashboard(" ")
