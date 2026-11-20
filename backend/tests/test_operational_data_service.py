"""Focused tests for data returned by the existing local MCP datasets."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import app.services.operational_data_service as operational_service
from app.services.operational_data_service import OperationalDataService


def test_log_query_uses_filters_and_returns_only_safe_fields(monkeypatch):
    payload = {
        "records": [
            {
                "dataset": "payment_api_logs.csv",
                "timestamp": "2026-10-01T00:10:00Z",
                "request_id": "request-1",
                "level": "ERROR",
                "method": "POST",
                "endpoint": "/checkout",
                "http_status": 500,
                "outcome": "error",
                "api_token": "must-not-be-returned",
            }
        ],
        "total": 1,
        "excluded_untimed_count": 2,
        "excluded_outside_range_count": 3,
    }
    calls = []

    async def fake_call_server_tools(server, requests):
        calls.append((server, requests))
        return SimpleNamespace(
            connected=True,
            errors=[],
            outputs=[SimpleNamespace(output=payload)],
        )

    monkeypatch.setattr(operational_service, "call_server_tools", fake_call_server_tools)

    result = asyncio.run(
        OperationalDataService().query_logs(
            service="payment-api",
            level="ERROR",
            search="checkout",
            start_time=datetime(2026, 10, 1, tzinfo=UTC),
            end_time=datetime(2026, 10, 2, tzinfo=UTC),
            limit=25,
            offset=5,
        )
    )

    assert calls[0][0] == "log"
    tool, arguments = calls[0][1][0]
    assert tool == "query_logs"
    assert arguments["service_name"] == "payment-api"
    assert arguments["keyword"] == "checkout"
    assert arguments["limit"] == 25
    assert arguments["offset"] == 5
    assert result.logs[0].service == "payment-api"
    assert "api_token" not in result.logs[0].raw
    assert result.excluded_untimed_count == 2
    assert result.excluded_outside_range_count == 3


def test_metrics_are_filtered_by_time_and_evaluation_labels_are_not_exposed(
    monkeypatch,
):
    records = [
        {
            "timestamp": "2026-10-01T00:10:00Z",
            "service_name": "payment-api",
            "cpu_usage_percent": 0,
            "request_count": 0,
            "error_count": 0,
            "scenario_label": "offline-only",
            "is_anomaly": True,
        },
        {
            "timestamp": "2026-10-02T00:10:00Z",
            "service_name": "payment-api",
            "cpu_usage_percent": 80,
            "request_count": 10,
            "error_count": 2,
        },
    ]
    calls = []

    async def fake_call_server_tools(server, requests):
        calls.append((server, requests))
        return SimpleNamespace(
            connected=True,
            errors=[],
            outputs=[SimpleNamespace(output={"result": records})],
        )

    monkeypatch.setattr(operational_service, "call_server_tools", fake_call_server_tools)

    result = asyncio.run(
        OperationalDataService().query_metrics(
            service="payment-api",
            start_time=datetime(2026, 10, 1, tzinfo=UTC),
            end_time=datetime(2026, 10, 1, 23, 59, tzinfo=UTC),
        )
    )

    assert calls[0][1][0][0] == "get_metrics_by_service"
    assert len(result.metrics) == 1
    assert result.metrics[0].cpu_usage_percent == 0
    assert not hasattr(result.metrics[0], "scenario_label")
    assert result.summary.total_requests == 0


def test_metrics_mcp_probe_counts_records_from_structured_result_envelope(
    monkeypatch,
):
    class FakeClient:
        connections = {}

        async def connect_all(self):
            return {
                "metrics": {
                    "connected": True,
                    "tools_count": 8,
                    "resources_count": 1,
                }
            }

        async def discover_all_tools(self):
            return {}

        async def call_tool(self, name, tool, arguments):
            assert name == "metrics"
            assert tool == "get_latest_metrics"
            assert arguments == {"limit": 500}
            return {
                "structured_content": {
                    "result": [{"timestamp": "2026-10-01T00:00:00Z"}] * 500
                }
            }

        async def close(self):
            return None

    monkeypatch.setattr(
        operational_service,
        "load_server_configs",
        lambda: {"metrics": object()},
    )
    monkeypatch.setattr(
        operational_service,
        "MultiServerMCPClient",
        lambda _configs: FakeClient(),
    )

    result = asyncio.run(OperationalDataService().mcp_status())

    assert len(result) == 1
    assert result[0].status == "healthy"
    assert result[0].records_available == 500
