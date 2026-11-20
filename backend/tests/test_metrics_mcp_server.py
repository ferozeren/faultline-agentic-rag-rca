"""Tests for the standalone metrics MCP server using the local CSV dataset."""

from app.mcp_servers import metrics_mcp_server as server


EVALUATION_FIELDS = {"scenario_label", "is_anomaly"}


def _service_name() -> str:
    return server.available_metrics()["available_services"][0]


def test_mcp_server_imports_with_registered_tools_and_resource():
    assert server.mcp.name == "Incident Metrics Server"
    assert set(server.mcp._tool_manager._tools) == {
        "get_latest_metrics",
        "get_metrics_by_service",
        "get_cpu_usage",
        "get_memory_usage",
        "get_latency_metrics",
        "get_error_metrics",
        "get_database_connections",
        "get_service_summary",
    }
    assert set(server.mcp._resource_manager._resources) == {"metrics://available"}


def test_latest_metrics_return_newest_records():
    results = server.get_latest_metrics(limit=5)

    assert len(results) == 5
    timestamps = [record["timestamp"] for record in results]
    assert timestamps == sorted(timestamps, reverse=True)


def test_metrics_by_service_filters_records():
    service_name = _service_name()

    results = server.get_metrics_by_service(service_name, limit=5)

    assert results
    assert all(record["service_name"] == service_name for record in results)


def test_cpu_usage_returns_only_cpu_fields():
    results = server.get_cpu_usage(service_name=_service_name(), limit=5)

    assert results
    assert set(results[0]) == {"timestamp", "service_name", "cpu_usage_percent"}


def test_memory_usage_returns_only_memory_fields():
    results = server.get_memory_usage(service_name=_service_name(), limit=5)

    assert results
    assert set(results[0]) == {
        "timestamp",
        "service_name",
        "memory_usage_percent",
    }


def test_latency_metrics_return_only_latency_fields():
    results = server.get_latency_metrics(service_name=_service_name(), limit=5)

    assert results
    assert set(results[0]) == {"timestamp", "service_name", "avg_latency_ms"}


def test_error_metrics_return_request_and_error_counts():
    results = server.get_error_metrics(service_name=_service_name(), limit=5)

    assert results
    assert set(results[0]) == {
        "timestamp",
        "service_name",
        "request_count",
        "error_count",
    }


def test_database_connections_return_only_connection_fields():
    results = server.get_database_connections(
        service_name=_service_name(),
        limit=5,
    )

    assert results
    assert set(results[0]) == {"timestamp", "service_name", "db_connections"}


def test_service_summary_calculates_supported_metrics():
    service_name = _service_name()
    records = server.get_metrics_by_service(service_name, limit=500)
    summary = server.get_service_summary(service_name)

    assert summary["record_count"] == len(records)
    assert summary["average_cpu_usage_percent"] == round(
        sum(float(record["cpu_usage_percent"]) for record in records) / len(records),
        2,
    )
    assert summary["maximum_cpu_usage_percent"] == max(
        float(record["cpu_usage_percent"]) for record in records
    )
    assert summary["average_memory_usage_percent"] == round(
        sum(float(record["memory_usage_percent"]) for record in records)
        / len(records),
        2,
    )
    assert summary["maximum_memory_usage_percent"] == max(
        float(record["memory_usage_percent"]) for record in records
    )
    assert summary["average_latency_ms"] == round(
        sum(float(record["avg_latency_ms"]) for record in records) / len(records),
        2,
    )
    assert summary["maximum_latency_ms"] == max(
        float(record["avg_latency_ms"]) for record in records
    )
    assert summary["total_requests"] == sum(
        int(record["request_count"]) for record in records
    )
    assert summary["total_errors"] == sum(
        int(record["error_count"]) for record in records
    )
    assert summary["maximum_db_connections"] == max(
        int(record["db_connections"]) for record in records
    )


def test_available_resource_describes_dataset_without_evaluation_fields():
    resource = server.available_metrics()

    assert resource["dataset_name"] == "metrics.csv"
    assert resource["record_count"] == 500
    assert resource["available_services"] == [
        "auth-service",
        "database-service",
        "order-service",
        "payment-api",
    ]
    assert resource["timestamp_range"] == {
        "start": "2026-11-02 10:00:00",
        "end": "2026-11-02 12:04:00",
    }
    assert resource["fields"] == list(server.OPERATIONAL_FIELDS)
    assert resource["supported_metrics"] == list(server.OPERATIONAL_FIELDS[2:])
    assert not EVALUATION_FIELDS.intersection(resource["fields"])


def test_evaluation_fields_are_never_returned_by_tools():
    records = server.get_latest_metrics(limit=500)
    assert records
    assert all(not EVALUATION_FIELDS.intersection(record) for record in records)

    summary = server.get_service_summary(_service_name())
    assert not EVALUATION_FIELDS.intersection(summary)


def test_empty_results_are_handled():
    assert server.get_metrics_by_service("missing-service", limit=5) == []
    assert server.get_cpu_usage("missing-service", limit=5) == []
    assert server.get_memory_usage("missing-service", limit=5) == []
    assert server.get_latency_metrics("missing-service", limit=5) == []
    assert server.get_error_metrics("missing-service", limit=5) == []
    assert server.get_database_connections("missing-service", limit=5) == []
    assert server.get_service_summary("missing-service") == {
        "service_name": "missing-service",
        "record_count": 0,
    }
