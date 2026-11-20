"""Tests for the standalone log MCP server using local log datasets."""

import pytest

from app.mcp_servers import log_mcp_server as server


def test_mcp_server_imports_with_registered_tools_and_resource():
    assert server.mcp.name == "Incident Log Server"
    assert set(server.mcp._tool_manager._tools) == {
        "get_recent_logs",
        "query_logs",
        "search_logs",
        "filter_logs_by_level",
        "get_error_logs",
    }
    assert set(server.mcp._resource_manager._resources) == {"logs://available"}
    assert callable(server.get_recent_logs)
    assert callable(server.query_logs)
    assert callable(server.search_logs)
    assert callable(server.filter_logs_by_level)
    assert callable(server.get_error_logs)
    assert callable(server.available_logs)
    assert not hasattr(server, "get_logs_by_service")


def test_recent_logs_return_newest_records():
    results = server.get_recent_logs(limit=5)

    assert len(results) == 5
    assert all("dataset" in record for record in results)
    timestamps = [server._record_time(record) for record in results]
    assert timestamps == sorted(timestamps, reverse=True)


def test_query_logs_filters_service_and_incident_window():
    result = server.query_logs(
        service_name="payment-api",
        start_time="2026-10-01T00:00:00Z",
        end_time="2026-10-01T01:00:00Z",
        limit=10,
    )

    assert result["total"] > 0
    assert len(result["records"]) <= 10
    assert all(record["dataset"] == "payment_api_logs.csv" for record in result["records"])
    assert result["excluded_untimed_count"] == 0


def test_query_logs_excludes_unrelated_and_out_of_window_records():
    result = server.query_logs(
        service_name="payment-api",
        start_time="2026-11-02T11:15:00Z",
        end_time="2026-11-02T12:04:00Z",
        limit=20,
    )

    assert result["records"] == []
    assert result["total"] == 0
    assert result["excluded_outside_range_count"] > 0


def test_query_logs_paginates_and_validates_time_ranges():
    first = server.query_logs(limit=5, offset=0)
    second = server.query_logs(limit=5, offset=5)

    assert first["total"] == 7000
    assert len(first["records"]) == len(second["records"]) == 5
    assert {
        record["dataset"] + ":" + str(record.get("LineId") or record.get("request_id"))
        for record in first["records"]
    }.isdisjoint(
        record["dataset"] + ":" + str(record.get("LineId") or record.get("request_id"))
        for record in second["records"]
    )
    with pytest.raises(ValueError, match="at or after"):
        server.query_logs(
            start_time="2026-11-03T00:00:00Z",
            end_time="2026-11-02T00:00:00Z",
        )


def test_keyword_search_matches_records_across_datasets():
    results = server.search_logs("mod_jk", limit=5)

    assert results
    assert all(
        "mod_jk" in " ".join(record.values()).casefold()
        for record in results
    )


def test_severity_filter_uses_available_case_insensitive_levels():
    results = server.filter_logs_by_level("warn", limit=10)

    assert results
    assert all(record["level"].upper() == "WARN" for record in results)


def test_error_log_retrieval_returns_error_related_records():
    results = server.get_error_logs(limit=10)

    assert results
    assert all(
        record.get("Level", "").upper() == "ERROR"
        or record.get("level", "").upper() == "ERROR"
        or record.get("outcome", "").casefold() in {"error", "failed", "failure"}
        or record.get("error_code")
        or record.get("error_message")
        or int(record.get("http_status", "0") or 0) >= 500
        or "error" in record.get("Content", "").casefold()
        for record in results
    )


def test_available_resource_lists_real_datasets_fields_and_counts():
    resource = server.available_logs()
    datasets = {dataset["name"]: dataset for dataset in resource["datasets"]}

    assert datasets["Apache_log.csv"]["record_count"] == 2000
    assert datasets["Apache_log.csv"]["fields"] == [
        "LineId",
        "Time",
        "Level",
        "Content",
        "EventId",
        "EventTemplate",
    ]
    assert datasets["payment_api_logs.csv"]["record_count"] == 5000
    assert "service" not in {
        field.casefold()
        for dataset in datasets.values()
        for field in dataset["fields"]
    }
    assert datasets["Apache_log.csv"]["raw_source"]["name"] == "Apache.log"
    assert datasets["payment_api_logs.csv"]["raw_source"]["format"] == "JSONL"


def test_empty_search_returns_no_results():
    assert server.search_logs("no-such-log-value-123", limit=5) == []


def test_unsupported_severity_reports_available_values():
    with pytest.raises(ValueError, match="Available levels: ERROR, INFO, NOTICE, WARN"):
        server.filter_logs_by_level("TRACE")