"""Unit tests for specialist evidence collection without external services."""

import asyncio
from datetime import UTC, datetime

import pytest
from app.agents import (
    grafana_analysis_agent as grafana_agent,
)
from app.agents import (
    historical_rag_agent as rag_agent,
)
from app.agents import (
    log_analysis_agent as log_agent,
)
from app.agents import mcp_tool_access
from app.agents import (
    metrics_analysis_agent as metrics_agent,
)
from app.agents import (
    prometheus_analysis_agent as prometheus_agent,
)
from app.agents.mcp_tool_access import (
    ToolBatchResult,
    ToolInvocation,
    ToolInvocationError,
)
from app.graph.state import create_initial_state
from app.schemas.evidence import EvidenceClassification, EvidenceSourceType
from app.schemas.incident import IncidentRequest


def _state(**updates):
    state = create_initial_state(
        IncidentRequest(
            description="payment-api timeout latency errors",
            affected_service="payment-api",
            start_time="2026-10-31T23:00:00Z",
            end_time="2026-10-31T23:30:00Z",
        )
    )
    state.update(updates)
    return state


def _batch(*outputs, connected=True, errors=(), unavailable=(), connection_error=None):
    return ToolBatchResult(
        connected=connected,
        outputs=tuple(outputs),
        errors=tuple(errors),
        unavailable_tools=tuple(unavailable),
        connection_error=connection_error,
    )


def _run(awaitable):
    return asyncio.run(awaitable)


def test_historical_rag_extracts_sources_and_marks_evidence_historical(monkeypatch):
    document = {
        "chunk_id": "INC-8_chunk_0",
        "text": "A prior incident showed database saturation.",
        "metadata": {"incident_id": "INC-8", "service": "payment-api"},
        "retrieval_method": "hybrid_rrf",
    }
    calls = []

    def answer_question(query, *, llm=None):
        calls.append((query, llm))
        return {
            "answer": "Prior incidents involved database saturation.",
            "sources": [{"chunk_id": "INC-8_chunk_0", "incident_id": "INC-8"}],
            "retrieved_documents": [document],
        }

    monkeypatch.setattr(rag_agent.rag_chain, "answer_question", answer_question)
    fake_llm = object()

    update = rag_agent.analyze_historical_incidents(_state(), llm=fake_llm)

    evidence = update["historical_evidence"][0]
    assert calls == [
        (
            "payment-api timeout latency errors\nAffected service: payment-api",
            fake_llm,
        )
    ]
    assert evidence.classification is EvidenceClassification.HISTORICAL
    assert evidence.source_type is EvidenceSourceType.HISTORICAL_RAG
    assert evidence.raw_reference == "INC-8_chunk_0"
    assert evidence.metadata["metadata"]["incident_id"] == "INC-8"
    assert update["historical_incident_matches"] == [document]
    assert "not proof" in update["rag_observations"][0].summary
    assert update["execution_metadata"]


def test_historical_rag_empty_retrieval_returns_gap_without_failure(monkeypatch):
    monkeypatch.setattr(
        rag_agent.rag_chain,
        "answer_question",
        lambda query, llm=None: {
            "answer": "No historical documents.",
            "sources": [],
            "retrieved_documents": [],
        },
    )

    update = rag_agent.analyze_historical_incidents(_state())

    assert update["historical_evidence"] == []
    assert update["evidence_gaps"]
    assert update["errors"] == []
    assert (
        update["execution_metadata"][next(iter(update["execution_metadata"]))]["status"]
        == "no_evidence"
    )


def test_historical_rag_exception_is_safe_and_does_not_escape(monkeypatch):
    monkeypatch.setattr(
        rag_agent.rag_chain,
        "answer_question",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("credential-value-must-not-leak")
        ),
    )

    update = rag_agent.analyze_historical_incidents(_state())

    assert update["historical_evidence"] == []
    assert update["errors"] == ["historical_rag failed (RuntimeError)."]
    assert "credential-value" not in repr(update)
    assert update["evidence_gaps"]


def test_log_agent_collects_error_and_search_results_with_source_references(
    monkeypatch,
):
    record = {
        "dataset": "service.log",
        "timestamp": "2026-10-31T23:10:00Z",
        "level": "ERROR",
        "request_id": "request-17",
        "trace_id": "trace-17",
        "outcome": "failed",
    }
    calls = []

    async def call_server_tools(server, requests):
        calls.append((server, requests))
        return _batch(
            ToolInvocation("get_recent_logs", [record]),
            ToolInvocation("get_error_logs", [record]),
        )

    monkeypatch.setattr(log_agent, "call_server_tools", call_server_tools)

    update = _run(log_agent.analyze_logs(_state()))

    assert calls[0][0] == "log"
    assert {name for name, _ in calls[0][1]} >= {
        "get_recent_logs",
        "get_error_logs",
        "filter_logs_by_level",
        "search_logs",
    }
    assert len(update["log_evidence"]) == 1
    evidence = update["log_evidence"][0]
    assert evidence.source_type is EvidenceSourceType.APPLICATION_LOGS
    assert evidence.classification is EvidenceClassification.CURRENT
    assert evidence.service is None
    assert evidence.raw_reference == "service.log:trace-17"
    assert evidence.observed_at == datetime(2026, 10, 31, 23, 10, tzinfo=UTC)
    assert update["observations"][0].evidence_ids == [evidence.evidence_id]


def test_log_agent_filters_environment_and_summarizes_safe_error_fields(monkeypatch):
    production = {
        "dataset": "payment_api_logs.csv",
        "service": "payment-api",
        "environment": "production",
        "timestamp": "2026-10-31T23:10:00Z",
        "level": "ERROR",
        "outcome": "failed",
        "event": "payment_confirm",
        "endpoint": "/api/v1/payments/12345/confirm?token=private-token",
        "http_status": "504",
        "error_code": "GATEWAY_TIMEOUT",
        "latency_ms": "3669",
        "retry_count": "2",
        "gateway": "adyen_mock",
        "user_id": "private-user",
        "source_ip": "192.0.2.1",
    }
    staging = {**production, "environment": "staging", "request_id": "staging-1"}
    state = _state()
    state["environment"] = "production"

    async def call_server_tools(_server, _requests):
        return _batch(ToolInvocation("get_recent_logs", [production, staging]))

    monkeypatch.setattr(log_agent, "call_server_tools", call_server_tools)

    update = _run(log_agent.analyze_logs(state))

    assert len(update["log_evidence"]) == 1
    description = update["log_evidence"][0].description
    assert "http_status=504" in description
    assert "error_code=GATEWAY_TIMEOUT" in description
    assert "latency_ms=3669" in description
    assert "retry_count=2" in description
    assert "endpoint=/api/v1/payments/{id}/confirm" in description
    assert all(
        private_value not in description
        for private_value in ("private-token", "private-user", "192.0.2.1")
    )
    assert any("environment did not match" in item for item in update["warnings"])


def test_log_agent_uses_dataset_identity_not_incident_service(monkeypatch):
    record = {
        "dataset": "Apache_log.csv",
        "Time": "Sun Dec 04 04:47:44 2005",
        "Level": "ERROR",
        "Content": "mod_jk child workerEnv in error state 6",
    }

    async def call_server_tools(server, requests):
        return _batch(ToolInvocation("get_recent_logs", [record]))

    monkeypatch.setattr(log_agent, "call_server_tools", call_server_tools)

    update = _run(
        log_agent.analyze_logs(
            _state(start_time=None, end_time=None),
        )
    )

    evidence = update["log_evidence"][0]
    assert evidence.service is None
    assert evidence.observed_at == datetime(2005, 12, 4, 4, 47, 44, tzinfo=UTC)


def test_log_agent_requests_incident_window_and_rejects_stale_records(monkeypatch):
    record = {
        "dataset": "Apache_log.csv",
        "Time": "Sun Dec 04 04:47:44 2005",
        "Level": "ERROR",
        "Content": "mod_jk child workerEnv in error state 6",
    }
    calls = []

    async def call_server_tools(server, requests):
        calls.append(requests)
        return _batch(
            ToolInvocation(
                "query_logs",
                {
                    "records": [record],
                    "total": 1,
                    "excluded_untimed_count": 0,
                    "excluded_outside_range_count": 0,
                },
            )
        )

    monkeypatch.setattr(log_agent, "call_server_tools", call_server_tools)

    update = _run(log_agent.analyze_logs(_state()))

    query = next(args for name, args in calls[0] if name == "query_logs")
    assert query["service_name"] == "payment-api"
    assert query["start_time"] == "2026-10-31T23:00:00+00:00"
    assert update["log_evidence"] == []
    assert update["evidence_gaps"]


def test_log_agent_empty_result_is_no_evidence(monkeypatch):
    async def call_server_tools(server, requests):
        return _batch(
            ToolInvocation("get_recent_logs", []),
            ToolInvocation("get_error_logs", []),
        )

    monkeypatch.setattr(log_agent, "call_server_tools", call_server_tools)

    update = _run(log_agent.analyze_logs(_state()))

    assert update["log_evidence"] == []
    assert update["evidence_gaps"]
    assert (
        update["execution_metadata"][next(iter(update["execution_metadata"]))]["status"]
        == "no_evidence"
    )


@pytest.mark.parametrize(
    ("batch", "expected_status"),
    [
        (
            _batch(
                connected=False,
                connection_error="MCP server connection is unavailable.",
            ),
            "failed",
        ),
        (
            _batch(
                connected=True,
                errors=(
                    ToolInvocationError(
                        "get_recent_logs",
                        "Tool invocation failed (MCPToolAdapterError).",
                    ),
                ),
            ),
            "failed",
        ),
    ],
)
def test_log_agent_server_and_tool_failures_are_structured(
    monkeypatch,
    batch,
    expected_status,
):
    async def call_server_tools(server, requests):
        return batch

    monkeypatch.setattr(log_agent, "call_server_tools", call_server_tools)

    update = _run(log_agent.analyze_logs(_state()))

    assert update["errors"]
    assert update["evidence_gaps"]
    assert (
        update["execution_metadata"][next(iter(update["execution_metadata"]))]["status"]
        == expected_status
    )


def test_metrics_agent_preserves_zeros_and_excludes_evaluation_labels(monkeypatch):
    record = {
        "timestamp": "2026-10-31T23:10:00Z",
        "service_name": "payment-api",
        "cpu_usage_percent": "0",
        "memory_usage_percent": "",
        "avg_latency_ms": "10",
        "request_count": "100",
        "error_count": "5",
        "scenario_label": "known_failure",
        "is_anomaly": True,
    }
    summary = {"service_name": "payment-api", "record_count": 1, "total_errors": 0}
    calls = []

    async def call_server_tools(server, requests):
        calls.append((server, requests))
        return _batch(
            ToolInvocation("get_metrics_by_service", [record]),
            ToolInvocation("get_service_summary", summary),
        )

    monkeypatch.setattr(metrics_agent, "call_server_tools", call_server_tools)

    update = _run(metrics_agent.analyze_service_metrics(_state()))

    assert calls[0][0] == "metrics"
    assert len(update["metrics_evidence"]) == 1
    metric_evidence = next(
        item
        for item in update["metrics_evidence"]
        if "cpu_usage_percent" in item.metadata
    )
    assert metric_evidence.metadata["cpu_usage_percent"] == "0"
    assert "scenario_label" not in metric_evidence.metadata
    assert "is_anomaly" not in metric_evidence.metadata
    assert any("memory_usage_percent" in warning for warning in update["warnings"])
    assert any("zero-valued" in item.summary for item in update["observations"])
    assert any(
        "derived error rate=5.00%" in item.summary for item in update["observations"]
    )
    assert update["metrics_evidence"][0].classification is EvidenceClassification.CURRENT


def test_metrics_agent_filters_known_environment_mismatch(monkeypatch):
    production = {
        "timestamp": "2026-10-31T23:10:00Z",
        "service_name": "payment-api",
        "environment": "production",
        "cpu_usage_percent": 70,
        "memory_usage_percent": 60,
        "avg_latency_ms": 150,
        "request_count": 100,
        "error_count": 2,
        "db_connections": 10,
    }
    staging = {**production, "environment": "staging", "timestamp": "2026-10-31T23:11:00Z"}
    unscoped = {
        **production,
        "timestamp": "2026-10-31T23:12:00Z",
    }
    unscoped.pop("environment")
    state = _state()
    state["environment"] = "production"

    async def call_server_tools(_server, _requests):
        return _batch(
            ToolInvocation("get_metrics_by_service", [production, staging, unscoped])
        )

    monkeypatch.setattr(metrics_agent, "call_server_tools", call_server_tools)

    update = _run(metrics_agent.analyze_service_metrics(state))

    assert len(update["metrics_evidence"]) == 2
    assert any(
        item.metadata.get("environment") == "production"
        for item in update["metrics_evidence"]
    )
    assert any(
        "environment=not specified" in item.description
        for item in update["metrics_evidence"]
    )
    assert any("environment did not match" in item for item in update["warnings"])
    assert any("environment field" in item for item in update["warnings"])


def test_metrics_agent_empty_data_is_not_conflated_with_zero(monkeypatch):
    async def call_server_tools(server, requests):
        return _batch(
            ToolInvocation("get_metrics_by_service", []),
            ToolInvocation(
                "get_service_summary",
                {"service_name": "payment-api", "record_count": 0},
            ),
        )

    monkeypatch.setattr(metrics_agent, "call_server_tools", call_server_tools)

    update = _run(metrics_agent.analyze_service_metrics(_state()))

    assert update["metrics_evidence"] == []
    assert "not zero" in update["evidence_gaps"][0].description
    assert "not be interpreted as zero" in update["observations"][0].summary


def test_prometheus_agent_preserves_query_and_series_metadata(monkeypatch):
    query = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {"job": "payment-api"},
                    "value": [1793488200, "1"],
                }
            ],
        },
    }
    calls = []

    async def call_server_tools(server, requests, *, follow_up=None):
        target_output = ToolInvocation(
            "get_prometheus_targets",
            [{"health": "up", "labels": {"job": "payment-api"}}],
        )
        query_requests = follow_up((target_output,)) if follow_up else []
        calls.append((server, requests, query_requests))
        return _batch(
            target_output,
            ToolInvocation("get_prometheus_alerts", []),
            ToolInvocation("query_prometheus_range", query),
        )

    monkeypatch.setattr(prometheus_agent, "call_server_tools", call_server_tools)

    update = _run(prometheus_agent.analyze_prometheus(_state()))

    assert calls[0][0] == "prometheus"
    query_args = next(
        args for name, args in calls[0][2] if name == "query_prometheus_range"
    )
    assert query_args["query"] == 'up{job="payment-api"}'
    series_evidence = next(
        item for item in update["prometheus_evidence"] if "query" in item.metadata
    )
    assert series_evidence.metadata["query"] == 'up{job="payment-api"}'
    assert series_evidence.metadata["series"]["metric"]["job"] == "payment-api"
    assert series_evidence.service == "payment-api"
    assert series_evidence.source_type is EvidenceSourceType.PROMETHEUS


def test_prometheus_empty_series_are_distinct_from_failed_queries(monkeypatch):
    async def call_server_tools(server, requests, *, follow_up=None):
        query_requests = follow_up(()) if follow_up else []
        query_tool = query_requests[0][0]
        return _batch(
            ToolInvocation("get_prometheus_targets", []),
            ToolInvocation("get_prometheus_alerts", []),
            ToolInvocation(
                query_tool,
                {
                    "status": "success",
                    "data": {"resultType": "vector", "result": []},
                },
            ),
        )

    monkeypatch.setattr(prometheus_agent, "call_server_tools", call_server_tools)

    update = _run(prometheus_agent.analyze_prometheus(_state()))

    assert len(update["prometheus_evidence"]) == 1
    assert "zero returned series" in update["prometheus_evidence"][0].description
    assert any(
        "not proof of zero errors" in item.summary for item in update["observations"]
    )
    assert update["errors"] == []
    assert (
        update["execution_metadata"][next(iter(update["execution_metadata"]))]["status"]
        == "no_evidence"
    )


def test_prometheus_tool_error_is_not_reported_as_empty_result(monkeypatch):
    async def call_server_tools(server, requests, *, follow_up=None):
        query_requests = follow_up(()) if follow_up else []
        return _batch(
            ToolInvocation("get_prometheus_targets", []),
            ToolInvocation("get_prometheus_alerts", []),
            connected=True,
            errors=(
                ToolInvocationError(
                    query_requests[0][0],
                    "Tool invocation failed (MCPToolAdapterError).",
                ),
            ),
        )

    monkeypatch.setattr(prometheus_agent, "call_server_tools", call_server_tools)

    update = _run(prometheus_agent.analyze_prometheus(_state()))

    assert update["errors"]
    assert not any(
        "zero returned series" in item.description
        for item in update["prometheus_evidence"]
    )
    assert update["evidence_gaps"]


def test_prometheus_uses_incident_range_when_provided(monkeypatch):
    calls = []

    async def call_server_tools(server, requests, *, follow_up=None):
        calls.extend(requests)
        calls.extend(
            follow_up(
                (
                    ToolInvocation(
                        "get_prometheus_targets",
                        [{"labels": {"service_name": "payment-api"}}],
                    ),
                )
            )
        )
        return _batch(
            ToolInvocation(
                "get_prometheus_targets",
                [{"labels": {"service_name": "payment-api"}}],
            ),
            ToolInvocation(
                "query_prometheus_range",
                {
                    "status": "success",
                    "data": {"resultType": "matrix", "result": []},
                },
            ),
        )

    monkeypatch.setattr(prometheus_agent, "call_server_tools", call_server_tools)

    _run(prometheus_agent.analyze_prometheus(_state()))

    name, arguments = next(
        item for item in calls if item[0] == "query_prometheus_range"
    )
    assert name == "query_prometheus_range"
    assert arguments["query"] == 'up{service_name="payment-api"}'
    assert arguments["step"] == "60s"
    assert arguments["start"] == "2026-10-31T23:00:00+00:00"


def test_grafana_agent_retrieves_dashboard_panels_and_alerts(monkeypatch):
    search = [{"uid": "dash-1", "title": "Payment API"}]
    dashboard = {
        "uid": "dash-1",
        "title": "Payment API",
        "panels": [{"id": 7, "title": "Latency", "type": "timeseries"}],
    }
    first_outputs = (ToolInvocation("search_dashboards", search),)
    calls = []

    async def call_server_tools(server, requests, *, follow_up=None):
        calls.append((server, requests))
        follow_up_requests = follow_up(first_outputs) if follow_up else []
        assert follow_up_requests == [("get_dashboard", {"uid": "dash-1"})]
        return _batch(
            ToolInvocation("search_dashboards", search),
            ToolInvocation("get_dashboard", dashboard),
            ToolInvocation(
                "list_alert_rules", [{"uid": "rule-1", "title": "High error rate"}]
            ),
            ToolInvocation(
                "get_alert_instances",
                [
                    {
                        "fingerprint": "alert-1",
                        "alert_name": "HighLatency",
                        "state": "firing",
                        "active_at": "2026-10-31T23:15:00Z",
                    }
                ],
            ),
        )

    monkeypatch.setattr(grafana_agent, "call_server_tools", call_server_tools)

    update = _run(grafana_agent.analyze_grafana(_state()))

    assert calls[0][0] == "grafana"
    dashboard_evidence = next(
        item
        for item in update["grafana_evidence"]
        if item.raw_reference == "grafana://dashboard/dash-1"
    )
    assert dashboard_evidence.metadata["panel_references"][0]["reference"].endswith(
        "/panel/7"
    )
    assert "does not establish" in dashboard_evidence.description
    assert any(
        item.raw_reference == "grafana:alert-instance:alert-1"
        for item in update["grafana_evidence"]
    )
    assert all(
        item.classification is EvidenceClassification.CURRENT
        for item in update["grafana_evidence"]
    )


def test_grafana_empty_results_are_not_api_failures(monkeypatch):
    async def call_server_tools(server, requests, *, follow_up=None):
        return _batch(
            ToolInvocation("search_dashboards", []),
            ToolInvocation("list_alert_rules", []),
            ToolInvocation("get_alert_instances", []),
        )

    monkeypatch.setattr(grafana_agent, "call_server_tools", call_server_tools)

    update = _run(grafana_agent.analyze_grafana(_state()))

    assert update["grafana_evidence"] == []
    assert update["errors"] == []
    assert update["evidence_gaps"]
    assert any("no alert instances" in warning for warning in update["warnings"])


def test_grafana_api_failure_is_reported_without_secret_text(monkeypatch):
    async def call_server_tools(server, requests, *, follow_up=None):
        return _batch(
            connected=True,
            errors=(
                ToolInvocationError(
                    "get_alert_instances",
                    "Tool invocation failed (MCPToolAdapterError).",
                ),
            ),
        )

    monkeypatch.setattr(grafana_agent, "call_server_tools", call_server_tools)

    update = _run(grafana_agent.analyze_grafana(_state()))

    assert update["errors"]
    assert update["evidence_gaps"]
    assert update["grafana_evidence"] == []


def test_shared_mcp_access_reuses_one_server_session_for_all_tool_calls(monkeypatch):
    class FakeTool:
        def __init__(self, name):
            self.name = name
            self.calls = []

        async def ainvoke(self, arguments):
            self.calls.append(arguments)
            return {
                "server_name": "log",
                "structured_content": {"result": [{"arguments": arguments}]},
            }

    tools = [FakeTool("log__search_logs"), FakeTool("log__get_error_logs")]

    class FakeClient:
        def __init__(self, configs):
            self.configs = configs
            self.connect_calls = 0
            self.close_calls = 0

        async def connect_all(self):
            self.connect_calls += 1
            return {"log": {"connected": True}}

        async def close(self):
            self.close_calls += 1

    clients = []

    def create_client(configs):
        client = FakeClient(configs)
        clients.append(client)
        return client

    class FakeAdapter:
        def __init__(self, client, *, secrets=()):
            self.client = client

        async def get_langchain_tools(self):
            return tools

    monkeypatch.setattr(
        mcp_tool_access,
        "load_server_configs",
        lambda: {"log": object(), "metrics": object()},
    )
    monkeypatch.setattr(mcp_tool_access, "MultiServerMCPClient", create_client)
    monkeypatch.setattr(mcp_tool_access, "LangChainMCPAdapter", FakeAdapter)

    result = _run(
        mcp_tool_access.call_server_tools(
            "log",
            [
                ("search_logs", {"keyword": "timeout"}),
                ("get_error_logs", {"limit": 4}),
            ],
            follow_up=lambda outputs: [("search_logs", {"keyword": "database"})],
        )
    )

    assert len(clients) == 1
    assert clients[0].configs.keys() == {"log"}
    assert clients[0].connect_calls == 1
    assert clients[0].close_calls == 1
    assert tools[0].calls == [
        {"keyword": "timeout"},
        {"keyword": "database"},
    ]
    assert len(result.outputs) == 3
    assert result.outputs[0].output == {
        "result": [{"arguments": {"keyword": "timeout"}}]
    }


def test_shared_mcp_access_preserves_prometheus_structured_result(monkeypatch):
    class FakeTool:
        name = "prometheus__query_prometheus"

        async def ainvoke(self, _arguments):
            return {
                "structured_content": {
                    "resultType": "vector",
                    "result": [{"metric": {"job": "payment-api"}}],
                }
            }

    class FakeClient:
        async def connect_all(self):
            return {"prometheus": {"connected": True}}

        async def close(self):
            return None

    class FakeAdapter:
        def __init__(self, _client, *, secrets=()):
            pass

        async def get_langchain_tools(self):
            return [FakeTool()]

    monkeypatch.setattr(
        mcp_tool_access,
        "load_server_configs",
        lambda: {"prometheus": object()},
    )
    monkeypatch.setattr(mcp_tool_access, "MultiServerMCPClient", lambda _configs: FakeClient())
    monkeypatch.setattr(mcp_tool_access, "LangChainMCPAdapter", FakeAdapter)

    result = _run(
        mcp_tool_access.call_server_tools(
            "prometheus",
            [("query_prometheus", {"query": "up"})],
        )
    )

    assert result.outputs[0].output == {
        "resultType": "vector",
        "result": [{"metric": {"job": "payment-api"}}],
    }


def test_shared_mcp_access_reports_safe_connection_and_tool_errors(monkeypatch):
    class FakeClient:
        def __init__(self, configs):
            self.close_calls = 0

        async def connect_all(self):
            return {"log": {"connected": False}}

        async def close(self):
            self.close_calls += 1

    clients = []

    def create_client(configs):
        client = FakeClient(configs)
        clients.append(client)
        return client

    monkeypatch.setattr(
        mcp_tool_access, "load_server_configs", lambda: {"log": object()}
    )
    monkeypatch.setattr(mcp_tool_access, "MultiServerMCPClient", create_client)

    unavailable = _run(
        mcp_tool_access.call_server_tools(
            "log",
            [("get_recent_logs", {})],
        )
    )

    assert unavailable.connected is False
    assert "unavailable" in unavailable.connection_error
    assert clients[0].close_calls == 1

    class BrokenTool:
        name = "log__get_recent_logs"

        async def ainvoke(self, arguments):
            raise RuntimeError("credential-value-must-not-leak")

    class ConnectedClient(FakeClient):
        async def connect_all(self):
            return {"log": {"connected": True}}

    class FakeAdapter:
        def __init__(self, client, *, secrets=()):
            self.client = client

        async def get_langchain_tools(self):
            return [BrokenTool()]

    monkeypatch.setattr(mcp_tool_access, "MultiServerMCPClient", ConnectedClient)
    monkeypatch.setattr(mcp_tool_access, "LangChainMCPAdapter", FakeAdapter)

    failed = _run(
        mcp_tool_access.call_server_tools(
            "log",
            [("get_recent_logs", {})],
        )
    )

    assert failed.errors[0].message == "Tool invocation failed (RuntimeError)."
    assert "credential-value" not in repr(failed)


@pytest.mark.parametrize(
    ("agent", "function", "evidence_field", "tool_name"),
    [
        (log_agent, log_agent.analyze_logs, "log_evidence", "get_recent_logs"),
        (
            metrics_agent,
            metrics_agent.analyze_service_metrics,
            "metrics_evidence",
            "get_latest_metrics",
        ),
        (
            prometheus_agent,
            prometheus_agent.analyze_prometheus,
            "prometheus_evidence",
            "get_prometheus_targets",
        ),
        (
            grafana_agent,
            grafana_agent.analyze_grafana,
            "grafana_evidence",
            "search_dashboards",
        ),
    ],
)
def test_specialist_contains_unexpected_payload_processing_errors(
    monkeypatch,
    agent,
    function,
    evidence_field,
    tool_name,
):
    async def return_payload(server, requests, **kwargs):
        return _batch(ToolInvocation(tool_name, {"records": []}))

    def fail_mapping(payload):
        raise RuntimeError("private-payload-detail")

    monkeypatch.setattr(agent, "call_server_tools", return_payload)
    monkeypatch.setattr(agent, "mapping_records", fail_mapping)

    update = _run(function(_state()))

    assert update[evidence_field] == []
    assert update["errors"] == [f"{agent.AGENT_NAME} failed (RuntimeError)."]
    assert update["evidence_gaps"]
    assert "private-payload-detail" not in repr(update)
    metadata = next(iter(update["execution_metadata"].values()))
    assert metadata["status"] == "failed"
