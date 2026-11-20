"""Read-only access to the project's local datasets and configured MCP sources."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from app.agents.agent_utils import log_service_from_record, mapping_records
from app.agents.mcp_tool_access import call_server_tools
from app.config import API_VERSION, APPLICATION_NAME, RAG_EMBEDDING_MODEL
from app.mcp_client.multi_server_client import MultiServerMCPClient
from app.mcp_client.server_config import load_server_configs
from app.rag.retriever import retrieve_documents
from app.schemas.operational import (
    DatasetStatus,
    KnowledgeDocument,
    KnowledgeSearchResponse,
    KnowledgeSearchResult,
    LogQueryResponse,
    LogRecord,
    MCPServerStatus,
    MCPToolInfo,
    MetricRecord,
    MetricSummary,
    MetricsQueryResponse,
    OperationOverview,
    VectorStoreStatus,
)

_LOG_LEVEL_FIELDS = ("level", "Level", "severity")
_LOG_TIME_FIELDS = ("timestamp", "time", "Time", "observed_at")
_LOG_MESSAGE_FIELDS = ("Content", "message", "error_message", "event", "EventTemplate")
_SAFE_LOG_FIELDS = {
    "dataset",
    "lineid",
    "time",
    "timestamp",
    "level",
    "content",
    "eventid",
    "eventtemplate",
    "request_id",
    "trace_id",
    "event",
    "method",
    "endpoint",
    "http_status",
    "outcome",
    "error_code",
    "latency_ms",
    "retry_count",
    "region",
    "environment",
}
_METRIC_FLOAT_FIELDS = (
    "cpu_usage_percent",
    "memory_usage_percent",
    "avg_latency_ms",
    "disk_usage_percent",
)
_METRIC_INT_FIELDS = (
    "request_count",
    "error_count",
    "active_connections",
    "db_connections",
)
_MCP_LABELS = {
    "log": "Log MCP",
    "metrics": "Metrics MCP",
    "prometheus": "Prometheus",
    "grafana": "Grafana",
}


class OperationalDataServiceError(RuntimeError):
    """Safe, typed failure for a requested operational data source."""

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        status_code: int = 503,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.status_code = status_code


def _utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _text_value(record: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_log_record(record: dict[str, Any]) -> LogRecord:
    timestamp = _text_value(record, _LOG_TIME_FIELDS)
    level = _text_value(record, _LOG_LEVEL_FIELDS)
    dataset = record.get("dataset")
    source = dataset if isinstance(dataset, str) else "unknown"
    message = _text_value(record, _LOG_MESSAGE_FIELDS)
    if message is None:
        message = " ".join(
            str(record[field])
            for field in ("method", "endpoint", "http_status", "outcome")
            if record.get(field) not in (None, "")
        ) or "Log record"

    safe_raw = {
        str(key): value
        for key, value in record.items()
        if str(key).casefold() in _SAFE_LOG_FIELDS
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    identity = next(
        (
            str(record[key])
            for key in ("LineId", "request_id", "trace_id")
            if record.get(key) not in (None, "")
        ),
        json.dumps(record, sort_keys=True, default=str),
    )
    digest = hashlib.sha256(f"{source}:{identity}".encode()).hexdigest()[:20]
    return LogRecord(
        id=f"{source}:{digest}",
        timestamp=timestamp,
        level=level,
        service=log_service_from_record(record),
        message=message,
        source=source,
        raw=safe_raw,
    )


def _number(record: dict[str, Any], field: str, *, integer: bool = False):
    value = record.get(field)
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise OperationalDataServiceError(
            "invalid_metrics_data",
            "The metrics source returned an invalid numeric value.",
        ) from None
    if not number.is_integer() and integer:
        raise OperationalDataServiceError(
            "invalid_metrics_data",
            "The metrics source returned a non-integer count.",
        )
    return int(number) if integer else number


def _normalise_metric(record: dict[str, Any]) -> MetricRecord:
    timestamp = record.get("timestamp")
    service = record.get("service_name")
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise OperationalDataServiceError(
            "invalid_metrics_data",
            "The metrics source returned a record without a timestamp.",
        )
    if not isinstance(service, str) or not service.strip():
        raise OperationalDataServiceError(
            "invalid_metrics_data",
            "The metrics source returned a record without a service name.",
        )
    values = {
        field: _number(record, field)
        for field in _METRIC_FLOAT_FIELDS
    }
    values.update(
        {
            field: _number(record, field, integer=True)
            for field in _METRIC_INT_FIELDS
        }
    )
    return MetricRecord(
        timestamp=timestamp,
        service_name=service,
        **values,
    )


def _summarize_metrics(records: list[MetricRecord]) -> MetricSummary:
    def values(field: str) -> list[float]:
        return [
            value
            for record in records
            if (value := getattr(record, field)) is not None
        ]

    cpu = values("cpu_usage_percent")
    memory = values("memory_usage_percent")
    latency = values("avg_latency_ms")
    request_counts = values("request_count")
    error_counts = values("error_count")
    connections = values("db_connections")
    return MetricSummary(
        average_cpu_usage_percent=round(sum(cpu) / len(cpu), 2) if cpu else None,
        maximum_cpu_usage_percent=max(cpu) if cpu else None,
        average_memory_usage_percent=(
            round(sum(memory) / len(memory), 2) if memory else None
        ),
        maximum_memory_usage_percent=max(memory) if memory else None,
        average_latency_ms=(
            round(sum(latency) / len(latency), 2) if latency else None
        ),
        maximum_latency_ms=max(latency) if latency else None,
        total_requests=sum(int(value) for value in request_counts)
        if request_counts
        else None,
        total_errors=sum(int(value) for value in error_counts)
        if error_counts
        else None,
        maximum_db_connections=max(connections) if connections else None,
    )


def _read_vector_store_status() -> VectorStoreStatus:
    try:
        import chromadb

        from app.rag.vector_store import (
            COLLECTION_NAME,
            DEFAULT_PERSIST_DIRECTORY,
        )

        if not DEFAULT_PERSIST_DIRECTORY.is_dir():
            raise FileNotFoundError
        client = chromadb.PersistentClient(path=str(DEFAULT_PERSIST_DIRECTORY))
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        result = collection.get(
            limit=10,
            include=["documents", "metadatas"],
        )
    except Exception as error:
        raise OperationalDataServiceError(
            "historical_store_unavailable",
            f"The local historical vector store is unavailable ({type(error).__name__}).",
        ) from None

    ids = result.get("ids", [])
    texts = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    documents = []
    for index, chunk_id in enumerate(ids):
        metadata = (
            metadatas[index]
            if index < len(metadatas) and isinstance(metadatas[index], dict)
            else {}
        )
        text = texts[index] if index < len(texts) else ""
        if not isinstance(text, str):
            text = ""
        incident_id = metadata.get("incident_id")
        service = metadata.get("service") or metadata.get("service_name")
        title = metadata.get("title")
        severity = metadata.get("severity")
        documents.append(
            KnowledgeDocument(
                chunk_id=str(chunk_id),
                incident_id=incident_id if isinstance(incident_id, str) else None,
                title=title if isinstance(title, str) else None,
                service=service if isinstance(service, str) else None,
                severity=severity if isinstance(severity, str) else None,
                preview=text[:500],
            )
        )

    status = "healthy" if count > 0 else "degraded"
    return VectorStoreStatus(
        status=status,
        vector_database="Chroma (local persistent)",
        vector_database_status=status,
        collection_name=COLLECTION_NAME,
        embedding_model=RAG_EMBEDDING_MODEL,
        document_count=count,
        documents=documents,
        message=(
            f"Connected to the local collection with {count} historical chunks."
            if count
            else "The collection is reachable but contains no historical chunks."
        ),
    )


def _read_local_dataset_status() -> tuple[list[DatasetStatus], dict[str, int]]:
    try:
        from app.mcp_servers.log_mcp_server import available_logs
        from app.mcp_servers.metrics_mcp_server import available_metrics

        log_catalog = available_logs()["datasets"]
        metrics_catalog = available_metrics()
    except Exception as error:
        raise OperationalDataServiceError(
            "local_dataset_unavailable",
            f"A local operational dataset is unavailable ({type(error).__name__}).",
        ) from None

    datasets = [
        DatasetStatus(
            name=item["name"],
            status="healthy" if item["record_count"] > 0 else "degraded",
            record_count=item["record_count"],
            message="Processed log dataset is readable.",
        )
        for item in log_catalog
    ]
    datasets.append(
        DatasetStatus(
            name=metrics_catalog["dataset_name"],
            status=(
                "healthy" if metrics_catalog["record_count"] > 0 else "degraded"
            ),
            record_count=metrics_catalog["record_count"],
            message="Processed metrics dataset is readable.",
        )
    )
    return datasets, {
        "logs": sum(item["record_count"] for item in log_catalog),
        "metrics": metrics_catalog["record_count"],
    }


class OperationalDataService:
    """Expose the real local datasets, vector store, and live MCP health."""

    async def query_logs(
        self,
        *,
        service: str | None,
        level: str | None,
        search: str | None,
        start_time: datetime | None,
        end_time: datetime | None,
        limit: int,
        offset: int,
    ) -> LogQueryResponse:
        requests = [
            (
                "query_logs",
                {
                    "service_name": service,
                    "level": level,
                    "keyword": search,
                    "start_time": (
                        _utc_datetime(start_time).isoformat()
                        if start_time is not None
                        else None
                    ),
                    "end_time": (
                        _utc_datetime(end_time).isoformat()
                        if end_time is not None
                        else None
                    ),
                    "limit": limit,
                    "offset": offset,
                },
            )
        ]
        batch = await call_server_tools("log", requests)
        if not batch.connected or batch.errors or not batch.outputs:
            raise OperationalDataServiceError(
                "logs_unavailable",
                "The local Log MCP data source is unavailable.",
            )
        payload = batch.outputs[0].output
        if not isinstance(payload, dict) or not isinstance(
            payload.get("records"), list
        ):
            raise OperationalDataServiceError(
                "invalid_logs_response",
                "The Log MCP data source returned an invalid response.",
            )
        return LogQueryResponse(
            logs=[
                _safe_log_record(record)
                for record in payload["records"]
                if isinstance(record, dict)
            ],
            total=int(payload.get("total", 0)),
            offset=offset,
            limit=limit,
            excluded_untimed_count=int(payload.get("excluded_untimed_count", 0)),
            excluded_outside_range_count=int(
                payload.get("excluded_outside_range_count", 0)
            ),
        )

    async def query_metrics(
        self,
        *,
        service: str | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> MetricsQueryResponse:
        tool_name = "get_metrics_by_service" if service else "get_latest_metrics"
        arguments = {"limit": 500}
        if service:
            arguments["service_name"] = service
        batch = await call_server_tools(
            "metrics",
            [(tool_name, arguments)],
        )
        if not batch.connected or batch.errors or not batch.outputs:
            raise OperationalDataServiceError(
                "metrics_unavailable",
                "The local Metrics MCP data source is unavailable.",
            )
        payload = batch.outputs[0].output
        if isinstance(payload, dict) and "result" in payload:
            payload = payload["result"]
        if not isinstance(payload, list):
            raise OperationalDataServiceError(
                "invalid_metrics_response",
                "The Metrics MCP data source returned an invalid response.",
            )
        raw_records = mapping_records(payload)

        start = _utc_datetime(start_time)
        end = _utc_datetime(end_time)
        records: list[MetricRecord] = []
        for candidate in raw_records:
            if not isinstance(candidate, dict):
                continue
            record = _normalise_metric(candidate)
            try:
                timestamp = datetime.fromisoformat(
                    record.timestamp.replace("Z", "+00:00")
                )
            except ValueError:
                raise OperationalDataServiceError(
                    "invalid_metrics_data",
                    "The metrics source returned an invalid timestamp.",
                ) from None
            timestamp = _utc_datetime(timestamp)
            if start is not None and timestamp < start:
                continue
            if end is not None and timestamp > end:
                continue
            if service and record.service_name.casefold() != service.casefold():
                continue
            records.append(record)

        records.sort(key=lambda item: item.timestamp)
        return MetricsQueryResponse(
            service=service,
            metrics=records,
            total=len(records),
            summary=_summarize_metrics(records),
        )

    async def knowledge_status(self) -> VectorStoreStatus:
        try:
            return await asyncio.to_thread(_read_vector_store_status)
        except OperationalDataServiceError:
            raise

    async def search_knowledge(
        self,
        query: str,
        *,
        limit: int,
    ) -> KnowledgeSearchResponse:
        try:
            documents = await asyncio.to_thread(
                retrieve_documents,
                query,
                top_k=limit,
            )
        except Exception as error:
            raise OperationalDataServiceError(
                "historical_search_unavailable",
                f"Historical search failed ({type(error).__name__}).",
            ) from None

        results = []
        for document in documents:
            metadata = document.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            text = document.get("text")
            if not isinstance(text, str):
                continue
            incident_id = metadata.get("incident_id")
            service = metadata.get("service") or metadata.get("service_name")
            title = metadata.get("title")
            severity = metadata.get("severity")
            results.append(
                KnowledgeSearchResult(
                    chunk_id=str(document["chunk_id"]),
                    incident_id=incident_id if isinstance(incident_id, str) else None,
                    title=title if isinstance(title, str) else None,
                    service=service if isinstance(service, str) else None,
                    severity=severity if isinstance(severity, str) else None,
                    score=float(document["score"]),
                    retrieval_method=str(document["retrieval_method"]),
                    retrieval_sources=[
                        str(source)
                        for source in document.get("retrieval_sources", [])
                    ],
                    snippet=text[:800],
                )
            )
        return KnowledgeSearchResponse(
            query=query,
            results=results,
            total=len(results),
        )

    async def mcp_status(self) -> list[MCPServerStatus]:
        configs = load_server_configs()
        client = MultiServerMCPClient(configs)
        try:
            connection_status = await client.connect_all()
            discovered = await client.discover_all_tools()
            probes = {
                "log": ("query_logs", {"limit": 1}),
                "metrics": ("get_latest_metrics", {"limit": 500}),
                "prometheus": ("get_prometheus_health", {}),
                "grafana": ("get_grafana_health", {}),
            }

            async def probe(name: str):
                if not connection_status.get(name, {}).get("connected"):
                    return name, None, "MCP server process is unavailable."
                tool_name, arguments = probes[name]
                try:
                    return name, await client.call_tool(name, tool_name, arguments), None
                except Exception as error:
                    return name, None, f"Health probe failed ({type(error).__name__})."

            probe_results = await asyncio.gather(
                *(probe(name) for name in configs)
            )
            probe_by_name = {
                name: (result, error)
                for name, result, error in probe_results
            }

            statuses = []
            for name in configs:
                connection = connection_status.get(name, {})
                mcp_connected = connection.get("connected") is True
                result, probe_error = probe_by_name[name]
                structured = (
                    result.get("structured_content")
                    if isinstance(result, dict)
                    else None
                )
                records_available = None
                http_status = None
                message = "MCP server process is unavailable."
                healthy = False
                if probe_error:
                    message = probe_error
                elif name in {"log", "metrics"}:
                    if name == "log" and isinstance(structured, dict):
                        records = structured.get("records")
                        records_available = structured.get("total")
                        healthy = isinstance(records, list) and bool(records)
                    else:
                        if name == "metrics" and isinstance(structured, dict):
                            structured = structured.get("result")
                        healthy = isinstance(structured, list) and bool(structured)
                        records_available = len(structured) if healthy else 0
                    message = (
                        "Connected; local data probe returned records."
                        if healthy
                        else "Connected, but the local data probe returned no records."
                    )
                else:
                    health = structured if isinstance(structured, dict) else {}
                    connected = health.get("connected") is True
                    http_status_value = health.get("http_status")
                    http_status = (
                        http_status_value
                        if isinstance(http_status_value, int)
                        else None
                    )
                    error_message = health.get("error")
                    message = (
                        "Connected to the upstream service."
                        if connected
                        else (
                            error_message
                            if isinstance(error_message, str)
                            else "The upstream health check did not confirm connectivity."
                        )
                    )
                    healthy = connected

                if not mcp_connected:
                    status = "unavailable"
                elif healthy:
                    status = "healthy"
                elif "not configured" in message.casefold():
                    status = "not_configured"
                else:
                    status = "unavailable"

                tool_infos = []
                server_connection = client.connections.get(name)
                if server_connection:
                    for tool in server_connection.tools.values():
                        discovered_name = f"{name}__{tool.name}"
                        metadata = discovered.get(discovered_name, {})
                        tool_infos.append(
                            MCPToolInfo(
                                name=tool.name,
                                description=tool.description or tool.name,
                                input_schema=metadata.get("input_schema", {}),
                            )
                        )
                statuses.append(
                    MCPServerStatus(
                        name=name,
                        display_name=_MCP_LABELS.get(name, name),
                        status=status,
                        mcp_connected=mcp_connected,
                        tool_count=connection.get("tools_count", 0),
                        resource_count=connection.get("resources_count", 0),
                        records_available=records_available,
                        http_status=http_status,
                        message=message,
                        tools=tool_infos,
                    )
                )
            return statuses
        finally:
            await client.close()

    async def overview(self) -> OperationOverview:
        checked_at = datetime.now(UTC).isoformat()
        mcp_servers, knowledge_result, catalog_result = await asyncio.gather(
            self.mcp_status(),
            self.knowledge_status(),
            asyncio.to_thread(_read_local_dataset_status),
            return_exceptions=True,
        )

        if isinstance(mcp_servers, Exception):
            mcp_statuses = [
                MCPServerStatus(
                    name=name,
                    display_name=_MCP_LABELS[name],
                    status="unavailable",
                    mcp_connected=False,
                    tool_count=0,
                    resource_count=0,
                    message=f"MCP health check failed ({type(mcp_servers).__name__}).",
                )
                for name in _MCP_LABELS
            ]
        else:
            mcp_statuses = mcp_servers

        if isinstance(knowledge_result, VectorStoreStatus):
            vector_store = knowledge_result
        else:
            vector_store = VectorStoreStatus(
                status="unavailable",
                vector_database="Chroma (local persistent)",
                vector_database_status="unavailable",
                collection_name="historical_incidents",
                embedding_model=RAG_EMBEDDING_MODEL,
                document_count=0,
                documents=[],
                message=(
                    "The local historical vector store is unavailable; "
                    "its records were not counted."
                ),
            )

        if isinstance(catalog_result, tuple):
            datasets = catalog_result[0]
        else:
            datasets = [
                DatasetStatus(
                    name="local operational datasets",
                    status="unavailable",
                    record_count=0,
                    message=(
                        f"Dataset check failed ({type(catalog_result).__name__})."
                    ),
                )
            ]

        statuses = [item.status for item in mcp_statuses]
        statuses.append(vector_store.status)
        statuses.extend(item.status for item in datasets)
        if statuses and all(status == "healthy" for status in statuses):
            overall = "healthy"
        elif any(status == "healthy" for status in statuses):
            overall = "degraded"
        else:
            overall = "unavailable"

        return OperationOverview(
            status=overall,
            application_status="healthy",
            application=APPLICATION_NAME,
            api_version=API_VERSION,
            checked_at=checked_at,
            mcp_servers=mcp_statuses,
            datasets=datasets,
            vector_store=vector_store,
        )
