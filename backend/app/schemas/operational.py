"""Validated API contracts for operational data and source health."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

SourceStatus = Literal["healthy", "degraded", "unavailable", "not_configured"]


class MCPToolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, JsonValue]


class MCPServerStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str
    status: SourceStatus
    mcp_connected: bool
    tool_count: int = Field(ge=0)
    resource_count: int = Field(ge=0)
    records_available: int | None = Field(default=None, ge=0)
    http_status: int | None = Field(default=None, ge=100, le=599)
    message: str
    tools: list[MCPToolInfo] = Field(default_factory=list)


class LogRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    timestamp: str | None = None
    level: str | None = None
    service: str | None = None
    message: str
    source: str
    raw: dict[str, JsonValue]


class LogQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logs: list[LogRecord]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    excluded_untimed_count: int = Field(default=0, ge=0)
    excluded_outside_range_count: int = Field(default=0, ge=0)


class MetricRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    service_name: str
    cpu_usage_percent: float | None = None
    memory_usage_percent: float | None = None
    request_count: int | None = None
    error_count: int | None = None
    avg_latency_ms: float | None = None
    active_connections: int | None = None
    db_connections: int | None = None
    disk_usage_percent: float | None = None


class MetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    average_cpu_usage_percent: float | None = None
    maximum_cpu_usage_percent: float | None = None
    average_memory_usage_percent: float | None = None
    maximum_memory_usage_percent: float | None = None
    average_latency_ms: float | None = None
    maximum_latency_ms: float | None = None
    total_requests: int | None = None
    total_errors: int | None = None
    maximum_db_connections: int | None = None


class MetricsQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str | None = None
    metrics: list[MetricRecord]
    total: int = Field(ge=0)
    summary: MetricSummary


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    incident_id: str | None = None
    title: str | None = None
    service: str | None = None
    severity: str | None = None
    preview: str


class VectorStoreStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: SourceStatus
    vector_database: str
    vector_database_status: SourceStatus
    collection_name: str
    embedding_model: str
    document_count: int = Field(ge=0)
    documents: list[KnowledgeDocument] = Field(default_factory=list)
    message: str


class KnowledgeSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    incident_id: str | None = None
    title: str | None = None
    service: str | None = None
    severity: str | None = None
    score: float
    retrieval_method: str
    retrieval_sources: list[str] = Field(default_factory=list)
    snippet: str


class KnowledgeSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[KnowledgeSearchResult]
    total: int = Field(ge=0)


class DatasetStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: SourceStatus
    record_count: int = Field(ge=0)
    message: str


class OperationOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["healthy", "degraded", "unavailable"]
    application_status: Literal["healthy"]
    application: str
    api_version: str
    checked_at: str
    mcp_servers: list[MCPServerStatus]
    datasets: list[DatasetStatus]
    vector_store: VectorStoreStatus
