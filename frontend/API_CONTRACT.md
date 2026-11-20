# Frontend/backend API contract

The frontend uses the FastAPI endpoints implemented by the backend:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Application liveness check; does not report external service health |
| `GET` | `/api/v1/ready` | Safe readiness response |
| `POST` | `/api/v1/investigations` | Runs the existing investigation workflow once and returns a validated report |

## Investigation request

The request uses the backend `IncidentRequest` schema. Only these fields are sent:

```json
{
  "incident_id": "INC-123",
  "description": "Payment requests are timing out.",
  "affected_service": "payments-api",
  "environment": "production",
  "severity": "SEV-2",
  "start_time": "2025-01-15T10:00:00Z",
  "end_time": "2025-01-15T10:30:00Z",
  "additional_context": {
    "deployment_id": "deploy-456"
  }
}
```

Only `description` is required. Times are optional ISO 8601 datetimes, and the backend validates that the end does not precede the start. `additional_context`, when supplied, must be a JSON object.

## Investigation response

The response follows `InvestigationResponse`: `investigation_id`, `status`, the validated `incident`, optional `supervisor_decision`, the complete `report`, and convenience fields for RCA, recommendations, timeline, evidence references/analysis, conclusion, warnings, and limitations. The UI displays current and historical evidence separately, preserves uncertainty and warnings, labels recommendations as advisory, and provides the complete report JSON for inspection.

The backend's safe errors contain `error`, `message`, and `correlation_id`. The frontend sends an `X-Correlation-ID` with each API request and displays only the safe API message and reference ID.

## Local development and state

By default, Vite proxies `/api` requests to `http://127.0.0.1:8000`; override the target with `VITE_API_PROXY_TARGET`. Set `VITE_API_BASE_URL` only when connecting directly to a backend origin that permits the frontend through CORS. The settings page supports the same-origin proxy when its backend URL is blank.

There is no backend history, polling, or websocket endpoint. The frontend keeps only the latest investigation response in memory for the current browser session. Refreshing or closing the page loses that result.

Dashboard aggregates, incident/report history, logs, metrics, MCP status, and RAG search are not part of this backend API. The frontend does not generate substitute telemetry or claim those services are healthy.
