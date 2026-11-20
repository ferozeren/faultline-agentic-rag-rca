# Faultline

Faultline investigates IT incidents using project logs, metrics, historical incident context, and configured observability integrations. It includes a React frontend, FastAPI backend, LangGraph workflow, and PostgreSQL-backed investigation history.

## Requirements

- uv (recommended) or Python 3.11 or newer
- Node.js and npm
- PostgreSQL
- NVIDIA NIM credentials for model-backed RCA and recommendations

Prometheus and Grafana are optional. Without a configured integration, its live evidence may be unavailable. The project includes sample operational data and a prebuilt historical Chroma collection; it does not seed fake investigation history.

Install uv using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).

## Configure and start

Create `backend/.env` from the example and set NVIDIA and PostgreSQL values. `DATABASE_URL` takes precedence over the individual `POSTGRES_*` settings. Keep `.env` private; it is excluded by `.gitignore`.

### Linux and macOS

```bash
cd backend
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
```

### Windows PowerShell

```powershell
Set-Location backend
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Copy-Item .env.example .env
```

If you prefer Python and pip, use `python3.11 -m venv .venv` on Linux or macOS, or `py -3.11 -m venv .venv` on Windows. Activate the environment as above, then install dependencies with `python -m pip install -r requirements.txt`.

Edit `backend/.env`. Use an existing PostgreSQL database and role, or create them with an administrator. For example, in `psql`:

```sql
CREATE ROLE incident_investigator LOGIN;
\password incident_investigator
CREATE DATABASE incident_investigator OWNER incident_investigator;
```

Set the matching connection values in `.env`. Apply the schema and run the backend from `backend/`:

```bash
alembic upgrade head
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open a second terminal at the project root and start the frontend:

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open <http://127.0.0.1:5173/>. The API docs are at <http://127.0.0.1:8000/docs>. Vite proxies API requests to port 8000. To change the backend target, copy `frontend/.env.example` to `frontend/.env`, edit `VITE_API_PROXY_TARGET`, and restart Vite.

On Windows, run `alembic upgrade head` and the Uvicorn command in the activated backend environment. The frontend commands are the same in PowerShell.

Check backend and database readiness:

```text
GET http://127.0.0.1:8000/api/v1/health
GET http://127.0.0.1:8000/api/v1/ready
```

Readiness should report `database_status: connected`. If PostgreSQL does not start, inspect its service logs and verify that the service account can read its configuration and data directory. Do not initialize or remove an existing database cluster.

Investigation history can be archived and restored from the Incidents page. Permanent deletion requires confirmation and removes the investigation, report, evidence, and agent-run records from PostgreSQL.

## Use the included data

The local datasets include payment API logs, service metrics, and historical incident documents. For example, search the included payment API logs for the recorded gateway timeout:

```bash
curl --get http://127.0.0.1:8000/api/v1/data/logs \
	--data-urlencode 'service_name=payment-api' \
	--data-urlencode 'search=GATEWAY_TIMEOUT' \
	--data-urlencode 'start_time=2026-10-01T02:45:00Z' \
	--data-urlencode 'end_time=2026-10-01T03:00:00Z'
```

Historical incident documents are the RAG corpus. Current logs and metrics are
queried directly for the requested service and time window; they are not added
to the historical index. Known environment mismatches are excluded. The included
metrics have no environment field, so production alignment cannot be verified.

Single-source evidence can trigger only a bounded RCA review. An unresolved
result keeps evidence-linked hypotheses; recommendations require a validated
primary finding and are not generated from hypotheses alone.

Submit an investigation for that same event from the frontend or with the API:

```bash
curl --request POST http://127.0.0.1:8000/api/v1/investigations \
	--header 'Content-Type: application/json' \
	--data '{
    "incident_id": "PAYMENT-API-2026-10-01-0252",
    "description": "Investigate the payment API gateway timeout recorded at 2026-10-01 02:52 UTC.",
    "affected_service": "payment-api",
    "environment": "production",
    "start_time": "2026-10-01T02:45:00Z",
    "end_time": "2026-10-01T03:00:00Z"
  }'
```

The workflow only reports causes supported by available evidence. Missing, stale, or conflicting telemetry can result in `insufficient_evidence`; historical incidents are context, not proof of a current cause.

## Saved investigations

Each submitted investigation is saved in PostgreSQL, including its status and available report, evidence, and agent runs. Use the returned `investigation_id` with:

```text
GET /api/v1/investigations
GET /api/v1/investigations/{investigation_id}
GET /api/v1/investigations/{investigation_id}/report
GET /api/v1/investigations/{investigation_id}/evidence
GET /api/v1/investigations/{investigation_id}/agents
GET /api/v1/investigations/stats
```

The history page supports text search, service, severity, and status filters, and pagination. Records remain available after restarting the backend.

## Configuration and project layout

Copy [`backend/.env.example`](backend/.env.example) to `backend/.env` and fill in the values for your environment. The main settings are:

```dotenv
# NVIDIA NIM (required for model-assisted RCA)
NVIDIA_MODEL=
NVIDIA_API_KEY=

# Optional Prometheus and Grafana integrations
PROMETHEUS_URL=
PROMETHEUS_BEARER_TOKEN=
PROMETHEUS_USERNAME=
GRAFANA_URL=
GRAFANA_SERVICE_ACCOUNT_TOKEN=

# PostgreSQL; DATABASE_URL takes precedence when set
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=incident_investigator
POSTGRES_USER=incident_investigator
POSTGRES_PASSWORD=
DATABASE_URL=
```

Keep `backend/.env` private. The checked-in example also lists local RAG, API, and CORS settings.

Project data stays under `backend/data`. Paths are resolved relative to the application files, not a machine-specific home directory. The code uses Python path handling and works with Windows path separators. Local MCP servers are launched by the backend when needed.

## Tests

From `backend/`:

```bash
python -m pytest -q
python -m compileall -q app tests
```

From `frontend/`:

```bash
npm test
npm run typecheck
npm run build
```

Automated tests mock model responses and do not require NVIDIA credentials.
