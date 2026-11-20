# End-to-End Acceptance Report

**Run date:** 2026-11-18
**Verdict:** **PARTIAL PASS — the real UI/API workflow completed, but evidence quality and several integrations prevent production acceptance.**

## A. Executive summary

Two unmocked investigations were submitted: one through the React incident form and one directly to FastAPI. Both returned HTTP 200 and a structured, schema-validated partial report. The API-to-LangGraph path and NVIDIA-backed RCA path ran; the RCA correctly returned `insufficient_evidence`, without a primary finding or recommendations, because its available evidence did not support a reliable cause.

The run confirms that the earlier “no RCA model configured” outcome is no longer the only blocker: the live request produced a substantive RCA analysis. It also exposed material evidence-quality and integration defects: logs from unrelated dates were attributed to `payment-api`, the historical RAG graph branch failed because no LLM is injected into that branch, Prometheus results were malformed before analysis, and Grafana returned errors. The supervisor nevertheless marked evidence sufficient and ready for RCA.

No application source code was changed for this acceptance run. The report and raw API response are the deliverables. The tracked Chroma database was restored byte-for-byte after testing.

## B. Test environment and running services

| Component | Observed |
|---|---|
| OS | Linux |
| Backend interpreter | Python 3.14.10 in `backend/.venv`; Python 3.11 was not available |
| Frontend runtime | Node 26.13.1, npm 12.2.2 |
| Backend | Uvicorn at `http://127.0.0.1:8000` |
| Frontend | Vite at `http://127.0.0.1:5173` |
| API docs | `/docs` and `/openapi.json` both returned HTTP 200 |
| Health | `/api/v1/health` returned `healthy`; external dependencies explicitly `not_checked` |
| Readiness | `/api/v1/ready` returned `ready`; it does not claim external MCP/NVIDIA/monitoring health |
| Process snapshot | After the live requests: backend 810,248 KiB RSS and 2.8% CPU; Vite 150,252 KiB RSS and 0.5% CPU. This is a single sample, not a resource benchmark. |

After the final backend restart and database cleanup, health, readiness, docs, OpenAPI, and the frontend root all returned HTTP 200. Both services were left running.

The Vite same-origin `/api` proxy worked. A direct CORS preflight from `http://127.0.0.1:5173` to the backend returned HTTP 400, `Disallowed CORS origin`; direct cross-origin browser use is not verified.

## C. Existing architecture exercised

The test used the existing application and graph; it did not create an alternate workflow.

```text
React incident form / POST /api/v1/investigations
                    |
                    v
          Investigation service
                    |
                    v
            Existing LangGraph
    +---------------+----------------+
    |               |                |
Historical RAG   Log/Metrics     Prometheus/Grafana
    +---------------+----------------+
                    |
          Evidence aggregation
                    |
                Supervisor
                    |
          Conditional RCA analysis
                    |
      Recommendation gate (valid RCA only)
                    |
          Report generation / response
```

The workflow uses the existing five specialist branches, evidence aggregation, supervisor, RCA, recommendation, and report-generation stages. Related implementation: [workflow](../../backend/app/graph/workflow.py), [investigation service](../../backend/app/services/investigation_service.py), and [API routes](../../backend/app/api/routes).

## D. Incidents submitted

Both submissions used the project’s local data and external integrations; the incident itself was synthetic and labelled as local test data.

| Route | Incident ID | Investigation/correlation ID | Result |
|---|---|---|---|
| React UI | `INC-PORTFOLIO-E2E-20261118-01` | `a11b5dc4-e3fe-45f8-8958-08727a200fa1` | POST succeeded; report rendered in the UI |
| Direct API | `INC-PORTFOLIO-API-E2E-20261118-02` | `portfolio-acceptance-api-20261118-02` | HTTP 200; 20.26 seconds; response saved as JSON |

The API incident targeted `payment-api`, environment `Synthetic/local testing`, severity `SEV-2`, from `2026-11-02T11:15:00Z` through `2026-11-02T12:04:00Z`. Its description requested investigation of latency and intermittent failures after a deployment. The request completed with report and investigation status `insufficient_evidence`; the report was partial rather than a successful root-cause result.

The API response included incident metadata, supervisor decision, RCA result, recommendations, timeline, evidence analysis/references, conclusion, warnings, and limitations. The complete response is [saved here](./artifacts/INC-PORTFOLIO-API-E2E-20261118-02_portfolio-acceptance-api-20261118-02.json) (about 1.36 MB).

## E. Agent outcomes

Per-agent duration and full execution metadata are not exposed in the API response, so this report does not infer them.

| Stage | Observed outcome |
|---|---|
| Historical RAG | Failed with `RuntimeError`; returned no historical evidence. |
| Log analysis | Returned 204 evidence references. They include records outside the incident time window, including 50 Apache records dated December 2005, attributed in the report to `payment-api`. Another 154 payment API records are from October 2026, also outside the November 2 incident window. |
| Metrics analysis | Returned 51 references. Fifty one-minute points cover the requested incident window; the additional reference is a service summary. |
| Prometheus | Returned no usable evidence. A response-shape error was recorded, and `get_prometheus_targets` failed. |
| Grafana | Returned no usable evidence. Dashboard search and alert calls failed; the health endpoint returned 404. |
| Supervisor | `ready_for_rca`, `sufficient` evidence, despite stale/misattributed log evidence and unavailable monitoring/RAG sources. |
| RCA | The configured NVIDIA path produced an evidence-based analysis and a structured insufficient-evidence result. No primary root-cause finding was returned. |
| Recommendations | None. The recommendation stage did not produce actions without a validated RCA finding; this is the safe outcome for this run. |
| Report generation | Produced and returned the partial, validated report. |

The RCA summary explicitly notes that 2005 log events conflict with the 2026 incident window, and that unavailable monitoring and historical context prevent reliable correlation. Thus, for this run, “no root cause” reflects the evidence and its quality—not simply the former missing-model configuration.

## F. MCP integration checks

All four configured stdio MCP servers initialized. Discovery returned 23 tools total; all 23 were adapted, with no tool-name collisions, object-shaped schemas, and one resource discovered per server.

| MCP server | Discovered tools | Smoke-test result |
|---|---:|---|
| Log | 4 | Tool calls returned records; returned data included dates unrelated to the incident. |
| Metrics | 8 | Tool calls returned metrics, including the requested 50-point interval. |
| Prometheus | 5 | Health reported connected; instant/range/alert queries returned no records. Target discovery returned HTTP 404. The workflow also recorded a response-shape error. |
| Grafana | 6 | Health returned HTTP 404; dashboard/alert operations returned HTTP 503. Dashboard retrieval was skipped because no valid dashboard UID was available. |

MCP client sessions closed cleanly after the smoke checks. A successful MCP handshake is not evidence that the upstream monitoring APIs are healthy.

## G. Historical RAG

Direct hybrid retrieval completed and returned five reciprocal-rank-fused candidates from semantic and BM25 retrieval. The leading candidates were general incidents, not validated payment-service history. The live graph’s historical-RAG branch still failed and contributed zero evidence to the report.

The API service passes the configured NVIDIA model as `rca_llm` to `run_investigation`, while the historical agent calls `rag_chain.answer_question(..., llm=llm)`. No historical-RAG LLM is passed on this API path. See [service model wiring](../../backend/app/services/investigation_service.py), [historical agent](../../backend/app/agents/historical_rag_agent.py), and [RAG chain](../../backend/app/rag/rag_chain.py).

Consequently, retrieval alone does not make the historical agent functional when its answer-generation path requires an LLM. The response correctly did not fabricate historical evidence.

## H. Data, RCA, and NVIDIA observations

The metrics covered the requested interval and contained:

- CPU: 36.19–96.32%, mean 66.40%.
- Memory: 46.67–82.32%, mean 65.15%.
- Latency: 259.27–8650.75 ms, mean 2885.66 ms.
- Requests: 187–992 per minute; errors: 3–136 per minute.
- Database connections: 10–108.

These are observed values, not thresholds or evidence of a particular cause. The retrieved logs were not temporally aligned with the metrics. The report classified all 255 evidence references as `current`, even though the log data includes October 2026 and December 2005 records. That classification, along with the service attribution, needs correction before treating the report as operationally reliable.

The live API response confirms that the NVIDIA-backed RCA path no longer simply reports “no RCA model was provided.” It returned a detailed, schema-valid insufficient-evidence result. The NVIDIA integration library did log warnings that the configured model type and structured-output support were unknown. The successful request demonstrates this response worked; it does not guarantee all structured generations will work.

The MCP unwrapping helper treats any nested `structured_content` object containing a `result` key as a transport wrapper and replaces it with that value. That can discard Prometheus’s own `resultType` field, consistent with the observed parser error. See [MCP tool access](../../backend/app/agents/mcp_tool_access.py).

## I. Frontend behavior

- The real incident form accepted the request once and rendered the returned partial report, including status, evidence, warnings, limitations, and the report view.
- The report displayed 255 unique evidence references: 204 log and 51 metrics references.
- The UI did not invent historical records, dashboard metrics, or saved reports. The Reports page says reports are not persisted; after a full navigation, no report is available in that browser session.
- MCP, Metrics, Logs, and Historical Knowledge pages show explicit “Backend API … is not implemented or documented” states because no corresponding API path is configured. The MCP page labels its reference tool information as not backend-verified.
- At a 390 px mobile viewport, the populated result page had a measured document scroll width of 618 px. This is horizontal overflow on the result view and should be fixed.

## J. Regression and build results

| Check | Result |
|---|---|
| Focused backend agent/MCP/RAG/graph/API tests | 285 passed, 0 failed; 15 warnings |
| Full backend suite | 421 passed, 0 failed; 15 warnings, 17.87 seconds |
| Current chunking module | 22 passed, 0 failed; 13 warnings |
| Frontend unit tests | 3 passed across 2 files |
| Frontend typecheck | Passed |
| Frontend production build | Passed; Vite built 2,293 modules |
| Backend compileall | Passed |
| `git diff --check` | Passed |

The recorded earlier baseline was 509 passed and 7 chunking failures (516 total). The current tree collects 421 tests—95 fewer than that baseline—so these results are not a like-for-like comparison and do not establish that the historical seven failures were fixed. The 22 chunking tests present in the current tree did pass. Python 3.11 validation was not possible because this environment has Python 3.14.10 and no Python 3.11 executable.

Warnings included LangSmith/Chroma deprecations, a `langchain-community` sunset notice, and NVIDIA model/structured-output capability warnings.

## K. Findings and severity

1. **High — stale and misattributed log evidence.** The log agent falls back to the requested service when a record lacks an explicit service field. This run therefore associated Apache records from 2005 with `payment-api`; it also included October logs for a November incident. See [log analysis](../../backend/app/agents/log_analysis_agent.py) and [service fallback](../../backend/app/agents/agent_utils.py). This contaminates both evidence context and the report narrative.
2. **High — supervisor overstates evidence sufficiency.** The supervisor marked evidence sufficient and ready for RCA despite temporally mismatched logs, missing historical context, and unavailable monitoring. The RCA safely rejected a primary cause, but the supervisor decision is inconsistent with the evidence gaps.
3. **High — historical RAG model is not wired on the API graph path.** The historical agent failed at runtime and contributed no evidence.
4. **Medium — Prometheus response normalization loses response metadata.** The observed response-shape error is consistent with the unwrapping behavior described in section H; the query contributed no evidence.
5. **Medium — Grafana upstream endpoints unavailable.** HTTP 503 responses and a 404 health endpoint prevented usable Grafana evidence. This may require environment/service configuration as well as application handling.
6. **Medium — direct CORS origin rejected.** Same-origin Vite proxy works, but direct browser requests from the frontend origin receive HTTP 400.
7. **Low — mobile result view overflows horizontally.** The populated report view exceeded the 390 px viewport.

## L. Acceptance decision and limitations

**Partial pass, not production acceptance.** The core request path works end to end: the real frontend and API submit an incident, the existing graph executes, NVIDIA-backed RCA returns a validated result, and the UI presents the partial report. API health/readiness and API documentation are reachable.

Before relying on root-cause or recommendation output operationally:

- Stop assigning a requested service to records without a verified service identity; filter or label evidence by actual timestamps and incident window.
- Revisit the supervisor sufficiency decision against evidence freshness, source coverage, and conflicts.
- Inject/configure a model for the historical-RAG branch or make its no-model behavior explicitly non-failing and clearly reported.
- Preserve Prometheus `resultType` and `result` together through MCP normalization.
- Restore/verify Grafana connectivity and configure direct CORS if the UI will call the backend cross-origin.
- Fix the result page’s narrow-viewport overflow.

This was a local synthetic incident run, not a production incident. The report does not claim that external services are healthy, that the historical RAG branch completed, or that a root cause was identified.

## M. Evidence appendix and cleanup

- API response: [JSON artifact](./artifacts/INC-PORTFOLIO-API-E2E-20261118-02_portfolio-acceptance-api-20261118-02.json).
- Direct API request: HTTP 200; response report status `insufficient_evidence`; 255 evidence references; 6 report errors, 160 warnings, 8 limitations, and zero recommendations.
- Artifact safety scan found no NVIDIA-token pattern, bearer credential, API-key assignment, or evaluation labels (`scenario_label`, `is_anomaly`). No `.env` file was read.
- The Chroma SQLite file changed during retrieval/testing. After all RAG-related tests, the backend was stopped, and the database was restored from its exact `HEAD` blob. The restored SHA-256 was `e407c3a550b8dcbd631f99f9c04791f222158db753ea74cb4e3866c69da8f434`; it is clean in the final working tree.
- No application source files were changed for this acceptance run.
