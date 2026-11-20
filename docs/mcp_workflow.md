# Standalone Metrics MCP Server

From the repository root, start the Metrics MCP Server with:

```bash
cd backend
.venv/bin/python -m app.mcp_servers.metrics_mcp_server
```

The server uses stdio transport and reads `data/metrics/processed/metrics.csv`.
It exposes operational metrics through its tools and the `metrics://available`
resource. The evaluation-only `scenario_label` and `is_anomaly` columns are
excluded. This starts the standalone server; it does not connect it to an MCP
client.

## Standalone Prometheus MCP Server

Export `PROMETHEUS_URL` and, when required, `PROMETHEUS_BEARER_TOKEN` in the
server process environment. The backend reads these settings from environment
variables and the Prometheus server loads only its own settings from
`backend/.env` without overriding exported values. For Grafana Cloud, configure
`PROMETHEUS_USERNAME` and use the Prometheus API token in
`PROMETHEUS_BEARER_TOKEN`; the server uses HTTP Basic authentication. For other
endpoints, a token without a username is sent as a Bearer token. Then start the
server from the repository root:

```bash
cd backend
.venv/bin/python -m app.mcp_servers.prometheus_mcp_server
```

The server uses stdio transport and exposes Prometheus query, target, alert,
and health APIs through its MCP tools and `prometheus://available` resource.
It does not connect to an MCP client.

To run read-only live connectivity checks against the configured instance:

```bash
cd backend
.venv/bin/python -m app.mcp_servers.validate_prometheus_connection
```