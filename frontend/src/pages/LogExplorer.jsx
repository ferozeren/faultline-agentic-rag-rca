import { Fragment, useEffect, useMemo, useState } from "react";
import { AlertTriangle, RefreshCw, Search, SlidersHorizontal } from "lucide-react";
import PageHeader from "../components/common/PageHeader.jsx";
import DataUnavailable from "../components/common/DataUnavailable.jsx";
import ChartCard from "../components/dashboard/ChartCard.jsx";
import { getErrorMessage, getLogs } from "../services/api.js";
import { formatDate } from "../utils/formatters.js";

export default function LogExplorer() {
  const [logs, setLogs] = useState([]);
  const [service, setService] = useState("all");
  const [level, setLevel] = useState("all");
  const [query, setQuery] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [offset, setOffset] = useState(0);
  const [total, setTotal] = useState(0);
  const [excludedUntimed, setExcludedUntimed] = useState(0);
  const [excludedOutOfRange, setExcludedOutOfRange] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(null);

  async function load(event, pageOffset = offset) {
    event?.preventDefault();
    setLoading(true);
    setOffset(pageOffset);
    try {
      const response = await getLogs({
        service: service === "all" ? undefined : service,
        level: level === "all" ? undefined : level,
        search: query.trim() || undefined,
        from: from ? new Date(from).toISOString() : undefined,
        to: to ? new Date(to).toISOString() : undefined,
        limit: 100,
        offset: pageOffset,
      });
      setLogs((Array.isArray(response) ? response : Array.isArray(response?.logs) ? response.logs : []).filter(Boolean));
      setTotal(Number.isFinite(response?.total) ? response.total : 0);
      setExcludedUntimed(response?.excluded_untimed_count || 0);
      setExcludedOutOfRange(response?.excluded_outside_range_count || 0);
      setError("");
    } catch (requestError) {
      setLogs([]);
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => { load();   }, []);
  const summary = useMemo(() => ({
    errors: logs.filter((log) => ["error", "critical"].includes(String(log.level).toLowerCase())).length,
    warnings: logs.filter((log) => String(log.level).toLowerCase() === "warning").length,
    info: logs.filter((log) => String(log.level).toLowerCase() === "info").length,
  }), [logs]);
  const serviceOptions = useMemo(
    () => [...new Set([...logs.map((log) => log.service).filter(Boolean), ...(service === "all" ? [] : [service])])].sort(),
    [logs, service],
  );
  const volume = useMemo(() => {
    const groups = new Map();
    logs.forEach((log) => {
      if (!log.timestamp) return;
      const date = new Date(log.timestamp);
      if (Number.isNaN(date.getTime())) return;
      const key = `${date.getHours().toString().padStart(2, "0")}:00`;
      groups.set(key, (groups.get(key) || 0) + 1);
    });
    return [...groups.entries()].map(([label, count]) => ({ label, count }));
  }, [logs]);

  return (
    <main className="page-content">
      <PageHeader eyebrow="OBSERVABILITY" title="Log explorer" description="Search application logs with service, level, and time-window filters." actions={<button className="button button-secondary" onClick={load} disabled={loading}><RefreshCw size={14} className={loading ? "spin" : ""} /> Refresh</button>} />
      <form className="panel log-filter-panel" onSubmit={(event) => load(event, 0)}>
        <div className="list-toolbar"><label className="search-field"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search message text" /></label>
          <label className="filter-select"><SlidersHorizontal size={14} /><select value={service} onChange={(event) => setService(event.target.value)}><option value="all">All services</option>{serviceOptions.map((name) => <option key={name}>{name}</option>)}</select></label>
          <select className="filter-select" value={level} onChange={(event) => setLevel(event.target.value)}><option value="all">All levels</option><option>ERROR</option><option>CRITICAL</option><option>WARNING</option><option>INFO</option></select>
          <label className="time-field"><span>From</span><input type="datetime-local" value={from} onChange={(event) => setFrom(event.target.value)} /></label>
          <label className="time-field"><span>To</span><input type="datetime-local" value={to} onChange={(event) => setTo(event.target.value)} /></label>
          <button type="submit" className="button button-primary">Apply filters</button>
        </div>
      </form>
      {error && <DataUnavailable message={error} onRetry={load} />}
      <section className="log-summary-grid">
        <article className="panel summary-card"><span>Errors / critical</span><strong className="summary-red">{error ? "—" : summary.errors}</strong><small>From returned records</small></article>
        <article className="panel summary-card"><span>Warnings</span><strong className="summary-yellow">{error ? "—" : summary.warnings}</strong><small>From returned records</small></article>
        <article className="panel summary-card"><span>Info messages</span><strong>{error ? "—" : summary.info}</strong><small>From returned records</small></article>
        <article className="panel summary-card"><span>Records in view</span><strong>{error ? "—" : logs.length}</strong><small>{loading ? "Refreshing…" : "Current result set"}</small></article>
      </section>
      <ChartCard title="Log volume" subtitle="Counts grouped from the returned log records" type="bar" data={volume} dataKey="count" color="#69cfc5" />
      <section className="panel list-panel">
        <div className="panel-heading"><div><h2>Application logs</h2><p>{loading ? "Loading from backend…" : `${logs.length} records in this page · ${total} matching records`}</p></div><span className="chart-period">SERVER FILTERS</span></div>
        {(excludedUntimed > 0 || excludedOutOfRange > 0) && <div className="page-footnote">Excluded from this time-scoped result: {excludedUntimed} untimed · {excludedOutOfRange} outside the selected window.</div>}
        {!error && !logs.length ? <DataUnavailable title="No logs returned" message="Log data is only queried through the configured backend API. No local or fabricated logs are shown." /> :
          <div className="table-scroll"><table className="data-table log-table"><thead><tr><th>Timestamp</th><th>Level</th><th>Service</th><th>Message</th><th>Source</th></tr></thead><tbody>
            {logs.map((log, index) => <Fragment key={log.id || index}><tr className="expandable-row" onClick={() => setExpanded(expanded === index ? null : index)}><td>{formatDate(log.timestamp)}</td><td><span className={`log-level level-${String(log.level || "unknown").toLowerCase()}`}>{log.level || "—"}</span></td><td>{log.service || "—"}</td><td className="log-message">{log.message || "—"}</td><td>{log.source || "—"}</td></tr>{expanded === index && <tr className="expanded-detail"><td colSpan="5"><pre>{JSON.stringify(log.raw || log, null, 2)}</pre></td></tr>}</Fragment>)}
          </tbody></table></div>}
        <div className="heading-actions log-pagination"><button className="button button-secondary" disabled={loading || offset === 0} onClick={() => load(undefined, Math.max(0, offset - 100))}>Previous</button><span>{total ? `${offset + 1}–${Math.min(offset + logs.length, total)} of ${total}` : "No results"}</span><button className="button button-secondary" disabled={loading || offset + logs.length >= total} onClick={() => load(undefined, offset + 100)}>Next</button></div>
      </section>
      {error && <div className="page-footnote"><AlertTriangle size={13} /><span>Check the local Log MCP data source and the selected time range.</span></div>}
    </main>
  );
}
