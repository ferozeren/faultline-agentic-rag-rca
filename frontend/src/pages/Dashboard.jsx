import { Activity, Database, FileText, RefreshCw, Server } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import PageHeader from "../components/common/PageHeader.jsx";
import DataUnavailable from "../components/common/DataUnavailable.jsx";
import StatusBadge from "../components/common/StatusBadge.jsx";
import { getDashboardData, getErrorMessage, getIncidents, getInvestigationStats } from "../services/api.js";
import { formatDate } from "../utils/formatters.js";

export default function Dashboard() {
  const [latestHistoryItem, setLatestHistoryItem] = useState(null);
  const [overview, setOverview] = useState(null);
  const [overviewError, setOverviewError] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyStats, setHistoryStats] = useState(null);
  const [historyError, setHistoryError] = useState("");
  const [historyLoading, setHistoryLoading] = useState(true);

  async function loadOverview() {
    setLoading(true);
    try {
      setOverview(await getDashboardData());
      setOverviewError("");
    } catch (error) {
      setOverview(null);
      setOverviewError(getErrorMessage(error));
    } finally {
      setLoading(false);
    }
  }

  async function loadHistoryStats() {
    setHistoryLoading(true);
    try {
      const [stats, latest] = await Promise.all([
        getInvestigationStats(),
        getIncidents({ page: 1, page_size: 1 }),
      ]);
      setHistoryStats(stats);
      setLatestHistoryItem(latest.items?.[0] || null);
      setHistoryError("");
    } catch (error) {
      setHistoryStats(null);
      setLatestHistoryItem(null);
      setHistoryError(getErrorMessage(error));
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => { loadOverview(); loadHistoryStats(); }, []);

  return (
    <main className="page-content dashboard-page">
      <PageHeader
        eyebrow="INCIDENT INVESTIGATION"
        title="Investigation workspace"
        description="Submit an incident to the existing workflow and review the structured response."
        actions={<><button className="button button-secondary" onClick={loadOverview} disabled={loading}><RefreshCw size={14} className={loading ? "spin" : ""} /> Refresh status</button><Link className="button button-primary" to="/investigations/new"><Activity size={15} /> New investigation</Link></>}
      />
      {overviewError && <DataUnavailable message={overviewError} onRetry={loadOverview} />}
      <section className="panel operations-overview">
        <div className="panel-heading"><div><h2>Operations overview</h2><p>Live local datasets, historical store, and upstream MCP checks.</p></div><StatusBadge status={overview?.status || "unavailable"} label={overview?.status || (loading ? "checking" : "unavailable")} /></div>
        <div className="operations-summary-grid">
          <article className="operations-summary-card"><span><Activity size={15} /> API process</span><StatusBadge status={overview?.application_status || "unavailable"} label={overview?.application_status || "unavailable"} /><small>Application liveness; external dependencies are checked separately.</small></article>
          <article className="operations-summary-card"><span><Database size={15} /> Historical knowledge</span><StatusBadge status={overview?.vector_store?.status || "unavailable"} label={overview?.vector_store?.status || "unavailable"} /><strong>{overview?.vector_store?.document_count ?? "—"} chunks</strong><small>{overview?.vector_store?.collection_name || "Vector store status unavailable"}</small></article>
          <article className="operations-summary-card"><span><Server size={15} /> Local datasets</span><strong>{overview?.datasets?.reduce((sum, item) => sum + (item.record_count || 0), 0) ?? "—"} records</strong><small>{overview?.datasets?.length ? overview.datasets.map((item) => `${item.name}: ${item.record_count}`).join(" · ") : "Dataset counts unavailable"}</small></article>
        </div>
        <div className="operations-source-grid">
          {(overview?.mcp_servers || []).map((server) => <article className="operations-source-card" key={server.name}>
            <div><strong>{server.display_name}</strong><StatusBadge status={server.status} label={server.status.replaceAll("_", " ")} /></div>
            <p>{server.message}</p>
            <small>{server.tool_count} tools · {server.records_available ?? "—"} records{server.http_status ? ` · HTTP ${server.http_status}` : ""}</small>
          </article>)}
          {!overview?.mcp_servers?.length && <div className="empty-table">{loading ? "Checking operational sources…" : "Operational source status is unavailable."}</div>}
        </div>
        {overview?.checked_at && <small className="operations-checked-at">Last checked {formatDate(overview.checked_at)}</small>}
      </section>
      <section className="panel latest-investigation-panel">
        <div className="panel-heading">
          <div><h2>Investigation history</h2><p>Saved investigations in PostgreSQL</p></div>
          <Link className="text-link" to="/incidents">View history</Link>
        </div>
        {historyError && <DataUnavailable message={historyError} onRetry={loadHistoryStats} />}
        {historyLoading ? <div className="empty-table">Loading saved investigation statistics…</div>
          : historyStats && <div className="operations-summary-grid">
            <article className="operations-summary-card"><span>Total investigations</span><strong>{historyStats.total_investigations}</strong></article>
            <article className="operations-summary-card"><span>Completed</span><strong>{historyStats.by_status.completed || 0}</strong></article>
            <article className="operations-summary-card"><span>Partial or insufficient evidence</span><strong>{(historyStats.by_status.partial || 0) + (historyStats.by_status.insufficient_evidence || 0)}</strong></article>
          </div>}
        {!historyLoading && !historyError && historyStats?.total_investigations === 0 && <div className="empty-table">No investigation history has been recorded yet.</div>}
      </section>
      <section className="panel latest-investigation-panel">
        <div className="panel-heading"><div><h2>Latest investigation</h2><p>Most recent record saved in PostgreSQL</p></div>{latestHistoryItem && <StatusBadge status={latestHistoryItem.status} />}</div>
        {historyLoading ? <div className="empty-table">Loading investigation history…</div>
          : latestHistoryItem ? <div className="latest-investigation-content">
            <div><span>Investigation ID</span><strong>{latestHistoryItem.investigation_id}</strong></div>
            <div><span>Incident</span><strong>{latestHistoryItem.incident_description}</strong></div>
            <div><span>Service</span><strong>{latestHistoryItem.service_name || "Not specified"}</strong></div>
            <div><span>Created</span><strong>{formatDate(latestHistoryItem.created_at)}</strong></div>
            <div className="heading-actions"><Link className="button button-secondary" to={`/investigations/${latestHistoryItem.investigation_id}`}><Activity size={14} /> View result</Link><Link className="button button-secondary" to={`/investigations/${latestHistoryItem.investigation_id}/report`}><FileText size={14} /> Full report</Link></div>
          </div> : !historyError && <div className="empty-table">No investigation history has been recorded yet.</div>}
      </section>
    </main>
  );
}
