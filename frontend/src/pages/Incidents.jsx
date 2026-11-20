import { Archive, RotateCcw, Search, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import DataUnavailable from "../components/common/DataUnavailable.jsx";
import PageHeader from "../components/common/PageHeader.jsx";
import StatusBadge from "../components/common/StatusBadge.jsx";
import {
  archiveInvestigation,
  deleteInvestigation,
  getErrorMessage,
  getIncidents,
  restoreInvestigation,
} from "../services/api.js";
import { formatDate } from "../utils/formatters.js";

const PAGE_SIZE = 20;

export default function Incidents() {
  const [searchParams, setSearchParams] = useSearchParams();
  const search = searchParams.get("search") || "";
  const [searchInput, setSearchInput] = useState(search);
  const [serviceName, setServiceName] = useState("");
  const [severity, setSeverity] = useState("");
  const [status, setStatus] = useState("");
  const [includeArchived, setIncludeArchived] = useState(false);
  const [page, setPage] = useState(1);
  const [history, setHistory] = useState({ items: [], total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const [actingId, setActingId] = useState("");

  async function load() {
    setLoading(true);
    try {
      const result = await getIncidents({
        page,
        page_size: PAGE_SIZE,
        include_archived: includeArchived,
        service_name: serviceName.trim() || undefined,
        severity: severity || undefined,
        status: status || undefined,
        search: search || undefined,
      });
      if (result.items.length === 0 && page > 1) {
        setPage((current) => current - 1);
        return;
      }
      setHistory(result);
      setError("");
    } catch (requestError) {
      setHistory({ items: [], total: 0 });
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { setSearchInput(search); }, [search]);
  useEffect(() => { load(); }, [page, serviceName, severity, status, search, includeArchived]);

  async function changeHistory(item, action) {
    const title = item.title || "this investigation";
    if (action === "archive" && !window.confirm(
      `Archive "${title}"? It will be hidden from active history and can be restored.`,
    )) return;
    if (action === "delete" && !window.confirm(
      `Permanently delete "${title}" from PostgreSQL, including its report, evidence, and agent runs? This cannot be undone.`,
    )) return;

    setActingId(item.investigation_id);
    setActionError("");
    setActionMessage("");
    try {
      if (action === "archive") {
        await archiveInvestigation(item.investigation_id);
        setActionMessage("Investigation archived.");
      } else if (action === "restore") {
        await restoreInvestigation(item.investigation_id);
        setActionMessage("Investigation restored.");
      } else {
        await deleteInvestigation(item.investigation_id);
        setActionMessage("Investigation permanently deleted.");
      }
      await load();
    } catch (requestError) {
      setActionError(getErrorMessage(requestError));
    } finally {
      setActingId("");
    }
  }

  function submitSearch(event) {
    event.preventDefault();
    const next = new URLSearchParams(searchParams);
    const value = searchInput.trim();
    if (value) next.set("search", value);
    else next.delete("search");
    setPage(1);
    setSearchParams(next);
  }

  const totalPages = Math.max(1, Math.ceil(history.total / PAGE_SIZE));

  return (
    <main className="page-content">
      <PageHeader
        eyebrow="INCIDENT HISTORY"
        title="Investigations"
        description="Saved investigation history from PostgreSQL."
        actions={<Link className="button button-primary" to="/investigations/new">New investigation</Link>}
      />
      <section className="panel list-panel">
        <form className="table-filters" onSubmit={submitSearch}>
          <label className="search-field history-search">
            <Search size={15} />
            <input
              type="search"
              aria-label="Search investigations"
              value={searchInput}
              onChange={(event) => setSearchInput(event.target.value)}
              placeholder="Search incident, ID, or service"
            />
          </label>
          <button className="button button-secondary" type="submit">Search</button>
          <label className="filter-select">
            <span className="sr-only">Filter by service</span>
            <input aria-label="Filter by service" value={serviceName} onChange={(event) => { setPage(1); setServiceName(event.target.value); }} placeholder="Service name" />
          </label>
          <label className="filter-select">
            <span className="sr-only">Filter by severity</span>
            <select aria-label="Filter by severity" value={severity} onChange={(event) => { setPage(1); setSeverity(event.target.value); }}>
              <option value="">All severities</option>
              {["SEV-1", "SEV-2", "SEV-3", "SEV-4", "critical", "high", "medium", "low"].map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <label className="filter-select">
            <span className="sr-only">Filter by status</span>
            <select aria-label="Filter by status" value={status} onChange={(event) => { setPage(1); setStatus(event.target.value); }}>
              <option value="">All statuses</option>
              {["running", "completed", "partial", "partial_success", "insufficient_evidence", "failed", "no_evidence"].map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <label className="archived-filter">
            <input
              type="checkbox"
              checked={includeArchived}
              onChange={(event) => {
                setPage(1);
                setIncludeArchived(event.target.checked);
              }}
            />
            <span>Show archived</span>
          </label>
          <span className="table-result-count">{history.total} investigations</span>
        </form>
        {error && <DataUnavailable message={error} onRetry={load} />}
        {actionError && <div className="inline-notice notice-error" role="alert">{actionError}</div>}
        {actionMessage && <div className="inline-notice notice-success" role="status">{actionMessage}</div>}
        {loading ? <div className="empty-table">Loading saved investigations…</div> : !error && history.items.length === 0
          ? <div className="empty-table">{includeArchived ? "No investigations match these filters." : "No investigations have been saved yet."}</div>
          : !error && <div className="table-scroll"><table className="data-table">
            <thead><tr><th>Incident</th><th>Service</th><th>Severity</th><th>Status</th><th>Created</th><th>Actions</th></tr></thead>
            <tbody>{history.items.map((item) => <tr key={item.investigation_id}>
              <td>
                <Link className="incident-title-cell" to={`/investigations/${item.investigation_id}`}>{item.title}</Link>
                {item.incident_id && item.incident_id !== item.title && <span className="muted-cell">{item.incident_id}</span>}
              </td>
              <td>{item.service_name || "Not specified"}</td>
              <td>{item.severity || "Not specified"}</td>
              <td>
                <StatusBadge status={item.status} />
                {item.archived_at && <span className="history-archived-label">Archived</span>}
              </td>
              <td>{formatDate(item.created_at)}</td>
              <td>
                <div className="history-actions">
                  <Link className="text-link" to={`/investigations/${item.investigation_id}`}>View</Link>
                  {item.archived_at
                    ? <button className="button button-secondary history-action" type="button" disabled={Boolean(actingId)} onClick={() => changeHistory(item, "restore")} aria-label={`Restore investigation ${item.title}`}>
                      <RotateCcw size={13} /> Restore
                    </button>
                    : <button className="button button-secondary history-action" type="button" disabled={Boolean(actingId)} onClick={() => changeHistory(item, "archive")} aria-label={`Archive investigation ${item.title}`}>
                      <Archive size={13} /> Archive
                    </button>}
                  <button className="button button-danger history-action" type="button" disabled={Boolean(actingId)} onClick={() => changeHistory(item, "delete")} aria-label={`Delete investigation ${item.title} permanently`}>
                    <Trash2 size={13} /> Delete permanently
                  </button>
                </div>
              </td>
            </tr>)}</tbody>
          </table></div>}
        {!error && history.total > PAGE_SIZE && <div className="form-actions">
          <button className="button button-secondary" disabled={page <= 1 || loading} onClick={() => setPage((current) => current - 1)}>Previous</button>
          <span>Page {page} of {totalPages}</span>
          <button className="button button-secondary" disabled={page >= totalPages || loading} onClick={() => setPage((current) => current + 1)}>Next</button>
        </div>}
      </section>
    </main>
  );
}
