import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import DataUnavailable from "../components/common/DataUnavailable.jsx";
import PageHeader from "../components/common/PageHeader.jsx";
import StatusBadge from "../components/common/StatusBadge.jsx";
import { getErrorMessage, getReports } from "../services/api.js";
import { formatDate } from "../utils/formatters.js";

const PAGE_SIZE = 20;

export default function Reports() {
  const [page, setPage] = useState(1);
  const [history, setHistory] = useState({ items: [], total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function load() {
    setLoading(true);
    try {
      setHistory(await getReports({ page, page_size: PAGE_SIZE }));
      setError("");
    } catch (requestError) {
      setHistory({ items: [], total: 0 });
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, [page]);
  const totalPages = Math.max(1, Math.ceil(history.total / PAGE_SIZE));

  return (
    <main className="page-content">
      <PageHeader
        eyebrow="STRUCTURED REPORTS"
        title="Saved reports"
        description="Reports persisted with each investigation."
        actions={<Link className="button button-primary" to="/investigations/new">New investigation</Link>}
      />
      <section className="panel list-panel">
        {error && <DataUnavailable message={error} onRetry={load} />}
        {loading ? <div className="empty-table">Loading saved reports…</div> : !error && history.items.length === 0
          ? <div className="empty-table">No saved reports yet.</div>
          : !error && <div className="table-scroll"><table className="data-table">
            <thead><tr><th>Investigation</th><th>Incident</th><th>Status</th><th>Created</th><th /></tr></thead>
            <tbody>{history.items.map((item) => <tr key={item.investigation_id}>
              <td>{item.report_available
                ? <Link className="text-link" to={`/investigations/${item.investigation_id}/report`}>{item.investigation_id}</Link>
                : item.investigation_id}</td>
              <td>{item.service_name || item.incident_id || item.title}</td>
              <td><StatusBadge status={item.status} /></td>
              <td>{formatDate(item.created_at)}</td>
              <td>{item.report_available
                ? <Link className="text-link" to={`/investigations/${item.investigation_id}/report`}>View report</Link>
                : "Report unavailable"}</td>
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
