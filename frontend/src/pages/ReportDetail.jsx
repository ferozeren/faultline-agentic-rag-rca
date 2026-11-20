import { ArrowLeft } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import PageHeader from "../components/common/PageHeader.jsx";
import DataUnavailable from "../components/common/DataUnavailable.jsx";
import RCAReport from "../components/investigation/RCAReport.jsx";
import { useInvestigation } from "../hooks/useInvestigation.js";

export default function ReportDetail() {
  const { id } = useParams();
  const { investigation, loading, error } = useInvestigation(id);
  const incident = investigation?.incident;

  return (
    <main className="page-content">
      <PageHeader
        eyebrow="STRUCTURED REPORT"
        title={incident?.affected_service || "Investigation report"}
        description={`Investigation ${investigation?.investigation_id || id}`}
        actions={<Link className="button button-secondary" to={`/investigations/${id}`}><ArrowLeft size={14} /> Investigation result</Link>}
      />
      {loading && <div className="empty-table">Loading saved report…</div>}
      {error && <DataUnavailable title="Report unavailable" message={error} />}
      {investigation && <RCAReport report={investigation.report} incident={incident} recommendations={investigation.recommendations || investigation.report.recommendation_result} timeline={investigation.timeline || investigation.report.timeline} />}
    </main>
  );
}
