import { AlertTriangle, ArrowLeft, FileText } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import DataUnavailable from "../components/common/DataUnavailable.jsx";
import PageHeader from "../components/common/PageHeader.jsx";
import StatusBadge from "../components/common/StatusBadge.jsx";
import EvidencePanel from "../components/investigation/EvidencePanel.jsx";
import AgentTimeline from "../components/investigation/AgentTimeline.jsx";
import RCAReport from "../components/investigation/RCAReport.jsx";
import { useInvestigation } from "../hooks/useInvestigation.js";
import { formatDate } from "../utils/formatters.js";

function unique(values) {
  return [...new Set(values.filter((value) => typeof value === "string" && value.trim()))];
}

export default function InvestigationResult() {
  const { id } = useParams();
  const { investigation, agents, loading, error } = useInvestigation(id);
  const incident = investigation?.incident;
  const report = investigation?.report;
  const evidenceAnalysis = investigation?.evidence_analysis || report?.evidence_analysis;
  const warnings = unique([
    ...(investigation?.warnings || []),
    ...(investigation?.limitations || []),
    ...(report?.warnings || []),
    ...(report?.limitations || []),
    ...(report?.missing_sections || []).map((section) => `Report section unavailable: ${section}`),
    ...(report?.errors || []),
  ]);
  const investigationOutcome =
    report?.executive_summary?.investigation_outcome?.text ||
    "The report did not include an explicit investigation outcome.";

  return (
    <main className="page-content result-page">
      <PageHeader
        eyebrow="INVESTIGATION RESULT"
        title={incident?.affected_service || "Investigation report"}
        description={`Investigation ${investigation?.investigation_id || id}`}
        actions={<><Link className="button button-secondary" to="/incidents"><ArrowLeft size={14} /> Latest result</Link><Link className="button button-primary" to={`/investigations/${id}/report`}><FileText size={14} /> Full report</Link></>}
      />
      {loading && <div className="empty-table">Loading saved investigation…</div>}
      {error && <DataUnavailable title="Result unavailable" message={error} />}
      {investigation && !report && <section className="panel incident-overview">
        <div className="incident-overview-title"><div><span className="muted-cell">INVESTIGATION ID</span><strong>{investigation.investigation_id}</strong></div><StatusBadge status={investigation.status} /></div>
        <p className="result-incident-description">{incident.description}</p>
        <DataUnavailable title="Report unavailable" message="This investigation did not produce a validated report. Its status and failure are retained in history." />
      </section>}
      {investigation && report && <>
        <section className="panel incident-overview">
          <div className="incident-overview-title">
            <div><span className="muted-cell">INVESTIGATION ID</span><strong>{investigation.investigation_id}</strong></div>
            <StatusBadge status={report.report_status || report.investigation_status} />
          </div>
          <div className="incident-meta-grid">
            <div><span>API response status</span><strong>{investigation.status}</strong></div>
            <div><span>Incident ID</span><strong>{incident.incident_id || "Not supplied"}</strong></div>
            <div><span>Affected service</span><strong>{incident.affected_service || "Not supplied"}</strong></div>
            <div><span>Severity</span><strong>{incident.severity || "Not supplied"}</strong></div>
          </div>
          <p className="result-incident-description">{incident.description}</p>
        </section>

        {warnings.length > 0 && <div className="inline-notice notice-warning"><AlertTriangle size={17} /><div><strong>Warnings and limitations</strong><ul>{warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul></div></div>}

        <div className="result-grid">
          <section className="panel result-summary">
            <div className="panel-heading"><div><h2>Investigation summary</h2><p>Returned by the report-generation agent</p></div></div>
            <p>{investigationOutcome}</p>
            <small>Report generated {formatDate(report.generated_at)}</small>
          </section>
          <section className="panel result-summary">
            <div className="panel-heading"><div><h2>Supervisor decision</h2><p>Evidence sufficiency and workflow routing</p></div><StatusBadge status={investigation.supervisor_decision?.decision_status || "not provided"} /></div>
            {investigation.supervisor_decision
              ? <><p>{investigation.supervisor_decision.decision_rationale}</p><small>Evidence sufficiency: {investigation.supervisor_decision.evidence_sufficiency} · Confidence: {investigation.supervisor_decision.confidence}</small></>
              : <p>The response did not include a supervisor decision.</p>}
          </section>
        </div>

        <EvidencePanel evidenceReferences={investigation.evidence_references || report.evidence_references || []} analysis={evidenceAnalysis} />
        <section className="panel"><div className="panel-heading"><div><h2>Specialist agent runs</h2><p>Execution data saved from the investigation workflow.</p></div></div><AgentTimeline agents={agents} /></section>
        <RCAReport report={report} incident={incident} recommendations={investigation.recommendations || report.recommendation_result} timeline={investigation.timeline || report.timeline} />
      </>}
    </main>
  );
}
