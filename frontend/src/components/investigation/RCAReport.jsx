import { AlertTriangle, CheckCircle2, Copy } from "lucide-react";
import { useState } from "react";
import EmptyState from "../common/EmptyState.jsx";
import { formatDate } from "../../utils/formatters.js";

function valueText(value) {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "string") return value;
  if (typeof value === "object" && !Array.isArray(value) && typeof value.text === "string") return value.text;
  return JSON.stringify(value, null, 2);
}

function ValueBlock({ value }) {
  const text = valueText(value);
  if (!text) return <p>Not provided by the backend.</p>;
  if (text.startsWith("{") || text.startsWith("[")) return <pre className="report-json">{text}</pre>;
  return <p>{text}</p>;
}

export default function RCAReport({ report, incident, recommendations, timeline = [] }) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState("");

  async function copyReport() {
    try {
      await navigator.clipboard.writeText(JSON.stringify({ incident, report }, null, 2));
      setCopied(true);
      setCopyError("");
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopyError("Could not copy the report. Clipboard access is unavailable.");
    }
  }

  if (!report) return <EmptyState title="No report returned" description="The investigation response did not include a report object." />;

  const rca = report.root_cause_analysis;
  const finding = rca?.primary_finding;
  const actions = recommendations?.recommendations || [];

  return (
    <article className="panel rca-report">
      <header className="report-header">
        <div><div className="eyebrow">STRUCTURED INVESTIGATION REPORT</div><h2>{incident?.affected_service || "Incident report"}</h2><p>{incident?.incident_id || "Incident ID not supplied"} · {incident?.environment || "Environment not supplied"} · Generated {formatDate(report.generated_at)}</p></div>
        <div className="heading-actions"><button type="button" className="button button-secondary" onClick={copyReport}><Copy size={14} />{copied ? "Copied" : "Copy report JSON"}</button></div>
      </header>
      {copyError && <p className="field-error" role="alert">{copyError}</p>}

      <section className={`root-cause-card ${finding ? "root-cause-unverified" : ""}`}>
        <div className="root-cause-heading">{finding ? <AlertTriangle size={17} /> : <CheckCircle2 size={17} />}<strong>{finding ? "RCA finding (not an execution directive)" : "No primary root-cause finding returned"}</strong></div>
        {finding && <>
          <div className="root-cause-content"><strong>{finding.summary}</strong><p>{finding.root_cause}</p><p>{finding.causal_reasoning}</p></div>
          <small>Finding ID: {finding.finding_id} · Confidence: {finding.confidence} · {finding.confidence_explanation}</small>
          {finding.supporting_evidence_ids.length > 0 && <small>Supporting evidence: {finding.supporting_evidence_ids.join(", ")}</small>}
          {finding.contradicting_evidence_ids.length > 0 && <small>Contradicting evidence: {finding.contradicting_evidence_ids.join(", ")}</small>}
        </>}
        {!finding && <ValueBlock value={report.root_cause} />}
        {rca?.analysis_summary && <p>{rca.analysis_summary}</p>}
        {rca?.status && <small>RCA status: {rca.status}</small>}
      </section>

      <section className="report-section report-narrative">
        <h3>Executive summary</h3>
        <ValueBlock value={report.executive_summary?.summary || report.incident_summary} />
        {report.executive_summary?.investigation_outcome && <ValueBlock value={report.executive_summary.investigation_outcome} />}
      </section>

      <section className="report-section report-narrative">
        <h3>Conclusion and methodology</h3>
        <ValueBlock value={report.conclusion?.summary} />
        {report.conclusion?.methodology?.length > 0 && <ul>{report.conclusion.methodology.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>}
        {report.conclusion?.limitations?.length > 0 && <ul>{report.conclusion.limitations.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>}
      </section>

      <section className="report-section">
        <h3>Recommendations · advisory only</h3>
        <p>{recommendations?.overall_summary || "No recommendation summary was returned."} Infrastructure changes are not executed by this interface.</p>
        {actions.length ? <div className="recommendation-list">{actions.map((action) => <article className="recommendation-item" key={action.recommendation_id}>
          <div className="recommendation-heading"><strong>{action.title}</strong><span>{action.category} · {action.priority}</span></div>
          <p>{action.description}</p>
          <p><strong>Why:</strong> {action.rationale}</p>
          <small>Findings: {action.related_finding_ids.join(", ")} · Evidence: {action.supporting_evidence_ids.join(", ")}</small>
          <ul>{action.verification_steps.map((step, index) => <li key={`${step}-${index}`}>{step}</li>)}</ul>
          {action.requires_human_approval && <strong className="approval-required">Human approval required</strong>}
          {action.implementation_risk && <small>Implementation risk: {action.implementation_risk}</small>}
        </article>)}</div> : <p>No recommendations were returned.</p>}
        {recommendations?.errors?.length > 0 && <ul>{recommendations.errors.map((error, index) => <li key={`${error}-${index}`}>{error}</li>)}</ul>}
      </section>

      <section className="report-section">
        <h3>Investigation timeline</h3>
        {timeline.length ? <ol className="report-timeline">{timeline.map((event) => <li key={event.event_id}>
          <div><strong>{event.description}</strong><small>{event.event_type} · {event.source} · {event.service || "Service not specified"} · {event.timestamp ? formatDate(event.timestamp) : "Time unavailable"}</small><small>Evidence: {event.evidence_ids.join(", ")}</small></div>
        </li>)}</ol> : <p>No timeline events were returned.</p>}
      </section>

      {(report.evidence_gaps.length > 0 || report.unresolved_issues.length > 0) && <section className="report-section">
        <h3>Open gaps and issues</h3>
        {report.evidence_gaps.length > 0 && <ul>{report.evidence_gaps.map((gap) => <li key={gap.gap_id}>{gap.description}{gap.impact ? ` — ${gap.impact}` : ""}</li>)}</ul>}
        {report.unresolved_issues.length > 0 && <ul>{report.unresolved_issues.map((issue, index) => <li key={index}>{valueText(issue)}</li>)}</ul>}
      </section>}

      <details className="report-complete-data">
        <summary>View complete validated report data</summary>
        <pre className="report-json">{JSON.stringify(report, null, 2)}</pre>
      </details>
    </article>
  );
}
