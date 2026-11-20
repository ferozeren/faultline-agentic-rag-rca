import { useMemo, useState } from "react";
import EmptyState from "../common/EmptyState.jsx";

const sourceGroups = [
  ["Application logs", "application_log_evidence_ids"],
  ["Service metrics", "service_metric_evidence_ids"],
  ["Prometheus", "prometheus_evidence_ids"],
  ["Grafana", "grafana_evidence_ids"],
  ["Historical RAG", "historical_rag_evidence_ids"],
];

export default function EvidencePanel({ evidenceReferences = [], analysis }) {
  const [filter, setFilter] = useState("all");
  const historicalIds = useMemo(
    () => new Set(analysis?.historical_rag_evidence_ids || []),
    [analysis?.historical_rag_evidence_ids],
  );
  const visibleEvidence = evidenceReferences.filter((evidence) => {
    const historical = evidence.classification === "historical" || historicalIds.has(evidence.evidence_id);
    return filter === "all" || (filter === "historical" ? historical : !historical);
  });

  return (
    <section className="panel evidence-panel">
      <div className="panel-heading">
        <div><h2>Evidence analysis</h2><p>Current observations are distinct from historical context.</p></div>
        <span className="evidence-count">{evidenceReferences.length} references</span>
      </div>
      <div className="evidence-tabs" role="group" aria-label="Filter evidence">
        {[["all", "All evidence"], ["current", "Current"], ["historical", "Historical context"]].map(([value, label]) =>
          <button key={value} type="button" aria-pressed={filter === value} className={filter === value ? "evidence-tab active" : "evidence-tab"} onClick={() => setFilter(value)}>{label}</button>,
        )}
      </div>
      {visibleEvidence.length ? <div className="evidence-reference-list">
        {visibleEvidence.map((evidence) => {
          const historical = evidence.classification === "historical" || historicalIds.has(evidence.evidence_id);
          return <article className="evidence-reference" key={evidence.evidence_id}>
            <div className="evidence-reference-heading">
              <strong>{evidence.source_name}</strong>
              <span className={`evidence-classification ${historical ? "evidence-historical" : "evidence-current"}`}>{historical ? "Historical · context only" : "Current evidence"}</span>
            </div>
            <p>{evidence.description}</p>
            <div className="evidence-reference-meta">
              <span>{evidence.evidence_id}</span><span>{evidence.source_type}</span>
              {evidence.service && <span>Service: {evidence.service}</span>}
              {evidence.observed_at && <time dateTime={evidence.observed_at}>{evidence.observed_at}</time>}
            </div>
            {(evidence.linked_finding_ids.length > 0 || evidence.linked_recommendation_ids.length > 0) && <small>
              {evidence.linked_finding_ids.length > 0 && `Findings: ${evidence.linked_finding_ids.join(", ")}`}
              {evidence.linked_recommendation_ids.length > 0 && ` · Recommendations: ${evidence.linked_recommendation_ids.join(", ")}`}
            </small>}
          </article>;
        })}
      </div> : <EmptyState compact title={filter === "historical" ? "No historical evidence references" : "No evidence references returned"} description="Evidence details are shown only when included in the backend report." />}

      {analysis && <div className="evidence-analysis-details">
        <h3>Evidence coverage and uncertainty</h3>
        <div className="evidence-source-groups">
          {sourceGroups.map(([label, key]) => {
            const ids = analysis[key] || [];
            return <div key={key}><span>{label}</span><strong>{ids.length ? ids.join(", ") : "None reported"}</strong></div>;
          })}
        </div>
        {analysis.verified_facts.length > 0 && <section><h4>Verified facts</h4><ul>{analysis.verified_facts.map((fact, index) => <li key={`${fact.description}-${index}`}>{fact.description} <small>({fact.classification}; evidence: {fact.evidence_ids.join(", ")})</small></li>)}</ul></section>}
        {analysis.hypotheses.length > 0 && <section><h4>Hypotheses</h4><ul>{analysis.hypotheses.map((item) => <li key={item.hypothesis_id}>{item.description} <small>({item.status}; confidence {item.confidence})</small></li>)}</ul></section>}
        {analysis.evidence_gaps.length > 0 && <section><h4>Evidence gaps</h4><ul>{analysis.evidence_gaps.map((gap) => <li key={gap.gap_id}>{gap.description}{gap.impact ? ` — ${gap.impact}` : ""}</li>)}</ul></section>}
        {analysis.unresolved_conflicts.length > 0 && <section><h4>Unresolved conflicts</h4><ul>{analysis.unresolved_conflicts.map((conflict, index) => <li key={`${conflict.conflict_type}-${index}`}>{conflict.description} <small>(evidence: {conflict.evidence_ids.join(", ")})</small></li>)}</ul></section>}
        {analysis.contradictory_evidence_ids.length > 0 && <p className="evidence-conflicts">Contradictory evidence IDs: {analysis.contradictory_evidence_ids.join(", ")}</p>}
      </div>}
      {historicalIds.size > 0 && <p className="historical-evidence-note">Historical evidence is contextual only; it is not presented as a current observation.</p>}
    </section>
  );
}
