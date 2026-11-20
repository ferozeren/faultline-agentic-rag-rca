export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface IncidentRequest {
  incident_id?: string | null;
  description: string;
  affected_service?: string | null;
  environment?: string | null;
  severity?: string | null;
  start_time?: string | null;
  end_time?: string | null;
  additional_context?: { [key: string]: JsonValue } | null;
}

export interface EvidenceGap {
  gap_id: string;
  description: string;
  missing_source_types: string[];
  impact?: string | null;
  recommended_data_sources: string[];
}

export interface EvidenceConflict {
  conflict_type: string;
  evidence_ids: string[];
  description: string;
}

export interface EvidenceReference {
  evidence_id: string;
  source_type: string;
  source_name: string;
  description: string;
  observed_at?: string | null;
  time_range_start?: string | null;
  time_range_end?: string | null;
  service?: string | null;
  severity?: string | null;
  raw_reference?: string | null;
  metadata: { [key: string]: JsonValue };
  classification: "historical" | "current";
  linked_finding_ids: string[];
  linked_hypothesis_ids: string[];
  linked_recommendation_ids: string[];
}

export interface RCAEvidenceClaim {
  description: string;
  evidence_ids: string[];
  classification: "historical" | "current";
}

export interface RCAFinding {
  finding_id: string;
  summary: string;
  root_cause: string;
  affected_services: string[];
  contributing_factors: RCAEvidenceClaim[];
  triggering_event?: RCAEvidenceClaim | null;
  downstream_impact: RCAEvidenceClaim[];
  supporting_evidence_ids: string[];
  contradicting_evidence_ids: string[];
  confidence: number;
  confidence_explanation: string;
  causal_reasoning: string;
}

export interface RCAHypothesis {
  hypothesis_id: string;
  description: string;
  supporting_evidence_ids: string[];
  contradicting_evidence_ids: string[];
  confidence: number;
  status: string;
}

export interface RCATimelineEvent {
  event_id: string;
  timestamp?: string | null;
  service?: string | null;
  source: string;
  description: string;
  evidence_ids: string[];
  event_type: string;
}

export interface Recommendation {
  recommendation_id: string;
  title: string;
  description: string;
  category: "corrective" | "preventive" | "diagnostic";
  priority: "critical" | "high" | "medium" | "low";
  urgency: "immediate" | "within_24_hours" | "planned" | "informational";
  rationale: string;
  supporting_evidence_ids: string[];
  related_finding_ids: string[];
  affected_services: string[];
  expected_impact?: string | null;
  implementation_risk?: string | null;
  implementation_effort?: string | null;
  prerequisites: string[];
  verification_steps: string[];
  rollback_considerations: string[];
  confidence: number;
  requires_human_approval: boolean;
}

export interface RecommendationResult {
  status: string;
  incident_id?: string | null;
  recommendations: Recommendation[];
  corrective_action_count: number;
  preventive_action_count: number;
  diagnostic_action_count: number;
  overall_summary: string;
  source_rca_finding_ids: string[];
  evidence_gaps: string[];
  unresolved_conflicts: string[];
  warnings: string[];
  errors: string[];
  generated_at: string;
}

export interface InvestigationReport {
  incident_id?: string | null;
  incident_summary: string;
  affected_service?: string | null;
  investigation_status: string;
  observations: JsonValue[];
  historical_findings: JsonValue[];
  current_findings: JsonValue[];
  root_cause?: JsonValue | null;
  recommendations: JsonValue[];
  evidence_gaps: EvidenceGap[];
  limitations: string[];
  generated_at: string;
  report_status?: string | null;
  executive_summary?: JsonValue | null;
  incident_overview?: JsonValue | null;
  evidence_references: EvidenceReference[];
  evidence_analysis?: {
    historical_rag_evidence_ids: string[];
    application_log_evidence_ids: string[];
    service_metric_evidence_ids: string[];
    prometheus_evidence_ids: string[];
    grafana_evidence_ids: string[];
    verified_facts: RCAEvidenceClaim[];
    hypotheses: RCAHypothesis[];
    contradictory_evidence_ids: string[];
    evidence_gaps: EvidenceGap[];
    unresolved_conflicts: EvidenceConflict[];
  } | null;
  timeline: RCATimelineEvent[];
  root_cause_analysis?: {
    status?: string | null;
    primary_finding?: RCAFinding | null;
    alternative_hypotheses: RCAHypothesis[];
    verified_facts: RCAEvidenceClaim[];
    evidence_gaps: EvidenceGap[];
    unresolved_conflicts: EvidenceConflict[];
    confidence?: number | null;
    confidence_explanation?: string | null;
    analysis_summary: string;
    limitations: string[];
  } | null;
  recommendation_result?: RecommendationResult | null;
  recommendation_sections?: JsonValue | null;
  unresolved_issues: JsonValue[];
  conclusion?: JsonValue | null;
  missing_sections: string[];
  warnings: string[];
  errors: string[];
}

export interface InvestigationResponse {
  investigation_id: string;
  status: string;
  incident: IncidentRequest;
  supervisor_decision?: {
    decision_status: string;
    evidence_sufficiency: string;
    decision_rationale: string;
    confidence: number;
  } | null;
  report: InvestigationReport;
  root_cause_analysis?: InvestigationReport["root_cause_analysis"];
  recommendations?: RecommendationResult | null;
  timeline: RCATimelineEvent[];
  evidence_analysis?: InvestigationReport["evidence_analysis"];
  evidence_references: EvidenceReference[];
  conclusion?: JsonValue | null;
  warnings: string[];
  limitations: string[];
}
