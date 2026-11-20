import { useState } from "react";
import { AlertTriangle, ArrowLeft, ArrowRight, Check, FileText } from "lucide-react";
import { Link, useNavigate, useOutletContext } from "react-router-dom";
import PageHeader from "../components/common/PageHeader.jsx";
import { useLatestInvestigation } from "../context/InvestigationContext.jsx";
import { submitInvestigation } from "../api/investigations.ts";
import { getErrorMessage } from "../services/api.js";
import { severityOptions } from "../utils/constants.js";

const initialForm = {
  incident_id: "",
  description: "",
  affected_service: "",
  environment: "",
  severity: "",
  start_time: "",
  end_time: "",
  additional_context: "",
};

function toIsoDateTime(value) {
  if (!value) return undefined;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

export default function NewIncident() {
  const [form, setForm] = useState(initialForm);
  const [errors, setErrors] = useState({});
  const [requestError, setRequestError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { notify } = useOutletContext();
  const { saveInvestigation } = useLatestInvestigation();
  const navigate = useNavigate();

  function update(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: "" }));
    setRequestError("");
  }

  function clearForm() {
    setForm(initialForm);
    setErrors({});
    setRequestError("");
  }

  async function submit(event) {
    event.preventDefault();
    if (isSubmitting) return;

    const nextErrors = {};
    if (!form.description.trim()) nextErrors.description = "Incident description is required.";
    const startTime = toIsoDateTime(form.start_time);
    const endTime = toIsoDateTime(form.end_time);
    if (form.start_time && !startTime) nextErrors.start_time = "Enter a valid start time.";
    if (form.end_time && !endTime) nextErrors.end_time = "Enter a valid end time.";
    if (startTime && endTime && new Date(endTime) < new Date(startTime)) {
      nextErrors.end_time = "End time must not be earlier than start time.";
    }

    let additionalContext;
    if (form.additional_context.trim()) {
      try {
        const parsed = JSON.parse(form.additional_context);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("Additional context must be a JSON object.");
        }
        additionalContext = parsed;
      } catch {
        nextErrors.additional_context = "Enter a valid JSON object.";
      }
    }

    setErrors(nextErrors);
    setRequestError("");
    if (Object.keys(nextErrors).length) return;

    const request = {
      description: form.description.trim(),
      ...(form.incident_id.trim() && { incident_id: form.incident_id.trim() }),
      ...(form.affected_service.trim() && { affected_service: form.affected_service.trim() }),
      ...(form.environment.trim() && { environment: form.environment.trim() }),
      ...(form.severity && { severity: form.severity }),
      ...(startTime && { start_time: startTime }),
      ...(endTime && { end_time: endTime }),
      ...(additionalContext && { additional_context: additionalContext }),
    };

    setIsSubmitting(true);
    try {
      const result = await submitInvestigation(request);
      saveInvestigation(result);
      notify({ type: "success", message: "Investigation saved. Review its evidence, status, and limitations." });
      navigate(`/investigations/${encodeURIComponent(result.investigation_id)}`);
    } catch (error) {
      setRequestError(getErrorMessage(error));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="page-content form-page">
      <PageHeader
        eyebrow="INCIDENT INTAKE"
        title="New investigation"
        description="Submit incident details to the existing investigation workflow and review its returned report."
        actions={<Link className="button button-secondary" to="/incidents"><ArrowLeft size={15} /> Back to history</Link>}
      />
      <div className="form-layout">
        <form className="panel investigation-form" onSubmit={submit} noValidate>
          <div className="form-section-title"><span className="section-index">01</span><div><h2>Incident details</h2><p>Only fields accepted by the backend incident schema are sent.</p></div></div>
          <div className="form-grid">
            <label className="field">
              <span>Incident ID <em>Optional</em></span>
              <input value={form.incident_id} onChange={(event) => update("incident_id", event.target.value)} placeholder="Existing incident identifier" />
            </label>
            <label className="field">
              <span>Affected service <em>Optional</em></span>
              <input value={form.affected_service} onChange={(event) => update("affected_service", event.target.value)} placeholder="Enter the affected service name" />
            </label>
            <label className="field field-wide">
              <span>Description <i>*</i></span>
              <textarea value={form.description} onChange={(event) => update("description", event.target.value)} placeholder="Describe the incident symptoms and observed impact..." rows={5} maxLength={4000} aria-invalid={Boolean(errors.description)} required />
              {errors.description && <small className="field-error">{errors.description}</small>}
            </label>
            <label className="field">
              <span>Environment <em>Optional</em></span>
              <input value={form.environment} onChange={(event) => update("environment", event.target.value)} placeholder="Enter the environment name" />
            </label>
            <label className="field">
              <span>Severity <em>Optional</em></span>
              <select value={form.severity} onChange={(event) => update("severity", event.target.value)}>
                <option value="">Not specified</option>
                {severityOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Start time <em>Optional</em></span>
              <input type="datetime-local" value={form.start_time} onChange={(event) => update("start_time", event.target.value)} aria-invalid={Boolean(errors.start_time)} />
              {errors.start_time && <small className="field-error">{errors.start_time}</small>}
            </label>
            <label className="field">
              <span>End time <em>Optional</em></span>
              <input type="datetime-local" value={form.end_time} onChange={(event) => update("end_time", event.target.value)} aria-invalid={Boolean(errors.end_time)} />
              {errors.end_time && <small className="field-error">{errors.end_time}</small>}
            </label>
            <label className="field field-wide">
              <span>Additional context <em>Optional JSON object</em></span>
              <textarea value={form.additional_context} onChange={(event) => update("additional_context", event.target.value)} placeholder="Enter incident-specific context as a JSON object" rows={3} aria-invalid={Boolean(errors.additional_context)} />
              {errors.additional_context && <small className="field-error">{errors.additional_context}</small>}
            </label>
          </div>
          {requestError && <div className="inline-notice notice-error" role="alert"><AlertTriangle size={17} /><div><strong>Investigation could not be completed</strong><span>{requestError}</span></div></div>}
          <div className="form-actions">
            <button type="button" className="button button-ghost" onClick={clearForm} disabled={isSubmitting}>Clear form</button>
            <button type="submit" className="button button-primary" disabled={isSubmitting}>
              <span>{isSubmitting ? "Running investigation…" : "Submit investigation"}</span><ArrowRight size={15} />
            </button>
          </div>
        </form>
        <aside className="form-aside">
          <section className="panel intake-note">
            <div className="note-icon"><Check size={17} /></div>
            <h3>One coordinated workflow</h3>
            <p>The existing LangGraph workflow runs once per submission. Its response includes the report and any available evidence, RCA, recommendations, warnings, and limitations.</p>
            <div className="note-flow"><span>Incident</span><i /><span>Workflow</span><i /><span>Report</span></div>
          </section>
          <section className="panel schema-note">
            <div className="schema-note-header"><FileText size={15} /><span>Saved investigation history</span></div>
            <p>Each submission is saved to PostgreSQL with its report, available evidence, and agent run details.</p>
            <Link to="/incidents">View investigation history <ArrowRight size={13} /></Link>
          </section>
        </aside>
      </div>
    </main>
  );
}
