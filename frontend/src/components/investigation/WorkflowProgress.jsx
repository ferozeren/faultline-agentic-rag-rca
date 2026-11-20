import { Check, Circle, LoaderCircle } from "lucide-react";
import EmptyState from "../common/EmptyState.jsx";

export default function WorkflowProgress({ stages }) {
  const reportedStages = Array.isArray(stages)
    ? stages.filter((stage) => (
      stage
      && typeof stage === "object"
      && (typeof stage.name === "string" || typeof stage.stage === "string")
    ))
    : [];

  if (!reportedStages.length) {
    return <EmptyState title="Workflow stage data unavailable" description="No per-stage execution data was returned by the investigation backend." />;
  }

  return (
    <div className="workflow-progress">
      {reportedStages.map((record, index) => {
        const label = record.name || record.stage;
        const status = record.status || "unavailable";
        return (
          <div className="workflow-stage" key={label}>
            <span className={`workflow-stage-marker marker-${status}`}>
              {status === "completed" || status === "complete" ? <Check size={13} /> : status === "running" ? <LoaderCircle size={13} className="spin" /> : <Circle size={9} />}
            </span>
            <div><strong>{label}</strong><small>{record?.status ? String(record.status).replaceAll("_", " ") : "Status unavailable"}</small></div>
            {index < reportedStages.length - 1 && <span className="workflow-stage-line" />}
          </div>
        );
      })}
    </div>
  );
}
