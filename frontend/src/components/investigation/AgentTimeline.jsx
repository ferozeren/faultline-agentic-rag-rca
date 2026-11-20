import { Activity, Check, Circle, FileSearch, GitBranch, LoaderCircle, Radio, X } from "lucide-react";
import EmptyState from "../common/EmptyState.jsx";
import { formatDate, humanize } from "../../utils/formatters.js";

const icons = [GitBranch, FileSearch, Radio, Activity];

function AgentIcon({ status, icon: Icon }) {
  if (status === "completed" || status === "complete") return <Check size={16} />;
  if (status === "running") return <LoaderCircle size={16} className="spin" />;
  if (status === "failed" || status === "error") return <X size={16} />;
  return <Icon size={16} />;
}

export default function AgentTimeline({ agents }) {
  const reportedAgents = Array.isArray(agents)
    ? agents.filter((agent) => (
      agent
      && typeof agent === "object"
      && (typeof agent.name === "string" || typeof agent.agent_name === "string")
    ))
    : [];

  if (!reportedAgents.length) {
    return <EmptyState title="Agent execution data unavailable" description="The investigation response did not include per-agent execution events." />;
  }

  return (
    <div className="agent-timeline">
      {reportedAgents.map((agent, index) => {
        const name = agent.name || agent.agent_name;
        const status = agent?.status ? String(agent.status).toLowerCase() : "unavailable";
        return (
          <div className="agent-step" key={name}>
            <div className={`agent-step-icon agent-${status}`}><AgentIcon status={status} icon={icons[index % icons.length]} /></div>
            <div className="agent-step-content">
              <div className="agent-step-head"><strong>{name}</strong><span className="agent-status-text">{humanize(status)}</span></div>
              <p>{agent?.activity || agent?.description || agent?.error_message || (agent?.result ? "Execution result was returned by the backend." : "No execution details were returned.")}</p>
              <div className="agent-step-meta">
                {agent?.started_at && <span>Started {formatDate(agent.started_at)}</span>}
                {agent?.completed_at && <span>Completed {formatDate(agent.completed_at)}</span>}
                {agent?.result?.evidence_count !== undefined && <span>{agent.result.evidence_count} evidence items</span>}
                {agent?.duration_ms !== null && agent?.duration_ms !== undefined && <span>{agent.duration_ms} ms</span>}
              </div>
            </div>
            <div className="agent-step-rail" />
          </div>
        );
      })}
    </div>
  );
}
