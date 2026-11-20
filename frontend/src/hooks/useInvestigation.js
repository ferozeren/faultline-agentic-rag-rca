import { useEffect, useState } from "react";
import {
  getErrorMessage,
  getInvestigation,
  getInvestigationAgents,
} from "../services/api.js";

export function useInvestigation(investigationId) {
  const [investigation, setInvestigation] = useState(null);
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(Boolean(investigationId));
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    if (!investigationId) {
      setInvestigation(null);
      setAgents([]);
      setLoading(false);
      setError("");
      return () => { active = false; };
    }

    setLoading(true);
    setError("");
    Promise.all([
      getInvestigation(investigationId),
      getInvestigationAgents(investigationId),
    ])
      .then(([result, agentResult]) => {
        if (!active) return;
        setInvestigation(result);
        setAgents(Array.isArray(agentResult?.agents) ? agentResult.agents : []);
      })
      .catch((requestError) => {
        if (!active) return;
        setInvestigation(null);
        setAgents([]);
        setError(getErrorMessage(requestError));
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => { active = false; };
  }, [investigationId]);

  return { investigation, agents, loading, error };
}
