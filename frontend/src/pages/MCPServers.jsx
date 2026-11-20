import { useEffect, useMemo, useState } from "react";
import { ChevronRight, Radio, Server, Wrench } from "lucide-react";
import PageHeader from "../components/common/PageHeader.jsx";
import DataUnavailable from "../components/common/DataUnavailable.jsx";
import StatusBadge from "../components/common/StatusBadge.jsx";
import { getErrorMessage, getMCPStatus } from "../services/api.js";

export default function MCPServers() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);
  async function load() {
    try {
      setData(await getMCPStatus());
      setError("");
    } catch (requestError) {
      setData(null);
      setError(getErrorMessage(requestError));
    }
  }
  useEffect(() => { load(); }, []);
  const reportedServers = Array.isArray(data) ? data : [];
  const selectedServerId = selected?.split(":")[0];
  const selectedToolName = selected && selected.includes(":") ? selected.split(":")[1] : "";
  const selectedServer = useMemo(() => reportedServers.find((server) => server?.name === selectedServerId), [reportedServers, selectedServerId]);
  const actualTool = selectedServer?.tools?.find((tool) => tool.name === selectedToolName);

  return (
    <main className="page-content">
      <PageHeader eyebrow="TOOLING PLANE" title="MCP servers" description="Inspect backend-reported MCP connectivity, health probes, and discovered tool schemas." />
      {error && <DataUnavailable message={error} onRetry={load} />}
      <section className="mcp-server-grid">
        {reportedServers.map((server) => {
          const actualTools = Array.isArray(server.tools) ? server.tools : [];
          return <article className="panel mcp-server-card" key={server.name}>
            <div className="mcp-card-top"><div className="mcp-server-icon"><Server size={18} /></div><StatusBadge status={server.status || "unavailable"} label={String(server.status || "unavailable").replaceAll("_", " ")} /></div>
            <h2>{server.display_name || server.name}</h2><p>{server.message}</p>
            <div className="mcp-detail-list"><div><span>MCP process</span><strong>{server.mcp_connected ? "Connected" : "Unavailable"}</strong></div><div><span>Discovered tools</span><strong>{server.tool_count ?? actualTools.length}</strong></div><div><span>Data probe</span><strong>{server.records_available ?? "Not reported"}</strong></div>{server.http_status && <div><span>Upstream HTTP status</span><strong>{server.http_status}</strong></div>}</div>
            <div className="mcp-tool-list"><div className="tool-list-title"><span>Discovered tools</span><small>Reported by backend</small></div>
              {actualTools.map((tool) => <button key={tool.name} className={`mcp-tool-row ${selected === `${server.name}:${tool.name}` ? "selected" : ""}`} onClick={() => setSelected(`${server.name}:${tool.name}`)}><Wrench size={13} /><code>{tool.name}</code><ChevronRight size={14} /></button>)}
              {!actualTools.length && <div className="tool-select-empty">No tools were discovered for this server.</div>}
            </div>
          </article>;
        })}
        {!reportedServers.length && !error && <DataUnavailable title="MCP status unavailable" message="The backend has not returned a source-status report." />}
      </section>
      <section className="panel tool-explorer">
        <div className="panel-heading"><div><h2>Tool explorer</h2><p>Select a discovered tool to inspect its backend-provided schema.</p></div><span className="tool-explorer-icon"><Radio size={16} /></span></div>
        {actualTool ? <div className="tool-detail-grid"><div><span className="detail-label">Tool</span><code>{selectedToolName}</code><span className="detail-label">Server</span><p>{selectedServer?.display_name || selectedServerId}</p><span className="detail-label">Description</span><p>{actualTool.description || "No description provided."}</p></div><div><span className="detail-label">Input schema</span><pre>{JSON.stringify(actualTool.input_schema || {}, null, 2)}</pre></div></div> : <div className="tool-select-empty">Choose a discovered tool above to inspect it.</div>}
      </section>
      <div className="page-footnote"><span><Radio size={13} /> Browser is not connected directly to MCP.</span><span>Tool calls are backend-mediated only.</span></div>
    </main>
  );
}
