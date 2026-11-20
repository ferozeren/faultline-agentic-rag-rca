import { useState } from "react";
import { Check, Moon, Save, Sun } from "lucide-react";
import { useOutletContext } from "react-router-dom";
import PageHeader from "../components/common/PageHeader.jsx";
import StatusBadge from "../components/common/StatusBadge.jsx";
import { useTheme } from "../context/ThemeContext.jsx";
import { apiClient } from "../services/api.js";
import { environmentName } from "../utils/constants.js";

const defaultApiUrl = import.meta.env.VITE_API_BASE_URL || "";

export default function Settings() {
  const { theme, toggleTheme } = useTheme();
  const { health, notify } = useOutletContext();
  const [apiUrl, setApiUrl] = useState(
    localStorage.getItem("faultline-api-base-url")
      ?? localStorage.getItem("sentinel-api-base-url")
      ?? defaultApiUrl,
  );
  const [refreshInterval, setRefreshInterval] = useState(
    localStorage.getItem("faultline-refresh-seconds")
      ?? localStorage.getItem("sentinel-refresh-seconds")
      ?? "30",
  );
  const [saved, setSaved] = useState(false);

  function save() {
    const trimmed = apiUrl.trim();
    let normalized = "";
    if (trimmed) {
      try {
        const parsed = new URL(trimmed);
        if (!["http:", "https:"].includes(parsed.protocol)) throw new Error("Use an http or https URL.");
        normalized = parsed.origin;
      } catch (error) {
        notify({ type: "error", message: `Invalid backend URL: ${error.message}` });
        return;
      }
    }
    localStorage.setItem("faultline-api-base-url", normalized);
    localStorage.setItem("faultline-refresh-seconds", refreshInterval);
    apiClient.defaults.baseURL = normalized;
    window.dispatchEvent(new Event("faultline-settings-change"));
    health.refresh();
    setApiUrl(normalized);
    setSaved(true);
    window.setTimeout(() => setSaved(false), 1800);
    notify({ type: "success", message: "Workspace preferences saved." });
  }

  return (
    <main className="page-content">
      <PageHeader eyebrow="WORKSPACE CONFIGURATION" title="Settings" description="Configure local preferences and the backend address. No credentials are stored here." />
      <div className="settings-layout">
        <section className="panel settings-section">
          <div className="settings-title"><div><h2>Backend connection</h2><p>API URL used by this browser session.</p></div><StatusBadge status={health.status} label={health.status} /></div>
          <label className="field"><span>Backend API URL</span><input value={apiUrl} onChange={(event) => setApiUrl(event.target.value)} placeholder="Leave blank to use the Vite API proxy" /><small className="field-help">Health is checked at <code>/api/v1/health</code>. Set an absolute URL only when the backend allows this frontend origin through CORS.</small></label>
        </section>
        <section className="panel settings-section">
          <div className="settings-title"><div><h2>Appearance</h2><p>Choose a workspace color theme.</p></div></div>
          <div className="theme-choice-row">
            <button className={`theme-choice ${theme === "dark" ? "selected" : ""}`} onClick={() => theme !== "dark" && toggleTheme()}><Moon size={17} /><span><strong>Dark</strong><small>Low-light command center</small></span>{theme === "dark" && <Check size={16} />}</button>
            <button className={`theme-choice ${theme === "light" ? "selected" : ""}`} onClick={() => theme !== "light" && toggleTheme()}><Sun size={17} /><span><strong>Light</strong><small>Bright workspace surfaces</small></span>{theme === "light" && <Check size={16} />}</button>
          </div>
        </section>
        <section className="panel settings-section">
          <div className="settings-title"><div><h2>Health checks</h2><p>Control how often the backend liveness endpoint is checked.</p></div></div>
          <label className="field setting-inline"><span>Refresh interval</span><select value={refreshInterval} onChange={(event) => setRefreshInterval(event.target.value)}><option value="15">15 seconds</option><option value="30">30 seconds</option><option value="60">60 seconds</option></select></label>
        </section>
        <section className="panel settings-section">
          <div className="settings-title"><div><h2>Application information</h2><p>Runtime status and configuration.</p></div></div>
          <div className="app-info-grid"><span>Application</span><strong>Faultline Incident Response</strong><span>Environment</span><strong>{environmentName}</strong><span>Backend liveness</span><strong>{health.status}</strong><span>API credentials</span><strong>Not stored in frontend</strong></div>
        </section>
        <div className="settings-save"><button className="button button-primary" onClick={save}><Save size={14} />{saved ? "Saved" : "Save preferences"}</button></div>
      </div>
    </main>
  );
}
