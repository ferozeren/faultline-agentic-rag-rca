import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import App from "../App.jsx";
import { apiClient } from "../api/client.ts";
import { ThemeProvider } from "../context/ThemeContext.jsx";

const response = {
  investigation_id: "inv-123",
  status: "completed",
  incident: { description: "Payment requests are timing out." },
  supervisor_decision: null,
  report: {
    incident_id: null,
    incident_summary: "The investigation returned a partial analysis.",
    affected_service: null,
    investigation_status: "partial",
    observations: [],
    historical_findings: [],
    current_findings: [],
    root_cause: null,
    recommendations: [],
    evidence_gaps: [],
    limitations: [],
    generated_at: "2025-01-15T10:30:00Z",
    report_status: "partial",
    evidence_references: [],
    timeline: [],
    unresolved_issues: [],
    missing_sections: [],
    warnings: [],
    errors: [],
  },
  timeline: [],
  evidence_references: [],
  warnings: [],
  limitations: [],
};

describe("incident submission", () => {
  let originalAdapter;
  let requests;

  beforeEach(() => {
    originalAdapter = apiClient.defaults.adapter;
    requests = [];
    apiClient.defaults.adapter = async (config) => {
      requests.push(config);
      const data = config.url === "/api/v1/health"
        ? { status: "healthy", application: "Faultline Incident Response", api_version: "1.0.0" }
        : response;
      return { data, status: 200, statusText: "OK", headers: {}, config };
    };
  });

  afterEach(() => {
    apiClient.defaults.adapter = originalAdapter;
    localStorage.clear();
  });

  function renderForm() {
    return render(
      <MemoryRouter
        initialEntries={["/investigations/new"]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <ThemeProvider><App /></ThemeProvider>
      </MemoryRouter>,
    );
  }

  it("validates the required description without sending a workflow request", async () => {
    renderForm();
    fireEvent.click(screen.getByRole("button", { name: /submit investigation/i }));

    expect(await screen.findByText("Incident description is required.")).toBeInTheDocument();
    expect(requests.filter((config) => config.method === "post")).toHaveLength(0);
  });

  it("submits once and displays the returned report, including partial status", async () => {
    renderForm();
    fireEvent.change(screen.getByLabelText(/description/i), {
      target: { value: "Payment requests are timing out." },
    });
    fireEvent.change(screen.getByLabelText(/affected service/i), {
      target: { value: "payments-api" },
    });
    fireEvent.click(screen.getByRole("button", { name: /submit investigation/i }));

    expect(await screen.findByText("The investigation returned a partial analysis.")).toBeInTheDocument();
    await waitFor(() => {
      expect(requests.filter((config) => config.method === "post")).toHaveLength(1);
    });
    expect(requests.find((config) => config.method === "post").url).toBe("/api/v1/investigations");
    expect(JSON.parse(requests.find((config) => config.method === "post").data)).toEqual({
      description: "Payment requests are timing out.",
      affected_service: "payments-api",
    });
  });
});
