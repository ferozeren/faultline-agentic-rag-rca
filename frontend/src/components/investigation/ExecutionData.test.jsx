import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AgentTimeline from "./AgentTimeline.jsx";
import WorkflowProgress from "./WorkflowProgress.jsx";

describe("execution data displays", () => {
  it("does not invent agents or stages when the backend returns none", () => {
    render(
      <>
        <AgentTimeline />
        <WorkflowProgress />
      </>,
    );

    expect(screen.getByText("Agent execution data unavailable")).toBeInTheDocument();
    expect(screen.getByText("Workflow stage data unavailable")).toBeInTheDocument();
    expect(screen.queryByText("Supervisor Agent")).not.toBeInTheDocument();
    expect(screen.queryByText("RCA generation")).not.toBeInTheDocument();
  });

  it("renders execution records returned by the backend", () => {
    render(
      <>
        <AgentTimeline agents={[{ name: "Current log agent", status: "completed", activity: "Reviewed project logs." }]} />
        <WorkflowProgress stages={[{ name: "Evidence collection", status: "completed" }]} />
      </>,
    );

    expect(screen.getByText("Current log agent")).toBeInTheDocument();
    expect(screen.getByText("Reviewed project logs.")).toBeInTheDocument();
    expect(screen.getByText("Evidence collection")).toBeInTheDocument();
  });
});
