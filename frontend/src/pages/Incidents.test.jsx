import { AxiosError } from "axios";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { apiClient } from "../api/client.ts";
import Incidents from "./Incidents.jsx";

describe("investigation history", () => {
  let originalAdapter;
  let requests;
  let archivedIds;
  let deletedIds;
  let confirmSpy;

  beforeEach(() => {
    originalAdapter = apiClient.defaults.adapter;
    requests = [];
    archivedIds = new Set();
    deletedIds = new Set();
    confirmSpy = null;
    apiClient.defaults.adapter = async (config) => {
      requests.push(config);
      const method = config.method?.toLowerCase();
      const archiveMatch = config.url.match(/\/investigations\/([^/]+)\/(archive|restore)$/);
      if (archiveMatch && method === "post") {
        const [, id, action] = archiveMatch;
        if (action === "archive") archivedIds.add(id);
        else archivedIds.delete(id);
        return { data: null, status: 204, statusText: "No Content", headers: {}, config };
      }
      const deleteMatch = config.url.match(/\/investigations\/([^/]+)$/);
      if (deleteMatch && method === "delete") {
        deletedIds.add(deleteMatch[1]);
        return { data: null, status: 204, statusText: "No Content", headers: {}, config };
      }

      const params = config.params || {};
      const pageItems = params.page === 2 ? [{
        investigation_id: "00000000-0000-0000-0000-000000000022",
        title: "Later saved investigation",
        service_name: "identity-service",
        status: "completed",
        created_at: "2026-11-19T10:00:00Z",
      }] : [{
        investigation_id: "00000000-0000-0000-0000-000000000011",
        title: "Database connections timed out",
        incident_id: "INC-22",
        service_name: "checkout",
        severity: "SEV-2",
        status: "partial",
        created_at: "2026-11-19T09:00:00Z",
      }];
      const items = pageItems
        .filter((item) => !deletedIds.has(item.investigation_id))
        .filter((item) => params.include_archived || !archivedIds.has(item.investigation_id))
        .map((item) => ({
          ...item,
          archived_at: archivedIds.has(item.investigation_id) ? "2026-11-20T10:00:00Z" : null,
        }));
      return {
        data: {
          items,
          total: params.service_name ? items.length : pageItems.length === 1 && (archivedIds.size || deletedIds.size) ? items.length : 21,
          page: params.page || 1,
          page_size: 20,
        },
        status: 200,
        statusText: "OK",
        headers: {},
        config,
      };
    };
  });

  afterEach(() => {
    apiClient.defaults.adapter = originalAdapter;
    confirmSpy?.mockRestore();
  });

  function renderHistory(initialEntry = "/incidents") {
    render(
      <MemoryRouter
        initialEntries={[initialEntry]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Incidents />
      </MemoryRouter>,
    );
  }

  it("renders persisted investigations and applies filters", async () => {
    renderHistory();

    expect(await screen.findByText("Database connections timed out")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Filter by service"), {
      target: { value: "checkout" },
    });

    await waitFor(() => {
      expect(requests.some((request) => request.params?.service_name === "checkout")).toBe(true);
    });
    expect(screen.getByText("1 investigations")).toBeInTheDocument();
  });

  it("paginates using server history results", async () => {
    renderHistory();

    fireEvent.click(await screen.findByRole("button", { name: "Next" }));

    expect(await screen.findByText("Later saved investigation")).toBeInTheDocument();
    expect(requests.some((request) => request.params?.page === 2)).toBe(true);
  });

  it("applies global-search query parameters to saved history", async () => {
    renderHistory("/incidents?search=Database");

    expect(await screen.findByText("Database connections timed out")).toBeInTheDocument();
    expect(screen.getByLabelText("Search investigations")).toHaveValue("Database");
    expect(requests.some((request) => request.params?.search === "Database")).toBe(true);
  });

  it("shows a truthful empty state when PostgreSQL history is empty", async () => {
    apiClient.defaults.adapter = async (config) => ({
      data: { items: [], total: 0, page: 1, page_size: 20 },
      status: 200,
      statusText: "OK",
      headers: {},
      config,
    });

    renderHistory();

    expect(await screen.findByText("No investigations have been saved yet.")).toBeInTheDocument();
  });

  it("shows an error and retry action when history cannot load", async () => {
    apiClient.defaults.adapter = async () => {
      throw new AxiosError("History API unavailable.", "ERR_NETWORK");
    };

    renderHistory();

    expect(await screen.findByText(/Could not reach the backend/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("archives and restores an investigation with confirmation", async () => {
    confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    renderHistory();

    expect(await screen.findByText("Database connections timed out")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", {
      name: "Archive investigation Database connections timed out",
    }));

    expect(await screen.findByText("Investigation archived.")).toBeInTheDocument();
    expect(await screen.findByText("No investigations have been saved yet.")).toBeInTheDocument();
    expect(requests.some((request) => (
      request.method === "post"
      && request.url.endsWith("/00000000-0000-0000-0000-000000000011/archive")
    ))).toBe(true);

    fireEvent.click(screen.getByLabelText("Show archived"));
    expect(await screen.findByText("Database connections timed out")).toBeInTheDocument();
    expect(requests.some((request) => request.params?.include_archived === true)).toBe(true);
    fireEvent.click(screen.getByRole("button", {
      name: "Restore investigation Database connections timed out",
    }));

    expect(await screen.findByText("Investigation restored.")).toBeInTheDocument();
    expect(confirmSpy).toHaveBeenCalledTimes(1);
  });

  it("requires confirmation before permanently deleting a record", async () => {
    confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    renderHistory();

    expect(await screen.findByText("Database connections timed out")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", {
      name: "Delete investigation Database connections timed out permanently",
    }));

    expect(confirmSpy).toHaveBeenCalledOnce();
    expect(requests.some((request) => request.method === "delete")).toBe(false);
  });

  it("permanently deletes a record after confirmation", async () => {
    confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    renderHistory();

    expect(await screen.findByText("Database connections timed out")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", {
      name: "Delete investigation Database connections timed out permanently",
    }));

    expect(await screen.findByText("Investigation permanently deleted.")).toBeInTheDocument();
    expect(requests.some((request) => request.method === "delete")).toBe(true);
    expect(await screen.findByText("No investigations have been saved yet.")).toBeInTheDocument();
  });
});
