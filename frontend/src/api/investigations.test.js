import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { apiClient } from "./client.ts";
import { submitInvestigation } from "./investigations.ts";

describe("submitInvestigation", () => {
  let originalAdapter;
  let requests;

  beforeEach(() => {
    originalAdapter = apiClient.defaults.adapter;
    requests = [];
    apiClient.defaults.adapter = async (config) => {
      requests.push(config);
      return {
        data: { investigation_id: "inv-123", status: "completed" },
        status: 200,
        statusText: "OK",
        headers: {},
        config,
      };
    };
  });

  afterEach(() => {
    apiClient.defaults.adapter = originalAdapter;
  });

  it("posts the incident schema once to the versioned investigation endpoint", async () => {
    const request = {
      description: "Payment API requests are timing out.",
      affected_service: "payments-api",
    };

    await submitInvestigation(request);

    expect(requests).toHaveLength(1);
    expect(requests[0].url).toBe("/api/v1/investigations");
    expect(requests[0].method).toBe("post");
    expect(JSON.parse(requests[0].data)).toEqual(request);
    expect(requests[0].timeout).toBe(300_000);
    expect(requests[0].headers.get("X-Correlation-ID")).toBeTruthy();
  });
});
