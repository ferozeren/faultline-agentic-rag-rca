import { apiClient, checkBackendHealth, getErrorMessage as getApiErrorMessage } from "../api/client.ts";
import { submitInvestigation } from "../api/investigations.ts";

export { apiClient, checkBackendHealth };

export class BackendContractUnavailableError extends Error {
  constructor(feature) {
    super(`Backend API for ${feature} is not implemented or documented in this workspace.`);
    this.name = "BackendContractUnavailableError";
    this.feature = feature;
  }
}

export function getErrorMessage(error) {
  if (error instanceof BackendContractUnavailableError) return error.message;
  return getApiErrorMessage(error);
}

export async function createIncident(requestBody) {
  return submitInvestigation(requestBody);
}

export async function startInvestigation(requestBody) {
  return submitInvestigation(requestBody);
}

export async function getInvestigation(investigationId) {
  const { data } = await apiClient.get(
    `/api/v1/investigations/${encodeURIComponent(investigationId)}`,
  );
  return data;
}

export async function getDashboardData() {
  const { data } = await apiClient.get("/api/v1/overview", { timeout: 60_000 });
  return data;
}

export async function getIncidents(params) {
  const { data } = await apiClient.get("/api/v1/investigations", {
    params: {
      ...params,
      search: params?.search?.trim() || undefined,
    },
  });
  return data;
}

export async function getInvestigationEvidence(investigationId) {
  const { data } = await apiClient.get(
    `/api/v1/investigations/${encodeURIComponent(investigationId)}/evidence`,
  );
  return data;
}

export async function getInvestigationAgents(investigationId) {
  const { data } = await apiClient.get(
    `/api/v1/investigations/${encodeURIComponent(investigationId)}/agents`,
  );
  return data;
}

export async function getInvestigationStats(params) {
  const { data } = await apiClient.get("/api/v1/investigations/stats", { params });
  return data;
}

export async function archiveInvestigation(investigationId) {
  await apiClient.post(
    `/api/v1/investigations/${encodeURIComponent(investigationId)}/archive`,
  );
}

export async function restoreInvestigation(investigationId) {
  await apiClient.post(
    `/api/v1/investigations/${encodeURIComponent(investigationId)}/restore`,
  );
}

export async function deleteInvestigation(investigationId) {
  await apiClient.delete(
    `/api/v1/investigations/${encodeURIComponent(investigationId)}`,
  );
}

export async function getLogs(params) {
  const { data } = await apiClient.get("/api/v1/data/logs", {
    params: {
      service_name: params?.service_name ?? params?.service,
      level: params?.level,
      search: params?.search,
      start_time: params?.start_time ?? params?.from,
      end_time: params?.end_time ?? params?.to,
      limit: params?.limit,
      offset: params?.offset,
    },
  });
  return data;
}

export async function getMetrics(params) {
  const { data } = await apiClient.get("/api/v1/data/metrics", {
    params: {
      service_name: params?.service_name ?? params?.service,
      start_time: params?.start_time ?? params?.from,
      end_time: params?.end_time ?? params?.to,
    },
  });
  return data;
}

export async function getMCPStatus() {
  const { data } = await apiClient.get("/api/v1/mcp/status", { timeout: 60_000 });
  return data;
}

export async function getRAGStatus() {
  const { data } = await apiClient.get("/api/v1/data/knowledge/status", { timeout: 30_000 });
  return data;
}

export async function searchHistoricalIncidents(params) {
  const { data } = await apiClient.get("/api/v1/data/knowledge/search", {
    params,
    timeout: 60_000,
  });
  return data;
}

export async function getReports(params) {
  const { data } = await apiClient.get("/api/v1/investigations", { params });
  return data;
}