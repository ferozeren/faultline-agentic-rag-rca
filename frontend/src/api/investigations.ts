import { apiClient } from "./client";
import type { IncidentRequest, InvestigationResponse } from "./contracts";

export async function submitInvestigation(
  request: IncidentRequest,
): Promise<InvestigationResponse> {
  const { data } = await apiClient.post<InvestigationResponse>(
    "/api/v1/investigations",
    request,
    { timeout: 300_000 },
  );
  return data;
}
