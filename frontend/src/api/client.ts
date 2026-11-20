import axios, { type AxiosError } from "axios";

const storedBaseURL =
  typeof localStorage === "undefined"
    ? null
    : localStorage.getItem("faultline-api-base-url") ??
      localStorage.getItem("sentinel-api-base-url");

export const apiClient = axios.create({
  baseURL:
    storedBaseURL !== null
      ? storedBaseURL
      : import.meta.env.VITE_API_BASE_URL || "",
  timeout: 15_000,
  headers: { "Content-Type": "application/json" },
});

apiClient.interceptors.request.use((config) => {
  const correlationId =
    globalThis.crypto?.randomUUID?.() ??
    `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  config.headers.set("X-Correlation-ID", correlationId);
  return config;
});

export function getErrorMessage(error: unknown): string {
  if (!axios.isAxiosError(error)) {
    return "An unexpected error occurred. Please try again.";
  }

  const apiError = error as AxiosError<{
    error?: unknown;
    message?: unknown;
    correlation_id?: unknown;
  }>;
  const body = apiError.response?.data;
  if (
    typeof body?.message === "string" &&
    typeof body.correlation_id === "string"
  ) {
    return `${body.message} (Reference: ${body.correlation_id})`;
  }
  if (apiError.response?.status === 422) {
    return "The incident details were not accepted. Check the fields and try again.";
  }
  if (apiError.response && apiError.response.status >= 500) {
    return "The backend could not complete the investigation. Try again later.";
  }
  if (apiError.code === "ECONNABORTED") {
    return "The request timed out before the backend returned a result.";
  }
  if (!apiError.response) {
    return "Could not reach the backend. Check the API URL and that the server is running.";
  }
  return "The backend returned an unexpected response.";
}

export async function checkBackendHealth(): Promise<{ status: "connected" | "disconnected"; error?: string }> {
  try {
    const response = await apiClient.get("/api/v1/health", { timeout: 4_000 });
    return {
      status: response.status >= 200 && response.status < 300 ? "connected" : "disconnected",
    };
  } catch (error) {
    return { status: "disconnected", error: getErrorMessage(error) };
  }
}
