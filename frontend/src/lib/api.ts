import type {
  AgentStatus,
  AttributionResponse,
  EvalResultsResponse,
  GraphResponse,
  HeatmapCell,
  RecoveryBreakdownRow,
  RecoveryFeedbackInput,
  RecoveryStats,
  SessionListResponse,
} from "./types";

/**
 * Typed API client against the AgentReflex backend.
 *
 * Every call attaches `Authorization: Bearer <key>` from the current auth
 * context. A 401 raises AuthError so the UI can redirect to the settings
 * screen; any network failure raises ApiError which views render as a
 * helpful state rather than a blank page.
 */

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export class AuthError extends Error {
  constructor(message: string) {
    super(message);
  }
}

const BASE_URL = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

interface Options {
  key?: string | null;
}

async function request<T>(path: string, opts: Options & { method?: string; body?: unknown } = {}): Promise<T> {
  const key = opts.key ?? null;
  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (key) headers["Authorization"] = `Bearer ${key}`;
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method: opts.method ?? "GET",
      headers,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
    });
  } catch {
    throw new ApiError(0, "Cannot reach the AgentReflex API. Is the backend running?");
  }

  if (response.status === 401) {
    throw new AuthError("API key is missing, invalid, or revoked.");
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  listSessions: (params: Record<string, string | number | undefined>, key: string) => {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== "") qs.set(k, String(v));
    }
    const suffix = qs.size > 0 ? `?${qs.toString()}` : "";
    return request<SessionListResponse>(`/traces${suffix}`, { key });
  },
  getAttribution: (sessionId: string, key: string) =>
    request<AttributionResponse>(`/traces/${encodeURIComponent(sessionId)}/attribution`, { key }),
  getGraph: (sessionId: string, key: string) =>
    request<GraphResponse>(`/traces/${encodeURIComponent(sessionId)}/graph`, { key }),
  agentsStatus: (key: string) => request<AgentStatus[]>("/agents/status", { key }),
  heatmap: (key: string) => request<HeatmapCell[]>("/stats/heatmap", { key }),
  recoveryStats: (key: string) => request<RecoveryStats>("/recovery/stats", { key }),
  recoveryBreakdown: (key: string) => request<RecoveryBreakdownRow[]>("/stats/recovery-breakdown", { key }),
  evalResults: (key: string) => request<EvalResultsResponse>("/eval/results", { key }),
  recoveryFeedback: (input: RecoveryFeedbackInput, key: string) =>
    request<{ status: string }>("/recovery/feedback", { method: "POST", body: input, key }),
};