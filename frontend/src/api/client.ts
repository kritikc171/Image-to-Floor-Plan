/**
 * Typed API client for the Floor Plan Generator backend.
 * Presentation-layer only: no credentials, no AI logic.
 */

export const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: init?.body ? {} : { Accept: "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(0, "network_error", "Cannot reach the server.");
  }

  const requestId = response.headers.get("X-Request-ID") ?? undefined;
  if (!response.ok) {
    let code = "request_failed";
    let message = "Something went wrong.";
    try {
      const body = (await response.json()) as { error?: { code?: string; message?: string } };
      code = body.error?.code ?? code;
      message = body.error?.message ?? message;
    } catch {
      // non-JSON error body; keep generic message
    }
    const error = new ApiError(response.status, code, message) as ApiError & {
      requestId?: string;
    };
    error.requestId = requestId;
    throw error;
  }
  return (await response.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export async function createGeneration(
  file: File,
): Promise<GenerationCreateResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<GenerationCreateResponse>("/api/generations", {
    method: "POST",
    body: form,
  });
}

export async function getGeneration(
  jobId: string,
): Promise<GenerationStatusResponse> {
  return request<GenerationStatusResponse>(`/api/generations/${encodeURIComponent(jobId)}`);
}

export function floorplanUrl(jobId: string): string {
  return `${API_BASE}/api/generations/${encodeURIComponent(jobId)}/floorplan`;
}

export interface HealthResponse {
  status: string;
  version: string;
}

export interface GenerationCreateResponse {
  job_id: string;
  status: string;
  created_at: string;
}

export interface GenerationStatusResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  error: { code: string; message: string } | null;
  result: GenerationResult | null;
}

export interface GenerationResult {
  schema_version: string;
  job_id: string;
  created_at: string;
  input: { width_px: number; height_px: number; format: string };
  hypothesis: SpatialHypothesis;
  floorplan_image_url: string;
}

export interface SpatialHypothesis {
  schema_version: string;
  room_boundary: RoomBoundary;
  objects: SpatialObject[];
  openings: Opening[];
  uncertainty: UncertaintySummary;
  provenance: Provenance;
}

export interface RoomBoundary {
  polygon: { x: number; y: number }[] | null;
  evidence_status: EvidenceStatus;
  uncertainty_reason: string | null;
}

export interface SpatialObject {
  id: string;
  type: string;
  label: string | null;
  position: { x: number; y: number } | null;
  orientation: number | null;
  dimensions: Record<string, number> | null;
  evidence_status: EvidenceStatus;
  confidence: number | null;
  evidence_source: string[];
  uncertainty_reason: string | null;
}

export interface Opening {
  id: string;
  kind: string;
  side: string | null;
  position: { x: number; y: number } | null;
  evidence_status: EvidenceStatus;
  uncertainty_reason: string | null;
}

export interface UncertaintySummary {
  overall: "low" | "medium" | "high";
  assumptions: string[];
  open_questions: string[];
}

export interface Provenance {
  reasoning_provider: string;
  depth_method: string;
  feature_method: string;
  generated_views_used: number;
  model_name: string | null;
}

export type EvidenceStatus =
  | "OBSERVED"
  | "INFERRED"
  | "GENERATED"
  | "UNCERTAIN";
