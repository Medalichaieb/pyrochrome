/**
 * Typed client for the Pyrochrome FastAPI service.
 *
 * The base URL comes from `VITE_API_URL` (default: local dev server). Network
 * and service errors are surfaced as `ApiError` so the UI can render a calm
 * offline state rather than throwing raw fetch failures.
 */
import type { PredictRequest, PredictResponse } from "./types";

const BASE_URL = (import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

/** Raised for any non-2xx response or network failure, with a human message. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly cause?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Health probe: which models the server has loaded. */
export interface Health {
  status: string;
  classifiers: string[];
  colour_lab: boolean;
  neighbours: boolean;
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch (cause) {
    throw new ApiError(`Cannot reach the API at ${BASE_URL}. Is it running?`, cause);
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new ApiError(`API error ${response.status}: ${detail || response.statusText}`);
  }
  return (await response.json()) as T;
}

/** Fetch the service health (loaded models). */
export function getHealth(): Promise<Health> {
  return getJson<Health>("/health");
}

/** Request a render prediction for a glaze. */
export function predict(request: PredictRequest): Promise<PredictResponse> {
  return getJson<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export { BASE_URL };
