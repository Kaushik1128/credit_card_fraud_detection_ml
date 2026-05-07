/**
 * Typed HTTP client for the FastAPI fraud detection service.
 *
 * The base URL comes from NEXT_PUBLIC_API_URL (set in .env.local).
 * Variables prefixed NEXT_PUBLIC_ are inlined into the browser bundle
 * at build time — never put secrets here.
 */
import type {
  BatchPredictionResponse,
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
  Transaction,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store", // ML predictions are not cacheable
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "<no body>");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return (await res.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getModelInfo(): Promise<ModelInfoResponse> {
  return request<ModelInfoResponse>("/model-info");
}

export function predictOne(transaction: Transaction): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(transaction),
  });
}

export function predictMany(
  transactions: Transaction[],
): Promise<BatchPredictionResponse> {
  return request<BatchPredictionResponse>("/predict/batch", {
    method: "POST",
    body: JSON.stringify({ transactions }),
  });
}
