const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/api/v1/evaluation${path}`, {
      credentials: "include",
      ...options,
    });
  } catch {
    throw new Error("Evaluation is unavailable. Check your connection and try again.");
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = response.status === 401 ? "Your session has expired. Please log in again."
      : response.status === 403 ? "Administrator access is required."
      : typeof payload?.detail === "string" ? payload.detail : "Evaluation could not finish.";
    throw new Error(message);
  }
  return payload as T;
}

export type QualityMetric = {
  id: string; priority: "HIGH" | "MEDIUM"; status: string; score: number | null;
  numerator: number | null; denominator: number | null; reason: string; evidence_refs: string[];
};

export type QualityRow = {
  conversation_id: string; message_id: string; paper_id: string; paper_filename?: string;
  question: string; revision: number; evaluated_at: string; evidence_type: string; state: string;
  high_status: string; medium_status: string; metrics: Record<string, QualityMetric>;
};

export type QualityMonitor = {
  rows: QualityRow[]; total_versions: number; evaluated_answers: number;
  counts: { complete: number; pending: number; failed: number; unavailable: number };
  metrics: Record<string, { average: number | null; measured: number; numerator: number; denominator: number }>;
};

export type QualityDetail = {
  question: string; paper_id: string; generated_answer?: string | null; answer?: string | null;
  evaluator_version: string; revision: number; response_ms?: number; evaluation_ms?: number;
  model_calls?: { model_id: string; role: string; provider: string; latency_ms: number;
    status?: number; error?: string; cached?: boolean; total_tokens?: number | null }[];
  metrics: Record<string, QualityMetric>;
  requirements?: { subject: string; dimension: string; addressed: boolean }[];
  sources?: { id: string; page_number?: number; title?: string; url?: string }[];
  claims?: { claim_id: string; text: string; section: string; status: string; reason?: string;
    attached_ids: string[]; supporting_ids: string[]; evaluated_ids: string[];
    supporting: { source_id: string; quote: string }[]; opposing: { source_id: string; quote: string }[] }[];
};

export function listAnswerQuality(query: string, signal: AbortSignal) {
  return request<QualityMonitor>(`/answer-quality?${query}`, { signal, cache: "no-store" });
}

export function getAnswerQuality(row: QualityRow, signal: AbortSignal) {
  return request<QualityDetail>(`/answer-quality/${encodeURIComponent(row.conversation_id)}/${encodeURIComponent(row.message_id)}/${row.revision}`, { signal, cache: "no-store" });
}
