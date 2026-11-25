type AcademicMetadata = {
  title?: string | null;
  authors?: string[] | null;
  abstract?: string | null;
  version?: string | null;
  year?: number | null;
  venue?: string | null;
  publisher?: string | null;
  doi?: string | null;
  arxiv_id?: string | null;
  pmid?: string | null;
  pmcid?: string | null;
};

export type Paper = {
  id: string;
  filename: string;
  status: "processing" | "ready" | "failed" | "not_research" | "needs_review";
  page_count: number;
  chunk_count: number;
  embedding_model: string;
  embedding_dimensions: number;
  created_at: string;
  error: string | null;
  publication_identity?: (AcademicMetadata & {
    status: string;
    demo_sample?: boolean;
    reason?: string;
    checked_at?: string;
    next_retry_at?: string;
    retry_after?: string;
    evidence?: { authors_matched?: string[] };
    source?: string;
    source_url?: string | null;
  }) | null;
};

export type PaperDetail = {
  extracted_metadata?: AcademicMetadata | null;
  paper: Paper;
  chunks: { chunk_index: number; page_number: number; text: string; content_hash: string }[];
};

export type PaperSummary = {
  total: number;
  ready: number;
  processing: number;
  failed: number;
};

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/api/v1/papers${path}`, {
      ...options,
      credentials: "include",
    });
  } catch {
    throw new Error("The paper service is unavailable. Check your connection and try again.");
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = typeof payload?.detail === "string" ? payload.detail : "The paper could not be loaded.";
    throw new Error(response.status === 401 ? "Your session has expired. Please sign in again." : message);
  }
  return payload as T;
}

export function listPapers(offset = 0, limit = 50, status?: Paper["status"]) {
  const filter = status ? `&status=${status}` : "";
  return request<Paper[]>(`?limit=${limit}&offset=${offset}${filter}`);
}

export function getPaperSummary() {
  return request<PaperSummary>("/summary");
}

export function getPaper(id: string, chunkOffset = 0) {
  return request<PaperDetail>(`/${encodeURIComponent(id)}?chunk_offset=${chunkOffset}`);
}

export function getLibraryPaper(id: string, signal?: AbortSignal) {
  return request<PaperDetail>(`/${encodeURIComponent(id)}?include_chunks=false`, { signal });
}

export function paperFileUrl(id: string) {
  return `${apiBaseUrl}/api/v1/papers/${encodeURIComponent(id)}/file`;
}

export function uploadPaper(file: File) {
  const form = new FormData();
  form.append("file", file);
  return request<PaperDetail>("", { method: "POST", body: form });
}

export function deletePaper(id: string) {
  return request<null>(`/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function retryPublicationVerification(id: string) {
  return request<{status: string}>(`/${encodeURIComponent(id)}/verification/retry`, { method: "POST" });
}
