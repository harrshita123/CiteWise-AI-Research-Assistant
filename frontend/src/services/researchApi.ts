export type RetrievedPassage = {
  chunk_index: number;
  page_number: number;
  text: string;
  content_hash: string;
  highlight_text?: string;
};

export type AnswerSource = RetrievedPassage & { id: string; highlight_text: string };
export type ExternalSource = { id: string; title: string; url: string; published_date: string | null; excerpt?: string | null;
  publisher?: string | null; author?: string | null; verified_publication_date?: string | null };
export type EvidenceReview = { level: "sufficient" | "partial" | "none"; chunk_ids: string[]; missing_information: string };
export type ClaimReview = { section: "pdf" | "external"; status: string; supporting: { source_id: string; quote: string }[];
  text?: string; explanation?: string; opposing?: { source_id: string; quote: string }[] };

export type AskResult = {
  paper_id: string;
  question?: string | null;
  request_id?: string | null;
  conversation_id: string | null;
  message_id: string | null;
  status: "answered" | "not_mentioned" | "partially_supported" | "insufficient_evidence" | "contradictory_evidence";
  answer: string | null;
  sources: AnswerSource[];
  external_sources?: ExternalSource[];
  assessment?: EvidenceReview | null;
  claims?: ClaimReview[];
  verification_status?: "verified" | "not_verified" | "unavailable";
  answer_state?: "VERIFIED" | "PARTIALLY_SUPPORTED" | "INSUFFICIENT_EVIDENCE" | "VERIFICATION_UNAVAILABLE" | null;
  retry_after_seconds?: number;
};

export type SavedMessage = {
  id: string;
  question: string;
  status: AskResult["status"];
  answer: string | null;
  sources: AnswerSource[];
  external_sources?: ExternalSource[];
  assessment?: EvidenceReview | null;
  claims?: ClaimReview[];
  created_at: string;
  feedback?: AnswerFeedback | null;
  verification_status?: AskResult["verification_status"];
  answer_state?: AskResult["answer_state"];
  retry_after_seconds?: number;
};

export type AnswerFeedback = {
  rating: "helpful" | "incorrect";
  reason: "wrong_page" | "unsupported_answer" | "other" | null;
};

export type RecommendedQuestion = {
  id: string;
  text: string;
  recommendation_type: "paper" | "methodology" | "findings" | "limitations" | string;
  page_number: number | null;
  chunk_index: number | null;
};

export type Conversation = {
  id: string;
  paper_id: string;
  paper_filename: string | null;
  title: string;
  pinned: boolean;
  saved: boolean;
  created_at: string;
  updated_at: string;
  messages: SavedMessage[];
};

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ResearchApiError extends Error {
  constructor(message: string, public status: number, public retryAfterSeconds: number | null = null,
    public retrieved: { paperId: string; passages: RetrievedPassage[] } | null = null,
    public limitSource: string | null = null) {
    super(message);
  }
}

async function researchRequest<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/api/v1/research${path}`, {
      credentials: "include",
      ...options,
    });
  } catch {
    throw new Error("Research is unavailable. Check your connection and try again.");
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const body = payload?.detail;
    const hasProviderPassages = (response.status === 429 && ["upstream_capacity", "rate_limited"].includes(body?.code))
      || (response.status === 502 && body?.code === "provider_unavailable");
    const detail = typeof body === "string" ? body
      : (hasProviderPassages || response.status === 429) && typeof body?.message === "string"
        ? body.message : "Research could not finish.";
    const retrieved = hasProviderPassages && typeof body.paper_id === "string" && Array.isArray(body.passages)
      ? { paperId: body.paper_id, passages: body.passages as RetrievedPassage[] } : null;
    const retryHeader = Number(response.headers.get("Retry-After"));
    const retryAfter = response.status === 429 && Number.isFinite(retryHeader) && retryHeader > 0
      ? Math.ceil(retryHeader) : null;
    const limitSource = response.status === 429 && typeof body?.limit_source === "string" ? body.limit_source : null;
    throw new ResearchApiError(detail, response.status, retryAfter, retrieved, limitSource);
  }
  return payload as T;
}

function questionOptions(paperId: string, question: string): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paper_id: paperId, question }),
  };
}

export async function searchPaper(paperId: string, question: string): Promise<RetrievedPassage[]> {
  const result = await researchRequest<{ passages: RetrievedPassage[] }>("/search", questionOptions(paperId, question));
  return result.passages;
}

export function askPaper(paperId: string, question: string, conversationId?: string,
  requestId?: string, replyToMessageId?: string): Promise<AskResult> {
  return researchRequest<AskResult>("/ask", {
    ...questionOptions(paperId, question),
    body: JSON.stringify({ paper_id: paperId, question, conversation_id: conversationId,
      request_id: requestId, reply_to_message_id: replyToMessageId }),
  });
}

export async function starterQuestions(paperId: string): Promise<RecommendedQuestion[]> {
  const result = await researchRequest<{ recommendations: RecommendedQuestion[] }>(
    `/papers/${encodeURIComponent(paperId)}/recommendations`,
  );
  return result.recommendations;
}

export async function alternativeQuestions(paperId: string, question: string, conversationId?: string): Promise<RecommendedQuestion[]> {
  const result = await researchRequest<{ recommendations: RecommendedQuestion[] }>(
    `/papers/${encodeURIComponent(paperId)}/recommendations/alternatives`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: conversationId }),
    },
  );
  return result.recommendations;
}

export async function followUpQuestions(conversationId: string, messageId: string): Promise<RecommendedQuestion[]> {
  const result = await researchRequest<{ recommendations: RecommendedQuestion[] }>(
    `/conversations/${encodeURIComponent(conversationId)}/recommendations`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message_id: messageId }) },
  );
  return result.recommendations;
}

export function listConversations(): Promise<Conversation[]> {
  return researchRequest<Conversation[]>("/conversations");
}

export function getConversation(id: string): Promise<Conversation> {
  return researchRequest<Conversation>(`/conversations/${encodeURIComponent(id)}`);
}

export function updateConversation(id: string, update: { title?: string; pinned?: boolean; saved?: boolean }): Promise<Conversation> {
  return researchRequest<Conversation>(`/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(update),
  });
}

export function deleteConversation(id: string): Promise<void> {
  return researchRequest<void>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function saveAnswerFeedback(conversationId: string, messageId: string, feedback: AnswerFeedback): Promise<AnswerFeedback> {
  return researchRequest<AnswerFeedback>(`/conversations/${encodeURIComponent(conversationId)}/messages/${encodeURIComponent(messageId)}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(feedback),
  });
}

export function retryVerification(conversationId: string, messageId: string): Promise<SavedMessage> {
  return researchRequest<SavedMessage>(`/conversations/${encodeURIComponent(conversationId)}/messages/${encodeURIComponent(messageId)}/verify`, { method: "POST" });
}

export async function answerGenerationAvailable(): Promise<{ available: boolean; retryAfterSeconds: number }> {
  const result = await researchRequest<{ answer_generation: boolean; retry_after_seconds: number }>("/capabilities");
  return { available: result.answer_generation, retryAfterSeconds: result.retry_after_seconds };
}
