export type RetrievedPassage = {
  chunk_index: number;
  page_number: number;
  text: string;
  content_hash: string;
  highlight_text?: string;
};

export type AnswerSource = RetrievedPassage & { id: string; highlight_text: string };
export type ExternalSource = { id: string; title: string; url: string; published_date: string | null; excerpt?: string | null };
export type EvidenceReview = { level: "sufficient" | "partial" | "none"; chunk_ids: string[]; missing_information: string };

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
};

export type SavedMessage = {
  id: string;
  question: string;
  status: AskResult["status"];
  answer: string | null;
  sources: AnswerSource[];
  external_sources?: ExternalSource[];
  assessment?: EvidenceReview | null;
  created_at: string;
  feedback?: AnswerFeedback | null;
};

export type AnswerFeedback = {
  rating: "helpful" | "incorrect";
  reason: "wrong_page" | "unsupported_answer" | "other" | null;
};

export type Conversation = {
  id: string;
  paper_id: string;
  created_at: string;
  updated_at: string;
  messages: SavedMessage[];
};

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ResearchApiError extends Error {
  constructor(message: string, public status: number, public retryAfterSeconds: number | null = null) {
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
    const detail = typeof payload?.detail === "string" ? payload.detail : "Research could not finish.";
    const retryHeader = Number(response.headers.get("Retry-After"));
    const retryAfter = response.status === 429 && Number.isFinite(retryHeader) && retryHeader > 0
      ? Math.ceil(retryHeader) : null;
    throw new ResearchApiError(retryAfter ? `${detail} Try again in ${Math.ceil(retryAfter / 60)} minute${retryAfter > 60 ? "s" : ""}.` : detail,
      response.status, retryAfter);
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

export function listConversations(): Promise<Conversation[]> {
  return researchRequest<Conversation[]>("/conversations");
}

export function getConversation(id: string): Promise<Conversation> {
  return researchRequest<Conversation>(`/conversations/${encodeURIComponent(id)}`);
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

export async function answerGenerationAvailable(): Promise<{ available: boolean; keyRevision: string | null }> {
  const result = await researchRequest<{ answer_generation: boolean; key_revision: string | null }>("/capabilities");
  return { available: result.answer_generation, keyRevision: result.key_revision };
}
