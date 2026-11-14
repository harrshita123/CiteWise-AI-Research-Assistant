export type EvaluationCase = {
  paper_id: string;
  question: string;
  expected_page: number;
};

export type CaseResult = EvaluationCase & {
  returned_pages: number[];
  hit_at_5: boolean;
};

export type EvaluationRun = {
  id: string;
  created_at: string;
  case_count: number;
  hit_at_5: number;
  results: CaseResult[];
};

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
    throw new Error(typeof payload?.detail === "string" ? payload.detail : "Evaluation could not finish.");
  }
  return payload as T;
}

export function listEvaluationRuns() {
  return request<EvaluationRun[]>("/runs");
}

export function createEvaluationRun(cases: EvaluationCase[]) {
  return request<EvaluationRun>("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cases }),
  });
}
