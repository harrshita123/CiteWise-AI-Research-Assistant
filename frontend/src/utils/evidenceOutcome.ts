import type { AskResult } from "../services/researchApi";

type EvidenceStatus = AskResult["status"];

const labels: Record<EvidenceStatus, string> = {
  answered: "Answer",
  not_mentioned: "Not mentioned",
  partially_supported: "Partially supported",
  insufficient_evidence: "Insufficient evidence",
  contradictory_evidence: "Contradictory evidence",
};

const fallbacks: Record<EvidenceStatus, string> = {
  answered: "No answer was saved.",
  not_mentioned: "The retrieved passages do not mention this topic.",
  partially_supported: "The retrieved passages support only part of the answer.",
  insufficient_evidence: "The retrieved passages do not establish a reliable conclusion.",
  contradictory_evidence: "The retrieved passages contain differing findings or interpretations.",
};

export function evidenceLabel(status: EvidenceStatus): string {
  return labels[status] ?? labels.insufficient_evidence;
}

export function evidenceText(status: EvidenceStatus, answer: string | null): string {
  return answer?.trim() || fallbacks[status] || fallbacks.insufficient_evidence;
}
