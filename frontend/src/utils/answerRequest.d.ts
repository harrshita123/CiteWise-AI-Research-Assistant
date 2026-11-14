import type { AskResult } from "../services/researchApi";

export type AnswerRequest = {
  requestId: string;
  paperId: string;
  question: string;
  conversationId: string | null;
  contextVersion: number;
};

export declare function matchesAnswerRequest(answer: AskResult, request: AnswerRequest,
  activeRequestId: string | null, contextVersion: number): boolean;
