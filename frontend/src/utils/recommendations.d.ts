export function recommendationView(state: { loading: boolean; error: boolean; count: number }): "loading" | "error" | "questions" | "hidden";
export function researchQuestionMode(state: { demoMode: boolean; demoSample: boolean; canAnswer: boolean; retryRemaining: number }): "ask" | "search";
