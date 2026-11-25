export function recommendationView({ loading, error, count }) {
  if (loading) return "loading";
  if (error) return "error";
  return count > 0 ? "questions" : "hidden";
}

export function researchQuestionMode({ demoMode, demoSample, canAnswer, retryRemaining }) {
  return (demoMode ? demoSample : canAnswer && retryRemaining === 0) ? "ask" : "search";
}
