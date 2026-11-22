export function recommendationView({ loading, error, count }) {
  if (loading) return "loading";
  if (error) return "error";
  return count > 0 ? "questions" : "hidden";
}
