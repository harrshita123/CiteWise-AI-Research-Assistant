import { useEffect, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import WorkspaceIcon from "../components/WorkspaceIcon";
import { AuthApiError, getCurrentUser, SignedInUser } from "../services/authApi";

type Area = "compare";

const pages = {
  compare: {
    label: "Compare",
    title: "Compare papers with evidence",
    description: "This area will show where two papers agree and differ, with page references for each point. Comparison is still being built.",
    next: "Add the papers you want to compare to your library first.",
    icon: "compare" as const,
  },
};

export default function ComingSoonPage({ area }: { area: Area }) {
  const [user, setUser] = useState<SignedInUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [expired, setExpired] = useState(false);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    getCurrentUser().then(
      (currentUser) => { if (active) { setUser(currentUser); setLoading(false); } },
      (caught: unknown) => {
        if (!active) return;
        if (caught instanceof AuthApiError && caught.status === 401) setExpired(true);
        else setError(caught instanceof Error ? caught.message : "The workspace could not be loaded.");
        setLoading(false);
      },
    );
    return () => { active = false; };
  }, [attempt]);

  if (expired) return <Navigate to="/login" replace />;
  if (loading) return <main className="session-check" role="status">Loading your workspace...</main>;
  if (!user) return <main className="session-check"><p>{error}</p><button className="primary-button" onClick={() => { setLoading(true); setAttempt((value) => value + 1); }}>Try again</button></main>;

  const page = pages[area];
  return <>
    <main className="coming-page">
      <p className="label">{page.label}</p>
      <div className="coming-card">
        <span className="coming-icon"><WorkspaceIcon name={page.icon} size={38} /></span>
        <span className="coming-badge">Coming soon</span>
        <h1>{page.title}</h1>
        <p>{page.description}</p>
        <p>{page.next}</p>
        <div className="coming-actions">
          <Link className="primary-button" to="/papers">Open library</Link>
          <Link className="secondary-button" to="/home">Back to dashboard</Link>
        </div>
      </div>
    </main>
  </>;
}
