import { useEffect, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import WorkspaceIcon from "../components/WorkspaceIcon";
import { AuthApiError, getCurrentUser, SignedInUser } from "../services/authApi";
import { getPaperSummary, listPapers, paperFileUrl, Paper, PaperSummary } from "../services/papersApi";

function greeting() {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

function displayDate(value: string) {
  return new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric" }).format(new Date(value));
}

function PaperStatus({ status }: { status: Paper["status"] }) {
  const icon = status === "ready" ? "check" : status === "processing" ? "clock" : "warning";
  return <span className={`home-status home-status-${status}`}><WorkspaceIcon name={icon} size={16} />{status === "ready" ? "Ready" : status === "processing" ? "Processing" : "Failed"}</span>;
}

export default function HomePage() {
  const [user, setUser] = useState<SignedInUser | null>(null);
  const [summary, setSummary] = useState<PaperSummary | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [sessionExpired, setSessionExpired] = useState(false);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const currentUser = await getCurrentUser();
        if (!active) return;
        setUser(currentUser);
        const [paperSummary, recentPapers] = await Promise.all([getPaperSummary(), listPapers(0)]);
        if (active) {
          setSummary(paperSummary);
          setPapers(recentPapers);
        }
      } catch (caught) {
        if (!active) return;
        if (caught instanceof AuthApiError && caught.status === 401) {
          setSessionExpired(true);
        } else {
          setError(caught instanceof Error ? caught.message : "Your dashboard could not be loaded.");
        }
      } finally {
        if (active) setLoading(false);
      }
    }
    void load();
    return () => { active = false; };
  }, [attempt]);

  if (sessionExpired) return <Navigate to="/login" replace />;
  if (loading) return <main className="session-check" role="status">Loading your workspace...</main>;
  if (!user) return <main className="session-check"><p>{error}</p><button className="primary-button" onClick={() => { setError(""); setLoading(true); setAttempt((value) => value + 1); }}>Try again</button></main>;

  const firstName = user.name?.trim().split(/\s+/)[0] || user.email.split("@")[0];
  const latestReady = papers.find((paper) => paper.status === "ready");
  const recent = papers.slice(0, 5);
  const inProgress = papers.filter((paper) => paper.status !== "ready").slice(0, 3);

  return (
    <>
      <main className="home-main">
        <div className="home-header">
          <div><p className="home-eyebrow">YOUR WORKSPACE</p><h1>{greeting()}, {firstName}</h1><p>Your research, grounded in evidence.</p></div>
          <Link className="home-upload-button" to="/papers?focus=upload"><WorkspaceIcon name="upload" /> Upload paper</Link>
        </div>
        {error && <div className="home-error" role="alert">{error} <button type="button" onClick={() => { setError(""); setLoading(true); setAttempt((value) => value + 1); }}>Retry</button></div>}

        <div className="home-columns">
          <div className="home-primary-column">
            <section className="home-stats" aria-label="Paper overview">
              <div className="home-stat"><span className="home-stat-icon lilac"><WorkspaceIcon name="file" size={28} /></span><strong>{summary?.total ?? "—"}</strong><span>Papers</span><Link to="/papers">View library <WorkspaceIcon name="chevron" size={15} /></Link></div>
              <div className="home-stat"><span className="home-stat-icon blue"><WorkspaceIcon name="check" size={28} /></span><strong>{summary?.ready ?? "—"}</strong><span>Ready to explore</span><Link to="/papers?status=ready">View ready papers <WorkspaceIcon name="chevron" size={15} /></Link></div>
              <div className="home-stat"><span className="home-stat-icon teal"><WorkspaceIcon name="clock" size={28} /></span><strong>{summary?.processing ?? "—"}</strong><span>Processing</span><Link to="/papers?status=processing">View uploads <WorkspaceIcon name="chevron" size={15} /></Link></div>
            </section>

            <section className="home-card home-continue" aria-labelledby="continue-title">
              <h2 id="continue-title">Continue with your research</h2>
              {latestReady ? <div className="home-featured-paper">
                <span className="home-paper-cover"><WorkspaceIcon name="file" size={52} /><small>PDF</small></span>
                <div><h3>{latestReady.filename}</h3><p>{latestReady.page_count} {latestReady.page_count === 1 ? "page" : "pages"} · {latestReady.chunk_count} evidence chunks</p><span>Added {displayDate(latestReady.created_at)}</span></div>
                <a className="home-outline-link" href={paperFileUrl(latestReady.id)} target="_blank" rel="noopener noreferrer">Open PDF <WorkspaceIcon name="arrow" size={16} /></a>
              </div> : <div className="home-empty"><p>Your first paper will appear here after it has been prepared.</p><Link to="/papers?focus=upload">Upload a paper <WorkspaceIcon name="arrow" size={16} /></Link></div>}
            </section>

            <section className="home-card home-recent" aria-labelledby="recent-title">
              <div className="home-card-header"><h2 id="recent-title">Recent papers</h2><Link to="/papers">View all papers <WorkspaceIcon name="chevron" size={16} /></Link></div>
              {recent.length ? <div className="home-table-wrap"><table><thead><tr><th>Paper</th><th>Status</th><th>Added</th><th><span className="sr-only">Open</span></th></tr></thead><tbody>
                {recent.map((paper) => <tr key={paper.id}><td><WorkspaceIcon name="file" size={18} /><span>{paper.filename}</span></td><td><PaperStatus status={paper.status} /></td><td>{displayDate(paper.created_at)}</td><td>{paper.status === "ready" ? <a aria-label={`View ${paper.filename} PDF`} href={paperFileUrl(paper.id)} target="_blank" rel="noopener noreferrer"><WorkspaceIcon name="arrow" size={18} /></a> : <Link aria-label={`View ${paper.filename} status`} to={`/papers?paper=${encodeURIComponent(paper.id)}`}><WorkspaceIcon name="arrow" size={18} /></Link>}</td></tr>)}
              </tbody></table></div> : <div className="home-empty"><p>No papers yet. Upload a PDF to start your library.</p></div>}
            </section>
          </div>

          <div className="home-secondary-column">
            <section className="home-card home-actions" aria-labelledby="actions-title"><h2 id="actions-title">Quick actions</h2>
              <Link to="/papers?focus=upload"><span className="home-action-icon"><WorkspaceIcon name="upload" size={28} /></span><span><strong>Upload PDF</strong><small>Add a paper to your library</small></span><WorkspaceIcon name="chevron" size={19} /></Link>
              <Link to="/papers"><span className="home-action-icon"><WorkspaceIcon name="library" size={28} /></span><span><strong>Browse library</strong><small>See your papers and text previews</small></span><WorkspaceIcon name="chevron" size={19} /></Link>
              {user.role === "admin" && <Link to="/admin"><span className="home-action-icon"><WorkspaceIcon name="user" size={28} /></span><span><strong>Admin access</strong><small>View user accounts</small></span><WorkspaceIcon name="chevron" size={19} /></Link>}
            </section>
            <section className="home-card home-processing" aria-labelledby="processing-title"><div className="home-card-header"><h2 id="processing-title">Ingestion status</h2><Link to="/papers?status=processing">View processing <WorkspaceIcon name="chevron" size={16} /></Link></div>
              {inProgress.length ? <ul>{inProgress.map((paper) => <li key={paper.id}><WorkspaceIcon name="file" size={19} /><span><strong>{paper.filename}</strong><PaperStatus status={paper.status} /></span></li>)}</ul> : <p>All your recent papers are ready. New uploads will show their status here.</p>}
            </section>
          </div>
        </div>
      </main>
    </>
  );
}
