import { publicationStatus } from "../utils/publicationStatus";
import { FormEvent, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useOutletContext, useSearchParams } from "react-router-dom";
import type { ConversationState } from "../components/WorkspaceLayout";
import WorkspaceIcon from "../components/WorkspaceIcon";
import PaperDiscovery from "../components/PaperDiscovery";
import { getCurrentUser, SignedInUser } from "../services/authApi";
import { deletePaper, getLibraryPaper, listPapers, paperFileUrl, Paper, PaperDetail, retryPublicationVerification, uploadPaper } from "../services/papersApi";

export default function PapersPage() {
  const { demoMode } = useOutletContext<ConversationState>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const paperId = searchParams.get("paper");
  const statusParam = searchParams.get("status");
  const statusFilter = statusParam === "ready" || statusParam === "processing" || statusParam === "failed" ? statusParam : undefined;
  const focusUpload = searchParams.get("focus") === "upload";
  const [user, setUser] = useState<SignedInUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<PaperDetail | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [loadingPaper, setLoadingPaper] = useState(false);
  const [retryingVerification, setRetryingVerification] = useState(false);
  const verificationRequest = useRef(false);
  const [retryClock, setRetryClock] = useState(Date.now);
  const [error, setError] = useState("");
  const [offset, setOffset] = useState(0);
  const [hasNextPage, setHasNextPage] = useState(false);
  const [loadingList, setLoadingList] = useState(false);
  const selectionRequest = useRef(0);
  const uploadInput = useRef<HTMLInputElement>(null);
  const detailsSection = useRef<HTMLElement>(null);
  const revealUploadedPaper = useRef<string | null>(null);

  const [pollingStopped, setPollingStopped] = useState<string | null>(null);
  const selectedId = selected?.paper.id;
  const identity = selected?.paper.publication_identity;
  const verifying = identity?.status === "pending" || identity?.status === "checking";
  const retryUntil = Date.parse(identity?.retry_after ?? "");
  const coolingDown = retryUntil > retryClock;
  const pollingExpired = !!selectedId && pollingStopped === selectedId;
  const registry = identity?.status === "verified_published" || identity?.status === "verified_preprint"
    ? identity : undefined;
  const extracted = selected?.extracted_metadata;
  const academicTitle = extracted?.title || registry?.title;
  const authors = (registry?.authors?.length ?? 0) > (extracted?.authors?.length ?? 0)
    ? registry?.authors : extracted?.authors || registry?.evidence?.authors_matched;
  const venue = extracted?.venue || registry?.venue;
  const year = extracted?.year || registry?.year;
  const publisher = extracted?.publisher || registry?.publisher;
  const doi = extracted?.doi || registry?.doi;
  const arxiv = extracted?.arxiv_id || registry?.arxiv_id;
  const pmid = extracted?.pmid || registry?.pmid;
  const pmcid = extracted?.pmcid || registry?.pmcid;

  useEffect(() => {
    if (!Number.isFinite(retryUntil)) return;
    setRetryClock(Date.now());
    const timer = window.setTimeout(() => setRetryClock(Date.now()), Math.max(0, retryUntil - Date.now()) + 10);
    return () => window.clearTimeout(timer);
  }, [retryUntil]);

  useEffect(() => {
    if (!selectedId || !verifying || pollingExpired) return;
    const controller = new AbortController();
    let timer: number;
    const deadline = window.setTimeout(() => {
      controller.abort();
      window.clearTimeout(timer);
      setPollingStopped(selectedId);
    }, 5 * 60 * 1000);

    async function poll() {
      if (!selectedId) return;
      try {
        const detail = await getLibraryPaper(selectedId, controller.signal);
        if (controller.signal.aborted) return;
        setSelected(current => current?.paper.id === selectedId ? detail : current);
        setPapers(current => current.map(paper => paper.id === selectedId ? detail.paper : paper));
        const status = detail.paper.publication_identity?.status;
        if (status !== "pending" && status !== "checking") {
          window.clearTimeout(deadline);
          return;
        }
      } catch {
        if (controller.signal.aborted) return;
      }
      timer = window.setTimeout(poll, 3000);
    }

    timer = window.setTimeout(poll, 3000);
    return () => {
      controller.abort();
      window.clearTimeout(deadline);
      window.clearTimeout(timer);
    };
  }, [selectedId, verifying, pollingExpired]);

  useEffect(() => {
    if (selectedId && revealUploadedPaper.current === selectedId) {
      revealUploadedPaper.current = null;
      detailsSection.current?.scrollIntoView({ block: "start" });
    }
  }, [selectedId]);

  async function retryVerification() {
    if (!selected || verificationRequest.current || retryUntil > Date.now()) return;
    const id = selected.paper.id;
    const selection = selectionRequest.current;
    verificationRequest.current = true;
    setRetryingVerification(true);
    setError("");
    try {
      await retryPublicationVerification(id);
      setPollingStopped(current => current === id ? null : current);
      setSelected(current => current?.paper.id === id ? { ...current, paper: { ...current.paper,
        publication_identity: { status: "pending", reason: "Publication identity check pending." } } } : current);
    } catch (error) {
      if (selection === selectionRequest.current) {
        setError(error instanceof Error ? error.message : "Verification could not be retried.");
      }
    } finally {
      verificationRequest.current = false;
      setRetryingVerification(false);
    }
  }

  useEffect(() => {
    let active = true;
    setSelected(current => current?.paper.id === paperId ? current : null);
    const requestId = selectionRequest.current;
    async function load() {
      try {
        const currentUser = await getCurrentUser();
        if (active) setUser(currentUser);
        const results = await listPapers(0, 51, statusFilter);
        if (active) {
          setPapers(results.slice(0, 50));
          setHasNextPage(results.length > 50);
          setOffset(0);
          if (paperId) {
            const detail = await getLibraryPaper(paperId);
            if (active && requestId === selectionRequest.current && (!statusFilter || detail.paper.status === statusFilter)) {
              setSelected(detail);
              setPapers(current => current.map(item => item.id === paperId ? detail.paper : item));
            }
          }
        }
      } catch (error) {
        if (active) setError(error instanceof Error ? error.message : "Your library could not be loaded.");
      } finally {
        if (active) setLoading(false);
      }
    }
    void load();
    return () => { active = false; };
  }, [paperId, statusFilter]);

  useEffect(() => {
    if (!loading && user && focusUpload) {
      uploadInput.current?.scrollIntoView({ block: "center" });
      uploadInput.current?.focus();
    }
  }, [focusUpload, loading, user]);

  async function refresh(pageOffset = offset, resetUploadForm = false) {
    setLoadingList(true);
    try {
      const results = await listPapers(pageOffset, 51, statusFilter);
      setPapers(results.slice(0, 50));
      setHasNextPage(results.length > 50);
      setOffset(pageOffset);
      if (resetUploadForm) {
        setError("");
        setFile(null);
        if (uploadInput.current) uploadInput.current.value = "";
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : "Your library could not be refreshed.");
    } finally {
      setLoadingList(false);
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || busy) return;
    setError("");
    if (!file.name.toLowerCase().endsWith(".pdf") || file.size > 25 * 1024 * 1024) {
      setError("Choose a PDF that is 25 MB or smaller.");
      return;
    }
    setBusy(true);
    selectionRequest.current += 1;
    setLoadingPaper(false);
    try {
      const result = await uploadPaper(file);
      revealUploadedPaper.current = result.paper.id;
      setSelected(result);
      setFile(null);
      if (uploadInput.current) uploadInput.current.value = "";
      navigate(`/papers?paper=${encodeURIComponent(result.paper.id)}`);
    } catch (error) {
      setError(error instanceof Error ? error.message : "Your paper could not be processed.");
      await refresh(0, false);
    } finally {
      setBusy(false);
    }
  }

  async function selectPaper(paper: Paper) {
    const requestId = ++selectionRequest.current;
    setLoadingPaper(true);
    setError("");
    try {
      const detail = await getLibraryPaper(paper.id);
      if (requestId === selectionRequest.current) {
        setPollingStopped(null);
        setSelected(detail);
        setPapers(current => current.map(item => item.id === paper.id ? detail.paper : item));
        if (paper.id !== paperId) {
          const params = new URLSearchParams(searchParams);
          params.set("paper", paper.id);
          params.delete("focus");
          navigate(`/papers?${params}`, { replace: true });
        }
      }
    } catch (error) {
      if (requestId === selectionRequest.current) {
        setError(error instanceof Error ? error.message : "The paper details could not be loaded.");
      }
    } finally {
      if (requestId === selectionRequest.current) setLoadingPaper(false);
    }
  }

  async function handleDelete(paper: Paper) {
    if (busy || deletingId || !window.confirm(`Delete "${paper.filename}" from your library? This also removes its saved PDF and extracted text.`)) return;
    setError("");
    setDeletingId(paper.id);
    try {
      await deletePaper(paper.id);
      if (selected?.paper.id === paper.id) {
        selectionRequest.current += 1;
        setSelected(null);
      }
      if (paperId === paper.id) navigate("/papers", { replace: true });
      await refresh(papers.length === 1 && offset > 0 ? offset - 50 : offset);
    } catch (error) {
      setError(error instanceof Error ? error.message : "The paper could not be deleted.");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    loading ? <main className="session-check" role="status">Loading your library...</main> : !user ? (
      <main className="session-check"><p>{error || "Sign in to view your library."}</p><Link className="primary-button" to="/login">Sign in</Link></main>
    ) : <>
      <main className="admin-main paper-main workspace-library-main">
        <p className="label">Your research library</p>
        <h1>{statusFilter === "processing" ? "Processing uploads" : statusFilter === "ready" ? "Ready papers" : statusFilter === "failed" ? "Failed uploads" : "Your papers"}</h1>
        <p className="paper-intro">{statusFilter === "processing" ? "Papers being prepared appear here. Finished papers are available in your library." : statusFilter === "ready" ? "These papers are ready to view and inspect." : statusFilter === "failed" ? "Papers that could not be prepared appear here." : "Upload a PDF, then open the original paper from your library."}</p>
        {error && <p className="oauth-error" role="alert">{error}</p>}
            {!statusFilter && <form className="paper-upload" onSubmit={handleUpload} aria-busy={busy}>
              <div className="paper-upload-heading"><span className="paper-upload-icon"><WorkspaceIcon name="upload" size={24} /></span><div><h2>Upload a PDF</h2><p id="upload-help">Text-based PDF · up to 25 MB · up to 300 pages</p></div></div>
              <input className="paper-file-input" id="paper-file" ref={uploadInput} type="file" accept=".pdf,application/pdf" required disabled={busy}
                aria-describedby="upload-help" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setError(""); }} />
              <label className="paper-file-picker" htmlFor="paper-file">
                <span className="paper-picker-copy"><strong>{file ? file.name : "Choose a paper from your device"}</strong><small>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB PDF selected` : "Scanned PDFs are not supported yet"}</small></span>
                <span className="paper-picker-action">Browse files</span>
              </label>
              <button className="primary-button" type="submit" disabled={!file || busy}>
                {busy ? "Preparing your paper..." : "Upload and prepare"}
              </button>
              {busy && <p role="status">Preparing your PDF. The first upload may take longer. Keep this page open.</p>}
            </form>}

            <div className="paper-library-heading">
              <h2>{statusFilter ? "Uploads" : "Your library"}</h2>
              <div className="paper-heading-actions">
                {statusFilter && <Link className="secondary-button small" to="/papers">All papers</Link>}
                <button className="secondary-button small" type="button" disabled={busy || loadingList} onClick={() => void refresh(offset, true)}>
                  {loadingList ? "Refreshing..." : "Refresh list"}
                </button>
              </div>
            </div>
            {!papers.length ? <div className="paper-empty-state">
              <span className="paper-empty-icon"><WorkspaceIcon name={statusFilter === "processing" ? "check" : "file"} size={30} /></span>
              <h3>{statusFilter === "processing" ? "All caught up" : statusFilter === "ready" ? "No ready papers yet" : statusFilter === "failed" ? "No failed uploads" : "Your library is empty"}</h3>
              <p>{statusFilter === "processing" ? "Nothing is processing right now. Your finished papers are ready in the library." : statusFilter === "ready" ? "Upload a PDF to add your first ready paper." : statusFilter === "failed" ? "Your uploads have no reported failures." : "Choose a PDF above to add your first paper."}</p>
              {statusFilter && <Link className="primary-button" to="/papers">View all papers</Link>}
            </div> : (
              <ul className="paper-list">
                {papers.map((paper) => (
                  <li key={paper.id}>
                    <div className="paper-list-row">
                      <button type="button" disabled={busy || deletingId !== null} aria-pressed={selected?.paper.id === paper.id}
                        onClick={() => void selectPaper(paper)}>
                        <span><strong>{paper.filename}</strong><small>{paper.page_count} pages</small></span>
                        <span className={`paper-status ${paper.status}`}>{paper.status === "not_research" ? "Not a research paper" : paper.status === "ready" ? "Ready for Research" : paper.status === "needs_review" ? "Needs Review" : paper.status}</span>
                      </button>
                      {(paper.status === "ready" || paper.status === "not_research") && <a className="secondary-button paper-list-view" href={paperFileUrl(paper.id)} target="_blank" rel="noopener noreferrer" aria-label={`View ${paper.filename} PDF`}>View PDF</a>}
                      {paper.status !== "processing" && <button className="secondary-button paper-list-delete" type="button" disabled={busy || deletingId !== null} onClick={() => void handleDelete(paper)} aria-label={`Delete ${paper.filename}`}>
                        {deletingId === paper.id ? "Deleting..." : "Delete"}
                      </button>}
                    </div>
                  </li>
                ))}
              </ul>
            )}
            {offset > 0 || hasNextPage ? (
              <nav className="paper-pagination" aria-label="Paper list pages">
                <button className="secondary-button" disabled={offset === 0 || busy || loadingList} onClick={() => void refresh(offset - 50)}>Previous</button>
                <span>Showing papers {offset + 1}–{offset + papers.length}</span>
                <button className="secondary-button" disabled={!hasNextPage || busy || loadingList} onClick={() => void refresh(offset + 50)}>Next</button>
              </nav>
            ) : papers.length > 0 && <p className="paper-list-count">Showing all {papers.length} {papers.length === 1 ? "paper" : "papers"}</p>}

            {loadingPaper && <p role="status">Loading paper details...</p>}
            {selected && (!statusFilter || selected.paper.status === statusFilter) && (
              <section ref={detailsSection} className="paper-preview" aria-labelledby="preview-title" aria-busy={loadingPaper}>
                <p className="label">Paper details</p>
              <h2 id="preview-title">{academicTitle || selected.paper.filename}</h2>
                {authors?.length ? <p>{authors.join(", ")}</p> : null}
                {(venue || year || publisher) && <p>{[venue, year, publisher].filter(Boolean).join(" · ")}</p>}
                {selected.paper.page_count > 0 && <p>{selected.paper.page_count} {selected.paper.page_count === 1 ? "page" : "pages"}</p>}
                {doi && <p>DOI: {doi}</p>}
                {arxiv && <p>arXiv: {arxiv}</p>}
                {pmid && <p>PMID: {pmid}</p>}
                {pmcid && <p>PMCID: {pmcid}</p>}
                {extracted?.version === "accepted_manuscript" && <p>Author-accepted manuscript</p>}
                {!!arxiv && !registry && <p>Preprint identifier found in the PDF. Publication identity is unverified.</p>}
                {extracted?.abstract && <p><strong>Abstract</strong><br />{extracted.abstract}</p>}
                <p role="status">{verifying
                  ? pollingExpired ? "Automatic updates paused. Refresh the verification status to check again."
                    : identity?.next_retry_at ? "Verification delayed. A retry is scheduled."
                    : "Verifying publication…"
                  : selected.paper.status === "needs_review" ? "Needs Review"
                  : selected.paper.status === "not_research" ? "Not a research paper"
                  : identity?.status === "verification_unavailable" ? "Verification temporarily unavailable"
                  : publicationStatus(identity?.status)}</p>
                {verifying && pollingExpired && <button type="button" className="secondary-button small" disabled={loadingPaper} onClick={() => void selectPaper(selected.paper)}>Refresh verification status</button>}
                {!verifying && identity?.reason && <p>{identity.reason}</p>}
                {!demoMode && selected.paper.status === "ready" && (!identity || ["verification_unavailable", "not_verified", "conflicting", "registry_record"].includes(identity.status)) && <button type="button" className="secondary-button small" disabled={retryingVerification || coolingDown} onClick={() => void retryVerification()}>{retryingVerification ? "Starting check…" : identity ? "Retry verification" : "Check publication identity"}</button>}
                {coolingDown && <p>Retry available at {new Date(retryUntil).toLocaleTimeString()}.</p>}
                {(registry?.source_url || registry?.source) && <p>
                  {registry.source_url && <a href={registry.source_url} target="_blank" rel="noopener noreferrer">Open official academic record</a>}
                  {registry.source_url && registry.source && " · "}{registry.source}
                </p>}
                {identity?.checked_at && <p>Checked: {new Date(identity.checked_at).toLocaleString()}</p>}
                {selected.paper.status === "needs_review" && <p>{selected.paper.error}</p>}
                {(selected.paper.status === "ready" || selected.paper.status === "not_research" || selected.paper.status === "needs_review") ? (
                    <a className="secondary-button paper-view-link" href={paperFileUrl(selected.paper.id)} target="_blank" rel="noopener noreferrer">View original PDF</a>
                ) : <p>{selected.paper.error || "This paper is still processing. Refresh the list to check its status."}</p>}
              </section>
            )}
        {!statusFilter && (demoMode
          ? <section className="paper-preview"><h2>Discover academic papers</h2><p>Live discovery is unavailable in Demo Mode. Open the bundled sample from your Library.</p></section>
          : <PaperDiscovery />)}
      </main>
    </>
  );
}
