import { FormEvent, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import WorkspaceIcon from "../components/WorkspaceIcon";
import { getCurrentUser, SignedInUser } from "../services/authApi";
import { deletePaper, getPaper, listPapers, paperFileUrl, Paper, PaperDetail, uploadPaper } from "../services/papersApi";

export default function PapersPage() {
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
  const [chunkOffset, setChunkOffset] = useState(0);
  const [error, setError] = useState("");
  const [offset, setOffset] = useState(0);
  const [hasNextPage, setHasNextPage] = useState(false);
  const [loadingList, setLoadingList] = useState(false);
  const selectionRequest = useRef(0);
  const uploadInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    setSelected(null);
    setChunkOffset(0);
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
            const detail = await getPaper(paperId);
            if (active && (!statusFilter || detail.paper.status === statusFilter)) {
              setSelected(detail);
              setChunkOffset(0);
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
      setSelected(result);
      setChunkOffset(0);
      setFile(null);
      if (uploadInput.current) uploadInput.current.value = "";
      if (statusFilter) {
        navigate(`/papers?paper=${encodeURIComponent(result.paper.id)}`);
      } else {
        await refresh(0);
      }
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
      const detail = await getPaper(paper.id);
      if (requestId === selectionRequest.current) {
        setSelected(detail);
        setChunkOffset(0);
      }
    } catch (error) {
      if (requestId === selectionRequest.current) {
        setError(error instanceof Error ? error.message : "The preview could not be loaded.");
      }
    } finally {
      if (requestId === selectionRequest.current) setLoadingPaper(false);
    }
  }

  async function showChunkPage(nextOffset: number) {
    if (!selected || loadingPaper) return;
    const requestId = ++selectionRequest.current;
    setLoadingPaper(true);
    setError("");
    try {
      const detail = await getPaper(selected.paper.id, nextOffset);
      if (requestId === selectionRequest.current) {
        setSelected(detail);
        setChunkOffset(nextOffset);
      }
    } catch (error) {
      if (requestId === selectionRequest.current) {
        setError(error instanceof Error ? error.message : "The chunks could not be loaded.");
      }
    } finally {
      if (requestId === selectionRequest.current) setLoadingPaper(false);
    }
  }

  async function handleDelete(paper: Paper) {
    if (busy || deletingId || !window.confirm(`Delete "${paper.filename}" from your library? This also removes its saved PDF and text chunks.`)) return;
    setError("");
    setDeletingId(paper.id);
    try {
      await deletePaper(paper.id);
      if (selected?.paper.id === paper.id) {
        selectionRequest.current += 1;
        setSelected(null);
        setChunkOffset(0);
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
        <p className="paper-intro">{statusFilter === "processing" ? "Papers being prepared appear here. Finished papers are available in your library." : statusFilter === "ready" ? "These papers are ready to view and inspect." : statusFilter === "failed" ? "Papers that could not be prepared appear here." : "Upload a PDF, then view the original paper and its extracted text."}</p>
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
                        <span><strong>{paper.filename}</strong><small>{paper.page_count} pages · {paper.chunk_count} chunks</small></span>
                        <span className={`paper-status ${paper.status}`}>{paper.status === "ready" ? "Ready" : paper.status}</span>
                      </button>
                      {paper.status === "ready" && <a className="secondary-button paper-list-view" href={paperFileUrl(paper.id)} target="_blank" rel="noopener noreferrer" aria-label={`View ${paper.filename} PDF`}>View PDF</a>}
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

            {loadingPaper && <p role="status">Loading paper preview...</p>}
            {selected && (!statusFilter || selected.paper.status === statusFilter) && (
              <section className="paper-preview" aria-labelledby="preview-title" aria-busy={loadingPaper}>
                <p className="label">Text preview</p>
                <h2 id="preview-title">{selected.paper.filename}</h2>
                {selected.paper.status === "ready" ? (
                  <>
                    <a className="secondary-button paper-view-link" href={paperFileUrl(selected.paper.id)} target="_blank" rel="noopener noreferrer">View original PDF</a>
                    <p>This PDF has {selected.paper.chunk_count} {selected.paper.chunk_count === 1 ? "text chunk" : "text chunks"}. Each chunk shows the page it came from.</p>
                    {selected.paper.chunk_count > 10 && <p>Showing chunks {chunkOffset + 1}–{chunkOffset + selected.chunks.length} of {selected.paper.chunk_count}.</p>}
                    {selected.chunks.map((chunk) => (
                      <article className="paper-chunk" key={chunk.chunk_index}>
                        <h3>Chunk {chunk.chunk_index + 1} <span>Page {chunk.page_number}</span></h3>
                        <p>{chunk.text}</p>
                      </article>
                    ))}
                    {selected.paper.chunk_count > 10 && <nav className="paper-pagination" aria-label="Chunk preview pages">
                      <button className="secondary-button" type="button" disabled={loadingPaper || chunkOffset === 0} onClick={() => void showChunkPage(Math.max(0, chunkOffset - 10))}>Previous chunks</button>
                      <span>Chunks {chunkOffset + 1}–{chunkOffset + selected.chunks.length}</span>
                      <button className="secondary-button" type="button" disabled={loadingPaper || chunkOffset + selected.chunks.length >= selected.paper.chunk_count} onClick={() => void showChunkPage(chunkOffset + 10)}>Next chunks</button>
                    </nav>}
                  </>
                ) : <p>{selected.paper.error || "This paper is still processing. Refresh the list to check its status."}</p>}
              </section>
            )}
      </main>
    </>
  );
}
