import { publicationStatus } from "../utils/publicationStatus";
import { useState, type FormEvent } from "react";

type Paper = { id: string; title: string; authors: string[]; year: number | null; venue: string | null;
  publisher: string | null; doi: string | null; arxiv_id: string | null; pmid?: string; pmcid?: string; status: string;
  landing_url: string; pdf_url: string | null; sources: string[]; status_note: string; verified_at: string };
type SearchResult = { results: Paper[]; provider_errors: Record<string, string>; offset: number; limit: number; has_more: boolean };
const base = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export default function PaperDiscovery() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function search(offset = 0) {
    if (busy || !query.trim()) return;
    setBusy(true); setError(""); setResult(null);
    try {
      const params = new URLSearchParams({ query: query.trim(), offset: String(offset), limit: "10" });
      const response = await fetch(`${base}/api/v1/discovery?${params}`, { credentials: "include" });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Search is unavailable.");
      setResult(body as SearchResult);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Search is unavailable."); }
    finally { setBusy(false); }
  }
  return <section className="paper-upload discovery" aria-labelledby="discovery-title">
    <header>
      <h2 id="discovery-title">Discover research papers</h2>
      <p>Search by title, topic, or identifier. For an author search, enter <strong>author:</strong> followed by a name.</p>
    </header>
    <form className="discovery-search" onSubmit={(event: FormEvent<HTMLFormElement>) => { event.preventDefault(); void search(); }}>
      <label>Search academic sources
        <input type="search" value={query} maxLength={200} placeholder="Title, topic, DOI, PMID, PMCID, or arXiv ID" onChange={(event) => setQuery(event.target.value)} required />
      </label>
      <button className="primary-button" type="submit" disabled={busy}>{busy ? "Searching…" : "Search papers"}</button>
    </form>
    {busy && <p role="status">Checking academic sources…</p>}
    {error && <p role="alert">Search is unavailable. Please try again.</p>}
    {result && <>
      {Object.keys(result.provider_errors).length > 0 && <p role="status">Some academic sources are unavailable. Results may be incomplete.</p>}
      {result.results.length === 0 ? <p className="discovery-empty" role="status">No matching papers returned. Try another title or identifier.</p> : <ul className="discovery-results">
        {result.results.map((paper) => {
          const identifiers = [paper.doi && `DOI: ${paper.doi}`, paper.arxiv_id && `arXiv: ${paper.arxiv_id}`, paper.pmid && `PMID: ${paper.pmid}`, paper.pmcid].filter(Boolean);
          const publication = [paper.venue, paper.publisher, paper.year].filter(Boolean).join(" · ");
          const status = publicationStatus(paper.status);
          return <li className="discovery-result" key={paper.id}>
            <div className="discovery-meta">
              {status && <span className="discovery-status">{status}</span>}
              {paper.sources.length > 0 && <span>{paper.sources.join(", ")}</span>}
            </div>
            <h3>{paper.title}</h3>
            {paper.authors.length > 0 && <p>{paper.authors.join(", ")}</p>}
            {publication && <p>{publication}</p>}
            {identifiers.length > 0 && <p>{identifiers.join(" · ")}</p>}
            {paper.status_note && <p>{paper.status_note}</p>}
            <div className="discovery-links">
              <a href={paper.landing_url} target="_blank" rel="noreferrer">Open official record</a>
              {paper.pdf_url && <a href={paper.pdf_url} target="_blank" rel="noreferrer">Open repository PDF</a>}
            </div>
          </li>;
        })}
      </ul>}
      <div className="discovery-pagination"><button type="button" className="secondary-button" disabled={busy || result.offset === 0}
        onClick={() => void search(Math.max(0, result.offset - result.limit))}>Previous</button>{" "}
        <button type="button" className="secondary-button" disabled={busy || !result.has_more}
          onClick={() => void search(result.offset + result.limit)}>Next</button></div>
    </>}
  </section>;
}
