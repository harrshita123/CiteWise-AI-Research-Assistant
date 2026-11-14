import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listPapers, paperFileUrl, Paper } from "../services/papersApi";
import { AnswerFeedback, Conversation, getConversation, listConversations, saveAnswerFeedback, SavedMessage } from "../services/researchApi";
import { evidenceLabel, evidenceText } from "../utils/evidenceOutcome";
import CitedAnswer from "../components/CitedAnswer";

function passageText(text: string, excerpt: string) {
  const start = excerpt ? text.indexOf(excerpt) : -1;
  if (start < 0) return text;
  return <>{text.slice(0, start)}<mark className="source-highlight">{excerpt}</mark>{text.slice(start + excerpt.length)}</>;
}

export default function AnswerCheckPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selected, setSelected] = useState<Conversation | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [reasons, setReasons] = useState<Record<string, "wrong_page" | "unsupported_answer" | "other">>({});
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    Promise.all([listConversations(), listPapers(0, 100)]).then(async ([saved, allPapers]) => {
      if (!active) return;
      setConversations(saved);
      setPapers(allPapers);
      if (saved.length) setSelected(await getConversation(saved[0].id));
    }).catch((caught: unknown) => {
      if (active) setError(caught instanceof Error ? caught.message : "Answer check could not load.");
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  async function openConversation(id: string) {
    if (busy) return;
    setBusy(true);
    setError("");
    try { setSelected(await getConversation(id)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Saved answer could not be opened."); }
    finally { setBusy(false); }
  }

  async function rate(message: SavedMessage, feedback: AnswerFeedback) {
    if (!selected) return;
    setSavingId(message.id);
    setError("");
    try {
      const saved = await saveAnswerFeedback(selected.id, message.id, feedback);
      setSelected((current) => current && current.id === selected.id
        ? { ...current, messages: current.messages.map((item) => item.id === message.id ? { ...item, feedback: saved } : item) }
        : current);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Feedback could not be saved.");
    } finally { setSavingId(null); }
  }

  const paperName = (id: string) => papers.find((paper) => paper.id === id)?.filename ?? "Paper unavailable";

  return <main className="answer-check-main">
    <header className="research-header">
      <div><p className="label">Your answers and sources</p><h1>Answer check</h1>
        <p className="research-intro">Review what CiteWise answered, read the supporting passage, and tell us when something is wrong.</p></div>
      <Link className="research-library-link" to="/research">Ask a question <span aria-hidden="true">↗</span></Link>
    </header>
    {error && <p className="form-error" role="alert">{error}</p>}
    {loading ? <p role="status">Loading saved answers…</p> : conversations.length === 0
      ? <section className="research-results"><h2>No answers to check yet</h2><p>Ask a question about a paper, then return here to review its evidence.</p><Link className="primary-button" to="/research">Open Research</Link></section>
      : <div className="answer-check-grid">
        <aside className="research-history answer-check-history" aria-label="Saved conversations">
          <div className="research-history-heading"><div><p className="label">Your workspace</p><h2>Saved answers</h2></div><span>{conversations.length}</span></div>
          <ul>{conversations.map((conversation) => <li key={conversation.id}>
            <button type="button" className={selected?.id === conversation.id ? "selected" : ""} disabled={busy} onClick={() => void openConversation(conversation.id)}>
              <strong>{conversation.messages.at(-1)?.question ?? "Saved question"}</strong><small>{paperName(conversation.paper_id)}</small>
            </button>
          </li>)}</ul>
        </aside>
        {selected && <section className="research-results answer-check-detail" aria-label="Answer review">
          <div className="research-result-heading"><div><p className="label">{paperName(selected.paper_id)}</p><h2>Questions and evidence</h2></div></div>
          {selected.messages.map((message) => <article className="answer-check-item" key={message.id}>
            <h3>{message.question}</h3>
            <strong className={`evidence-status evidence-status-${message.status}`}>{evidenceLabel(message.status)}</strong>
            <p className="answer-check-answer"><CitedAnswer text={evidenceText(message.status, message.answer)} sources={message.sources} externalSources={message.external_sources} sourcePrefix={`check-source-${message.id}`} detailsId={`check-sources-${message.id}`} /> {!message.external_sources?.length && message.sources.map((source, index) => <a href={`#check-source-${message.id}-${index}`} className="answer-citation" key={source.id} aria-label={`Show source ${index + 1}, page ${source.page_number}`} onClick={() => {
                const details = document.getElementById(`check-sources-${message.id}`);
                if (details instanceof HTMLDetailsElement) details.open = true;
              }}>[{index + 1}]</a>)}</p>
            {message.sources.length > 0 && <details className="answer-sources" id={`check-sources-${message.id}`}>
              <summary><span className="sources-closed-label">Show</span><span className="sources-open-label">Hide</span> supporting {message.sources.length === 1 ? "passage" : "passages"}</summary>
              <p className="research-source-help">Compare the highlighted text with the answer and original PDF.</p>
              {message.sources.map((source, index) => <div className="paper-chunk" key={source.id} id={`check-source-${message.id}-${index}`}>
                <h4>Page {source.page_number}</h4><p>{passageText(source.text, source.highlight_text)}</p>
                <a href={`${paperFileUrl(selected.paper_id)}#page=${source.page_number}`} target="_blank" rel="noreferrer">Open page {source.page_number} in original PDF ↗</a>
              </div>)}
            </details>}
            {!!message.external_sources?.length && <div className="external-sources"><strong>External sources</strong><ul>{message.external_sources.map((source) => <li key={source.id}><a href={source.url} target="_blank" rel="noreferrer">[{source.id}] {source.title} ↗</a>{source.published_date && <small> · Search-reported date: {source.published_date}</small>}{source.excerpt && <details><summary>View source excerpt</summary><p>{source.excerpt}</p></details>}</li>)}</ul></div>}
            <div className="answer-feedback">
              <span>Was this answer helpful?</span>
              <div className="answer-feedback-actions">
                <button type="button" aria-pressed={message.feedback?.rating === "helpful"} disabled={savingId === message.id} onClick={() => void rate(message, { rating: "helpful", reason: null })}>Helpful</button>
                <button type="button" aria-pressed={message.feedback?.rating === "incorrect"} disabled={savingId === message.id} onClick={() => setReasons((current) => ({ ...current, [message.id]: current[message.id] ?? "unsupported_answer" }))}>Incorrect</button>
              </div>
              {reasons[message.id] && <div className="answer-feedback-reason">
                <label htmlFor={`check-reason-${message.id}`}>What was wrong?</label>
                <select id={`check-reason-${message.id}`} value={reasons[message.id]} onChange={(event) => setReasons((current) => ({ ...current, [message.id]: event.target.value as "wrong_page" | "unsupported_answer" | "other" }))}>
                  <option value="unsupported_answer">Answer does not match the source</option><option value="wrong_page">Wrong page</option><option value="other">Other</option>
                </select>
                <button type="button" disabled={savingId === message.id} onClick={() => void rate(message, { rating: "incorrect", reason: reasons[message.id] })}>Send feedback</button>
              </div>}
              {message.feedback && <small role="status">Feedback saved: {message.feedback.rating === "helpful" ? "Helpful" : "Incorrect"}.</small>}
            </div>
          </article>)}
        </section>}
      </div>}
    <p className="answer-check-note">Sources help you check an answer. They do not guarantee every claim is correct.</p>
  </main>;
}
