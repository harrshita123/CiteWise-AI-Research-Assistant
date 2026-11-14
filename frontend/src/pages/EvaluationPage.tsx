import { FormEvent, useEffect, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import { Paper, listPapers } from "../services/papersApi";
import { createEvaluationRun, EvaluationRun, listEvaluationRuns } from "../services/evaluationApi";
import { getCurrentUser } from "../services/authApi";

type DraftCase = { id: string; paperId: string; question: string; expectedPage: string };

function newCase(paperId: string): DraftCase {
  return { id: crypto.randomUUID(), paperId, question: "", expectedPage: "" };
}

function percentage(score: number) {
  return `${Math.round(score * 100)}%`;
}

export default function EvaluationPage() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [cases, setCases] = useState<DraftCase[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    getCurrentUser().then(async (user) => {
      if (!active) return;
      if (user.role !== "admin") { setIsAdmin(false); setLoading(false); return; }
      setIsAdmin(true);
      const [readyPapers, savedRuns] = await Promise.all([listPapers(0, 100, "ready"), listEvaluationRuns()]);
        if (!active) return;
        setPapers(readyPapers);
        setRuns(savedRuns);
        setSelectedRunId(savedRuns[0]?.id ?? null);
        setCases([newCase(readyPapers[0]?.id ?? "")]);
        setLoading(false);
      }).catch((caught: unknown) => {
        if (!active) return;
        setError(caught instanceof Error ? caught.message : "Evaluation could not be loaded.");
        setLoading(false);
      });
    return () => { active = false; };
  }, []);

  function updateCase(id: string, update: Partial<DraftCase>) {
    setCases((current) => current.map((item) => item.id === id ? { ...item, ...update } : item));
  }

  async function runEvaluation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    const invalid = cases.find((item) => {
      const paper = papers.find((candidate) => candidate.id === item.paperId);
      const expectedPage = Number(item.expectedPage);
      return !paper || !item.question.trim() || !Number.isInteger(expectedPage)
        || expectedPage < 1 || expectedPage > paper.page_count;
    });
    if (invalid) {
      setError("Enter a question and a valid page number for every case.");
      return;
    }
    setRunning(true);
    try {
      const run = await createEvaluationRun(cases.map((item) => ({
        paper_id: item.paperId, question: item.question.trim(), expected_page: Number(item.expectedPage),
      })));
      setRuns((current) => [run, ...current].slice(0, 20));
      setSelectedRunId(run.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Evaluation could not finish.");
    } finally {
      setRunning(false);
    }
  }

  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;

  if (isAdmin === false) return <Navigate to="/research" replace />;

  return <main className="evaluation-main">
    <header className="evaluation-header">
      <div><p className="label">Developer tools</p><h1>Retrieval evaluation</h1>
        <p>Check whether CiteWise finds the page that answers each labeled question.</p></div>
      <Link className="secondary-button" to="/research">Open Research</Link>
    </header>

    {loading ? <p className="evaluation-notice" role="status">Loading evaluation...</p> : <>
      {error && <p className="evaluation-error" role="alert">{error}</p>}
      <section className="evaluation-overview" aria-label="Latest evaluation results">
        <div className="evaluation-metric"><span>Hit@5</span><strong>{selectedRun ? percentage(selectedRun.hit_at_5) : "—"}</strong>
          <p>Expected page is in the first five results</p></div>
        <div className="evaluation-metric"><span>Questions tested</span><strong>{selectedRun?.case_count ?? "—"}</strong>
          <p>{selectedRun ? new Date(selectedRun.created_at).toLocaleString() : "Run an evaluation to see results"}</p></div>
      </section>

      <div className="evaluation-grid">
        <section className="evaluation-card" aria-labelledby="evaluation-form-title">
          <div className="evaluation-card-heading"><div><p className="label">Build a test set</p><h2 id="evaluation-form-title">Labeled questions</h2></div><span>{cases.length} / 50</span></div>
          <p className="evaluation-help">For each question, choose the PDF and the page where you know the answer appears. Use the PDF viewer's page number.</p>
          {papers.length === 0 ? <div className="evaluation-empty"><p>Upload and prepare a PDF before evaluating retrieval.</p><Link className="primary-button" to="/papers">Open library</Link></div>
            : <form onSubmit={(event) => void runEvaluation(event)}>
              <div className="evaluation-cases">{cases.map((item, index) => {
                const paper = papers.find((candidate) => candidate.id === item.paperId);
                return <fieldset className="evaluation-case" key={item.id} disabled={running}>
                  <legend>Question {index + 1}</legend>
                  <div className="evaluation-fields">
                    <label>Paper<select value={item.paperId} onChange={(event) => updateCase(item.id, { paperId: event.target.value, expectedPage: "" })} required>
                      {papers.map((option) => <option key={option.id} value={option.id}>{option.filename}</option>)}
                    </select></label>
                    <label>Expected page<input type="number" min={1} max={paper?.page_count} value={item.expectedPage}
                      onChange={(event) => updateCase(item.id, { expectedPage: event.target.value })} placeholder={`1–${paper?.page_count ?? "?"}`} required /></label>
                  </div>
                  <label>Question<textarea value={item.question} maxLength={1000} rows={2} required
                    onChange={(event) => updateCase(item.id, { question: event.target.value })} placeholder="What does this paper say about…?" /></label>
                  {cases.length > 1 && <button className="evaluation-remove" type="button" onClick={() => setCases((current) => current.filter((candidate) => candidate.id !== item.id))}>Remove question</button>}
                </fieldset>;
              })}</div>
              <div className="evaluation-form-actions"><button className="secondary-button" type="button" disabled={running || cases.length >= 50}
                onClick={() => setCases((current) => [...current, newCase(papers[0].id)])}>+ Add question</button>
                <button className="primary-button" type="submit" disabled={running}>{running ? "Evaluating…" : "Run evaluation"}</button></div>
            </form>}
        </section>

        <aside className="evaluation-card evaluation-runs" aria-labelledby="evaluation-runs-title">
          <div className="evaluation-card-heading"><div><p className="label">Past results</p><h2 id="evaluation-runs-title">Recent runs</h2></div><span>{runs.length}</span></div>
          {runs.length === 0 ? <p className="evaluation-help">No runs yet. Add labeled questions and run your first test.</p> :
            <ul>{runs.map((run) => <li key={run.id}><button type="button" className={run.id === selectedRunId ? "active" : ""} onClick={() => setSelectedRunId(run.id)}>
              <strong>{new Date(run.created_at).toLocaleString()}</strong><small>{run.case_count} questions · Hit@5 {percentage(run.hit_at_5)}</small>
            </button></li>)}</ul>}
        </aside>
      </div>

      {selectedRun && <section className="evaluation-card evaluation-breakdown" aria-labelledby="evaluation-breakdown-title">
        <div className="evaluation-card-heading"><div><p className="label">Page-level retrieval</p><h2 id="evaluation-breakdown-title">Question breakdown</h2></div></div>
        <div className="evaluation-result-list">{selectedRun.results.map((item, index) => <article className="evaluation-result" key={`${item.paper_id}-${index}`}>
          <div><span className={`evaluation-outcome ${item.hit_at_5 ? "hit" : "miss"}`}>{item.hit_at_5 ? "In top 5" : "Missed"}</span>
            <h3>{item.question}</h3><p>{papers.find((paper) => paper.id === item.paper_id)?.filename ?? "Paper unavailable"}</p></div>
          <div className="evaluation-pages"><span>Expected page <strong>{item.expected_page}</strong></span>
            <span>Returned pages <strong>{item.returned_pages.length ? item.returned_pages.join(", ") : "None"}</strong></span></div>
        </article>)}</div>
      </section>}
      <p className="evaluation-footnote">These scores measure page retrieval only. They do not measure whether generated answers are correct.</p>
    </>}
  </main>;
}
