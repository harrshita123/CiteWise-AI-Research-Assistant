import { useEffect, useState } from "react";
import { getAnswerQuality, listAnswerQuality, QualityDetail, QualityMonitor, QualityRow } from "../services/evaluationApi";
import { listPapers, paperFileUrl, Paper } from "../services/papersApi";

const pageSize = 25;
const summaryLabels = {
  citation_coverage: "Average Citation Coverage", citation_precision: "Average Citation Precision",
  claim_support: "Claim Support Rate", completeness: "Answer Completeness", relevance: "Answer Relevance",
  contradiction_rate: "Contradiction Rate",
};
const percentage = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 });
const percent = (score?: number | null) => score == null ? "Not measured" : percentage.format(score);
const words = (value: string) => value.toLowerCase().replaceAll("_", " ");

export default function AnswerQualityMonitor() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [paper, setPaper] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [evidence, setEvidence] = useState("");
  const [state, setState] = useState("");
  const [priority, setPriority] = useState("");
  const [offset, setOffset] = useState(0);
  const [monitor, setMonitor] = useState<QualityMonitor | null>(null);
  const [error, setError] = useState("");
  const [paperError, setPaperError] = useState("");
  const [selected, setSelected] = useState<QualityRow | null>(null);
  const [detail, setDetail] = useState<QualityDetail | null>(null);
  const [detailError, setDetailError] = useState("");
  const query = new URLSearchParams(Object.entries({ paper_id: paper, date_from: from, date_to: to,
    evidence, state, priority, offset: String(offset), limit: String(pageSize) }).filter(([, value]) => value)).toString();

  useEffect(() => {
    let active = true;
    listPapers(0, 100).then(result => { if (active) setPapers(result); })
      .catch(() => { if (active) setPaperError("The PDF filter is temporarily unavailable."); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    async function refresh() {
      try {
        const result = await listAnswerQuality(query, controller.signal);
        if (!controller.signal.aborted) { setMonitor(result); setError(""); }
      } catch (caught) {
        if (!controller.signal.aborted) setError((caught as Error).message);
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(refresh, 30000);
      }
    }
    void refresh();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [query]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    getAnswerQuality(selected, controller.signal).then(result => {
      if (!controller.signal.aborted) setDetail(result);
    }).catch(caught => { if (!controller.signal.aborted) setDetailError(caught.message); });
    return () => controller.abort();
  }, [selected]);

  function change(setter: (value: string) => void, value: string) {
    setter(value); setOffset(0); setSelected(null); setMonitor(null); setError("");
  }

  return <section className="answer-quality" aria-labelledby="answer-quality-title">
    <h2 id="answer-quality-title">Answer Quality Monitoring</h2>
    <div className="answer-quality-filters">
      <label>PDF<select value={paper} onChange={event => change(setPaper, event.target.value)}>
        <option value="">All PDFs</option>{papers.map(item => <option value={item.id} key={item.id}>{item.filename}</option>)}
      </select></label>
      <label>From (UTC)<input type="date" value={from} onChange={event => change(setFrom, event.target.value)} /></label>
      <label>Through (UTC)<input type="date" value={to} onChange={event => change(setTo, event.target.value)} /></label>
      <label>Evidence<select value={evidence} onChange={event => change(setEvidence, event.target.value)}>
        <option value="">All evidence types</option><option value="pdf">PDF</option><option value="external">External</option>
        <option value="mixed">Mixed</option><option value="none">None</option>
      </select></label>
      <label>Evaluation state<select value={state} onChange={event => change(setState, event.target.value)}>
        <option value="">All states</option>{["COMPLETE", "PENDING", "FAILED", "UNAVAILABLE"].map(item => <option key={item} value={item}>{words(item)}</option>)}
      </select></label>
      <label>Priority<select value={priority} onChange={event => change(setPriority, event.target.value)}>
        <option value="">Both priorities</option><option value="HIGH">High</option><option value="MEDIUM">Medium</option>
      </select></label>
    </div>
    {paperError && <p role="status">{paperError}</p>}
    {error && <p role="alert">{error}</p>}
    {!monitor && !error && <p role="status">Loading answer evaluations…</p>}
    {monitor && <>
      <p>{monitor.evaluated_answers} evaluated answers · {monitor.total_versions} versions · {monitor.counts.pending} pending · {monitor.counts.failed} failed · {monitor.counts.unavailable} unavailable</p>
      {monitor.total_versions ? <>
        <dl className="answer-quality-summary">
          {Object.entries(summaryLabels).map(([key, label]) => {
            const measure = monitor.metrics[key];
            return <div key={key}><dt>{label}</dt><dd>{percent(measure?.average)}</dd>
              <dd>{measure?.measured ?? 0}/{monitor.total_versions} versions measured{measure?.denominator ? ` · ${measure.numerator}/${measure.denominator} observations` : ""}</dd></div>;
          })}
        </dl>
        <div className="admin-table-wrap"><table className="admin-table">
          <thead><tr><th>Question</th><th>Evidence</th><th>Evaluated</th><th>Coverage</th><th>Precision</th><th>Completeness</th><th>High</th><th>Medium</th><th>State</th></tr></thead>
          <tbody>{monitor.rows.map(row => <tr key={`${row.conversation_id}/${row.message_id}/${row.revision}`}>
            <td><button type="button" className="secondary-button small" onClick={() => {
              if (selected === row) return;
              setSelected(row); setDetail(null); setDetailError("");
            }}>{row.question}</button></td>
            <td>{row.paper_filename || row.paper_id} · {words(row.evidence_type)}</td>
            <td>{new Date(row.evaluated_at).toLocaleString()} · v{row.revision}</td>
            {(["citation_coverage", "citation_precision", "completeness"]).map(key => <td key={key}>{percent(row.metrics[key]?.score)}</td>)}
            <td>{words(row.high_status)}</td><td>{words(row.medium_status)}</td><td>{words(row.state)}</td>
          </tr>)}</tbody>
        </table></div>
        {monitor.total_versions > pageSize && <nav aria-label="Answer evaluation pages" className="paper-pagination">
          <button className="secondary-button small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - pageSize))}>Previous</button>
          <span>Page {offset / pageSize + 1}</span>
          <button className="secondary-button small" disabled={offset + pageSize >= monitor.total_versions} onClick={() => setOffset(offset + pageSize)}>Next</button>
        </nav>}
      </> : <p>No answer evaluations yet{from || to || evidence || state || priority ? " for these filters" : ""}.</p>}
    </>}
    {selected && <section className="answer-quality-detail" aria-label="Answer evaluation details">
      <button className="secondary-button small" onClick={() => setSelected(null)}>Close details</button>
      {detailError && <p role="alert">{detailError}</p>}
      {!detail && !detailError && <p role="status">Loading evaluation details…</p>}
      {detail && <QualityDetails detail={detail} priority={priority} />}
    </section>}
  </section>;
}

function QualityDetails({ detail, priority }: { detail: QualityDetail; priority: string }) {
  return <>
    <h3>{detail.question}</h3>
    <p>Answer version {detail.revision} · Evaluator {detail.evaluator_version}</p>
    {detail.response_ms != null && <p>Response preparation {detail.response_ms} ms · Evaluation {detail.evaluation_ms} ms</p>}
    {!!detail.model_calls?.length && <details><summary>Model calls</summary><ul>
      {detail.model_calls.map((call, index) => <li key={index}>
        {call.role.replaceAll("_", " ")} · {call.model_id} · {call.provider} · {call.latency_ms} ms
        {call.cached ? " · Cached" : call.error ? ` · ${call.error}` : call.status ? ` · HTTP ${call.status}` : ""}
        {call.total_tokens != null && ` · ${call.total_tokens} tokens`}
      </li>)}
    </ul></details>}
    {detail.generated_answer && <details open><summary>Generated draft</summary><p>{detail.generated_answer}</p></details>}
    {detail.answer && <details><summary>Answer shown to the user</summary><p>{detail.answer}</p></details>}
    <h4>Checks</h4><ul>{Object.values(detail.metrics).filter(metric => !priority || metric.priority === priority).map(metric =>
      <li key={metric.id}><strong>{words(metric.id)}</strong> · {words(metric.status)} · {percent(metric.score)}
        {metric.denominator ? ` (${metric.numerator}/${metric.denominator})` : ""} · {words(metric.reason)}</li>)}</ul>
    {!!detail.requirements?.length && <><h4>Question requirements</h4><ul>{detail.requirements.map(part =>
      <li key={`${part.subject}/${part.dimension}`}>{[part.subject, part.dimension].filter(Boolean).join(" · ")} — {part.addressed ? "Addressed" : "Missing"}</li>)}</ul></>}
    {!!detail.claims?.length && <><h4>Claims and evidence</h4><ul>{detail.claims.map(claim =>
      <li key={claim.claim_id}><strong>{words(claim.status)}</strong> · {words(claim.section)} · {claim.text}
        <p>{claim.attached_ids.length ? `Draft cites ${claim.attached_ids.join(", ")}` : "No citation attached to this draft claim"}
          {claim.reason ? ` · ${words(claim.reason)}` : ""}</p>
        {!!claim.evaluated_ids.length && <p>Supporting citations {claim.supporting_ids.join(", ") || "none"} · Checked {claim.evaluated_ids.join(", ")}</p>}
        {claim.supporting.map((quote, index) => <blockquote key={`support-${index}`}>[{quote.source_id}] {quote.quote}</blockquote>)}
        {claim.opposing.map((quote, index) => <blockquote key={`oppose-${index}`}>Opposing evidence [{quote.source_id}] {quote.quote}</blockquote>)}
      </li>)}</ul></>}
    {!!detail.sources?.length && <><h4>Source links</h4><ul>{detail.sources.map(source => <li key={source.id}>
      {source.url?.startsWith("https://") ? <a href={source.url} target="_blank" rel="noreferrer">[{source.id}] {source.title}</a>
        : source.page_number ? <a href={`${paperFileUrl(detail.paper_id)}#page=${source.page_number}`} target="_blank" rel="noreferrer">[{source.id}] PDF page {source.page_number}</a> : source.id}
    </li>)}</ul></>}
  </>;
}
