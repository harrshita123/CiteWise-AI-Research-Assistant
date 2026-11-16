import { FormEvent, useEffect, useRef, useState } from "react";
import { Navigate, useOutletContext, useSearchParams } from "react-router-dom";
import { AuthApiError, getCurrentUser, SignedInUser } from "../services/authApi";
import { getPaper, listPapers, paperFileUrl, Paper } from "../services/papersApi";
import WorkspaceIcon from "../components/WorkspaceIcon";
import { answerGenerationAvailable, askPaper, AskResult, getConversation, ResearchApiError, RetrievedPassage, SavedMessage, saveAnswerFeedback, searchPaper } from "../services/researchApi";
import { evidenceText } from "../utils/evidenceOutcome";
import { matchesAnswerRequest } from "../utils/answerRequest";
import { deadlineFromRetry, remainingSeconds } from "../utils/groqCooldown";
import CitedAnswer from "../components/CitedAnswer";
import type { ConversationState } from "../components/WorkspaceLayout";

function highlightedSource(text: string, excerpt: string) {
  const start = excerpt ? text.indexOf(excerpt) : -1;
  if (start < 0) return text;
  return <>{text.slice(0, start)}<mark className="source-highlight">{excerpt}</mark>{text.slice(start + excerpt.length)}</>;
}

type ResearchDraft = {
  paperId: string;
  result: { paperId: string; passages: RetrievedPassage[] } | null;
  answerResult: AskResult | null;
  conversationId?: string | null;
  messages?: SavedMessage[];
};

const pendingQuestionKey = (userId: string) => `citewise-pending-question:${userId}`;

function readResearchDraft(userId: string, paperId?: string): ResearchDraft | null {
  try {
    const saved = sessionStorage.getItem(`citewise-research:${userId}${paperId ? `:${paperId}` : ""}`);
    return saved ? JSON.parse(saved) as ResearchDraft : null;
  } catch {
    return null;
  }
}

function saveResearchDraft(userId: string, draft: ResearchDraft) {
  try {
    sessionStorage.setItem(`citewise-research:${userId}:${draft.paperId}`, JSON.stringify(draft));
    sessionStorage.setItem(`citewise-research:${userId}`, JSON.stringify({ paperId: draft.paperId }));
  } catch {
    // Research still works when browser storage is unavailable.
  }
}

export default function ResearchPage() {
  const [params, setParams] = useSearchParams();
  const { refreshConversations } = useOutletContext<ConversationState>();
  const [user, setUser] = useState<SignedInUser | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [paperId, setPaperId] = useState(params.get("paper") ?? "");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<{ paperId: string; passages: RetrievedPassage[] } | null>(null);
  const [answerResult, setAnswerResult] = useState<AskResult | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<SavedMessage[]>([]);
  const [canAnswer, setCanAnswer] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [expired, setExpired] = useState(false);
  const [error, setError] = useState("");
  const [retryUntil, setRetryUntil] = useState(0);
  const [retryRemaining, setRetryRemaining] = useState(0);
  const [feedbackError, setFeedbackError] = useState("");
  const [feedbackBusy, setFeedbackBusy] = useState<string | null>(null);
  const [feedbackReasons, setFeedbackReasons] = useState<Record<string, "wrong_page" | "unsupported_answer" | "other">>({});
  const [attempt, setAttempt] = useState(0);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const openedFromRouteRef = useRef<string | null>(null);
  const questionRef = useRef<HTMLTextAreaElement>(null);
  const paperSelectRef = useRef<HTMLSelectElement>(null);
  const activeRequestRef = useRef<string | null>(null);
  const contextVersionRef = useRef(0);
  const capabilityVersionRef = useRef(0);

  function invalidateRequest() {
    contextVersionRef.current += 1;
    activeRequestRef.current = null;
    setPendingQuestion(null);
    setBusy(false);
  }

  useEffect(() => {
    let active = true;
    try {
      sessionStorage.removeItem("citewise-groq-key-revision");
      sessionStorage.removeItem("citewise-groq-retry-until-v2");
    } catch { /* Storage is optional. */ }
    const refreshCapability = () => {
      const version = ++capabilityVersionRef.current;
      void answerGenerationAvailable().then((capability) => {
        if (!active || version !== capabilityVersionRef.current) return;
        setCanAnswer(capability.available);
        const deadline = deadlineFromRetry(capability.retryAfterSeconds);
        setRetryUntil(deadline);
        setRetryRemaining(remainingSeconds(deadline));
        if (!deadline) setError((current) => current.includes("Try again in ") ? "" : current);
      }).catch(() => { /* Passage search remains available if capabilities cannot load. */ });
    };
    refreshCapability();
    window.addEventListener("focus", refreshCapability);
    Promise.all([getCurrentUser(), listPapers(0, 100, "ready")]).then(
      ([currentUser, readyPapers]) => {
        if (!active) return;
        setUser(currentUser);
        setPapers(readyPapers);
        const previousSelection = readResearchDraft(currentUser.id);
        const requestedPaper = params.get("paper");
        const selectedPaper = [requestedPaper, previousSelection?.paperId].find((id) =>
          id && readyPapers.some((paper) => paper.id === id)) ?? "";
        const draft = readResearchDraft(currentUser.id, selectedPaper)
          ?? (previousSelection?.paperId === selectedPaper ? previousSelection : null);
        setPaperId(selectedPaper);
        try {
          setQuestion(sessionStorage.getItem(pendingQuestionKey(currentUser.id)) ?? "");
          sessionStorage.removeItem(pendingQuestionKey(currentUser.id));
        } catch { setQuestion(""); }
        if (draft && !params.get("conversation")) {
          setResult(draft.result?.paperId === selectedPaper ? draft.result : null);
        }
        setLoading(false);
      },
      (caught: unknown) => {
        if (!active) return;
        if (caught instanceof AuthApiError && caught.status === 401) setExpired(true);
        else setError(caught instanceof Error ? caught.message : "Research could not be loaded.");
        setLoading(false);
      },
    );
    return () => { active = false; window.removeEventListener("focus", refreshCapability); };
  }, [attempt]);

  useEffect(() => {
    if (!user || loading) return;
    if (paperId) saveResearchDraft(user.id, { paperId, result, answerResult, conversationId, messages });
  }, [user, loading, paperId, result, answerResult, conversationId, messages]);

  useEffect(() => {
    const transcript = transcriptRef.current;
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
  }, [messages.length, conversationId, result, pendingQuestion]);

  useEffect(() => {
    const field = questionRef.current;
    if (!field) return;
    field.style.height = "auto";
    field.style.height = `${Math.min(field.scrollHeight, 180)}px`;
  }, [question]);

  useEffect(() => {
    if (!retryUntil) return;
    const update = () => setRetryRemaining(remainingSeconds(retryUntil));
    update();
    const timer = window.setInterval(update, 1000);
    return () => window.clearInterval(timer);
  }, [retryUntil]);

  useEffect(() => {
    if (loading) return;
    const targetId = params.get("conversation");
    if (!targetId) {
      if (openedFromRouteRef.current) {
        setConversationId(null); setMessages([]); setAnswerResult(null); setResult(null);
        setQuestion(""); setError("");
      }
      openedFromRouteRef.current = null;
      return;
    }
    if (targetId === openedFromRouteRef.current) return;
    openedFromRouteRef.current = targetId;
    invalidateRequest();
    setConversationId(null); setMessages([]); setAnswerResult(null); setResult(null);
    setQuestion(""); setError("");
    void openSaved(targetId);
  }, [loading, params]);

  function selectPaper(nextPaperId: string) {
    if (nextPaperId === paperId) return;
    invalidateRequest();
    if (user && paperId) saveResearchDraft(user.id, { paperId, result, answerResult, conversationId, messages });
    const draft = user ? readResearchDraft(user.id, nextPaperId) : null;
    setPaperId(nextPaperId);
    setQuestion("");
    setResult(draft?.result?.paperId === nextPaperId ? draft.result : null);
    setAnswerResult(null);
    setConversationId(null);
    setMessages([]);
    setError("");
  }

  async function submit(mode: "search" | "ask") {
    if (!paperId || !question.trim() || activeRequestRef.current || (mode === "ask" && retryUntil > Date.now())) return;
    const askedQuestion = question.trim();
    const selectedPaperId = paperId;
    const selectedConversationId = conversationId;
    const replyToMessageId = messages.at(-1)?.id;
    const requestId = crypto.randomUUID();
    const contextVersion = contextVersionRef.current;
    activeRequestRef.current = requestId;
    setBusy(true);
    if (mode === "ask") { setPendingQuestion(askedQuestion); setQuestion(""); }
    setError("");
    setResult(null);
    setAnswerResult(null);
    try {
      if (mode === "ask") {
        const answer = await askPaper(selectedPaperId, askedQuestion, selectedConversationId ?? undefined,
          requestId, replyToMessageId);
        if (activeRequestRef.current !== requestId || contextVersionRef.current !== contextVersion) return;
        if (!matchesAnswerRequest(answer, { requestId, paperId: selectedPaperId,
          question: askedQuestion, conversationId: selectedConversationId, contextVersion },
          activeRequestRef.current, contextVersionRef.current)) {
          throw new Error("The answer did not match this question. Please try again.");
        }
        capabilityVersionRef.current += 1;
        setRetryUntil(0);
        setRetryRemaining(0);
        setAnswerResult(answer);
        setConversationId(answer.conversation_id);
        if (answer.conversation_id) {
          openedFromRouteRef.current = answer.conversation_id;
          setParams({ conversation: answer.conversation_id }, { replace: true });
        }
        setMessages((current) => [...current, {
          id: answer.message_id ?? crypto.randomUUID(), question: askedQuestion, status: answer.status,
          answer: answer.answer, sources: answer.sources, external_sources: answer.external_sources,
          assessment: answer.assessment, created_at: new Date().toISOString(),
        }]);
        void refreshConversations().catch(() => {});
      } else {
        const passages = await searchPaper(selectedPaperId, askedQuestion);
        if (activeRequestRef.current !== requestId || contextVersionRef.current !== contextVersion) return;
        setResult({ paperId: selectedPaperId, passages });
        setQuestion("");
      }
    } catch (caught) {
      if (activeRequestRef.current !== requestId || contextVersionRef.current !== contextVersion) return;
      if (mode === "ask") setQuestion(askedQuestion);
      if (caught instanceof ResearchApiError && caught.status === 401) {
        try { if (user) sessionStorage.setItem(pendingQuestionKey(user.id), askedQuestion); } catch { /* Storage is optional. */ }
        setExpired(true);
      }
      if (caught instanceof ResearchApiError && caught.status === 429 && caught.retryAfterSeconds) {
        setRetryUntil((current) => Math.max(current, deadlineFromRetry(caught.retryAfterSeconds)));
      }
      setError(caught instanceof Error ? caught.message : "Research could not finish.");
    } finally {
      if (activeRequestRef.current === requestId) {
        activeRequestRef.current = null;
        setPendingQuestion(null);
        setBusy(false);
      }
    }
  }

  async function openSaved(conversationId: string) {
    const contextVersion = ++contextVersionRef.current;
    setBusy(true);
    setError("");
    try {
      const saved = await getConversation(conversationId);
      if (contextVersionRef.current !== contextVersion) return;
      const latest = saved.messages.at(-1);
      if (!papers.some((paper) => paper.id === saved.paper_id)) {
        const detail = await getPaper(saved.paper_id).catch(() => null);
        if (contextVersionRef.current !== contextVersion) return;
        if (!detail || detail.paper.status !== "ready") {
          setError("This conversation's paper is no longer available.");
          return;
        }
        setPapers((items) => [...items, detail.paper]);
      }
      if (saved.paper_id !== paperId) {
        if (user && paperId) saveResearchDraft(user.id, { paperId, result, answerResult, conversationId, messages });
        setPaperId(saved.paper_id);
      }
      setConversationId(saved.id);
      setMessages(saved.messages);
      setQuestion("");
      setResult(null);
      setAnswerResult(latest ? { paper_id: saved.paper_id, conversation_id: saved.id, message_id: latest.id,
        status: latest.status, answer: latest.answer, sources: latest.sources } : null);
    } catch (caught) {
      if (contextVersionRef.current !== contextVersion) return;
      setConversationId(null); setMessages([]); setAnswerResult(null); setResult(null); setQuestion("");
      setError(caught instanceof Error ? caught.message : "Saved answer could not be opened.");
    } finally {
      if (contextVersionRef.current === contextVersion) setBusy(false);
    }
  }

  async function rateAnswer(message: SavedMessage, rating: "helpful" | "incorrect", reason: "wrong_page" | "unsupported_answer" | "other" | null = null) {
    if (!conversationId) return;
    setFeedbackBusy(message.id);
    setFeedbackError("");
    try {
      const feedback = await saveAnswerFeedback(conversationId, message.id, { rating, reason });
      setMessages((current) => current.map((item) => item.id === message.id ? { ...item, feedback } : item));
    } catch (caught) {
      setFeedbackError(caught instanceof Error ? caught.message : "Feedback could not be saved.");
    } finally {
      setFeedbackBusy(null);
    }
  }

  if (expired) return <Navigate to="/login?next=/research" replace />;
  if (loading) return <main className="session-check" role="status">Loading research...</main>;
  if (!user) return <main className="session-check"><p>{error}</p><button className="primary-button" onClick={() => { setLoading(true); setAttempt((value) => value + 1); }}>Try again</button></main>;

  return <>
    <main className="paper-main research-main">
      <header className="research-header">
        <div><h1>Ask your paper</h1>
          <p className="research-intro">Get a concise answer, then open its supporting passage when you need to verify it.</p></div>
      </header>
      <div className="research-grid">
        <div className="research-content">
          <div className="research-transcript" ref={transcriptRef} aria-label="Research conversation">
          {messages.length === 0 && !result && !answerResult && !busy && <div className="research-welcome">
            <h2>Ask your paper anything</h2><p>Your answers and supporting passages will appear here.</p>
          </div>}
          {messages.length > 0 && <section className="research-results research-conversation" aria-label="Conversation" aria-live="polite">
            {feedbackError && <p className="form-error" role="alert">{feedbackError}</p>}
            <div className="research-message-list">{messages.map((message) => <article className="research-message" key={message.id}>
              <div className="research-message-question"><span>You</span><p>{message.question}</p></div>
              <div className="research-message-answer">
                <p><CitedAnswer text={evidenceText(message.status, message.answer)} sources={message.sources} externalSources={message.external_sources} sourcePrefix={`source-${message.id}`} detailsId={`sources-${message.id}`} /></p>
                {message.sources.length > 0 && <details className="answer-sources" id={`sources-${message.id}`}>
                  <summary><span className="sources-closed-label">Show</span><span className="sources-open-label">Hide</span> supporting {message.sources.length === 1 ? "passage" : "passages"}</summary>
                  <p className="research-source-help">Check the highlighted text against the answer and original PDF.</p>
                  {message.sources.map((source, index) => <article className="paper-chunk" key={source.id} id={`source-${message.id}-${index}`}>
                    <h3>Page {source.page_number}</h3><p>{highlightedSource(source.text, source.highlight_text)}</p>
                    <a href={`${paperFileUrl(paperId)}#page=${source.page_number}`} target="_blank" rel="noreferrer">Open original PDF ↗</a>
                  </article>)}
                </details>}
                {!!message.external_sources?.length && <div className="external-sources"><strong>External sources</strong><ul>{message.external_sources.map((source) => <li key={source.id}><a href={source.url} target="_blank" rel="noreferrer">[{source.id}] {source.title} ↗</a>{source.published_date && <small> · Search-reported date: {source.published_date}</small>}{source.excerpt && <details><summary>View source excerpt</summary><p>{source.excerpt}</p></details>}</li>)}</ul></div>}
                <div className="answer-feedback">
                  <div className="answer-feedback-actions">
                    <button type="button" title="Helpful" aria-label="Helpful" disabled={feedbackBusy === message.id} aria-pressed={message.feedback?.rating === "helpful"} onClick={() => void rateAnswer(message, "helpful")}><WorkspaceIcon name="thumbUp" size={16} /></button>
                    <button type="button" title="Incorrect" aria-label="Incorrect" disabled={feedbackBusy === message.id} aria-pressed={message.feedback?.rating === "incorrect"} onClick={() => setFeedbackReasons((current) => ({ ...current, [message.id]: current[message.id] ?? "unsupported_answer" }))}><WorkspaceIcon name="thumbDown" size={16} /></button>
                  </div>
                  {feedbackReasons[message.id] && <div className="answer-feedback-reason"><label htmlFor={`reason-${message.id}`}>What was wrong?</label>
                    <select id={`reason-${message.id}`} value={feedbackReasons[message.id]} onChange={(event) => setFeedbackReasons((current) => ({ ...current, [message.id]: event.target.value as "wrong_page" | "unsupported_answer" | "other" }))}>
                      <option value="unsupported_answer">Answer does not match the source</option><option value="wrong_page">Wrong page</option><option value="other">Other</option>
                    </select><button type="button" disabled={feedbackBusy === message.id} onClick={() => void rateAnswer(message, "incorrect", feedbackReasons[message.id])}>Send feedback</button></div>}
                  {message.feedback && <small role="status">Feedback saved: {message.feedback.rating === "helpful" ? "Helpful" : "Incorrect"}.</small>}
                </div>
              </div>
            </article>)}</div>
          </section>}
          {pendingQuestion && <div className="research-pending" role="status">
            <div className="research-message-question"><span>You</span><p>{pendingQuestion}</p></div>
            <p className="research-pending-status">Searching your paper and preparing an answer…</p>
          </div>}
          {busy && !pendingQuestion && <p className="research-progress" role="status">Searching your paper…</p>}
          {answerResult && messages.length === 0 && <section className="research-results research-answer-card" aria-live="polite">
            <div className="research-result-heading"><span className="research-step">02</span><div><p className="label">From your paper</p><h2>Answer</h2></div></div>
            <p className="research-answer"><CitedAnswer text={evidenceText(answerResult.status, answerResult.answer)} sources={answerResult.sources} externalSources={answerResult.external_sources} sourcePrefix="fallback-source" detailsId="fallback-sources" /></p>
            {answerResult.sources.length > 0 && <details className="answer-sources" id="fallback-sources">
              <summary><span className="sources-closed-label">Show</span><span className="sources-open-label">Hide</span> supporting {answerResult.sources.length === 1 ? "passage" : "passages"}</summary>
              <p className="research-source-help">Highlighted text supports the answer. Open the original PDF to check its context.</p>
              {answerResult.sources.map((source, index) => <article className="paper-chunk" key={source.id} id={`fallback-source-${index}`}>
                <h3>Page {source.page_number}</h3>
                <p>{highlightedSource(source.text, source.highlight_text)}</p>
                <a href={`${paperFileUrl(answerResult.paper_id)}#page=${source.page_number}`} target="_blank" rel="noreferrer">Open page {source.page_number} in original PDF ↗</a>
              </article>)}
            </details>}
            {!!answerResult.external_sources?.length && <div className="external-sources"><strong>External sources</strong><ul>{answerResult.external_sources.map((source) => <li key={source.id}><a href={source.url} target="_blank" rel="noreferrer">[{source.id}] {source.title} ↗</a>{source.published_date && <small> · Search-reported date: {source.published_date}</small>}{source.excerpt && <details><summary>View source excerpt</summary><p>{source.excerpt}</p></details>}</li>)}</ul></div>}
          </section>}
          {result && <section className="research-results research-passages" aria-live="polite">
            <div className="research-result-heading"><span className="research-step">02</span><div><p className="label">Search results</p><h2>Relevant passages</h2></div></div>
            <p>Most relevant passages first. Open the original page to verify the context.</p>
            {result.passages.length === 0 ? <p>No saved passages were found for this paper.</p> : result.passages.map((passage, index) => <article className={`paper-chunk${index === 0 ? " research-top-passage" : ""}`} key={passage.chunk_index}>
              <h3>Page {passage.page_number}{index === 0 && <span>Best match</span>}</h3><p>{highlightedSource(passage.text, passage.highlight_text ?? "")}</p>
              <a href={`${paperFileUrl(result.paperId)}#page=${passage.page_number}`} target="_blank" rel="noreferrer">Open page {passage.page_number} in original PDF ↗</a>
            </article>)}
          </section>}
          </div>
          <form className="research-form research-composer" aria-label="Ask your paper" onSubmit={(event: FormEvent<HTMLFormElement>) => { event.preventDefault(); void submit(canAnswer ? "ask" : "search"); }}>
            {error && <p className="form-error" role="alert">{error}</p>}
            {!error && retryRemaining > 0 && <p className="form-error" role="status">Answer generation is rate limited. Try again in {Math.ceil(retryRemaining / 60)} min.</p>}
            <div className="research-paper-tools">
              <select ref={paperSelectRef} aria-label="Select Library PDF" value={paperId} disabled={busy} onChange={(event) => { openedFromRouteRef.current = null; setParams(event.target.value ? { paper: event.target.value } : {}, { replace: true }); selectPaper(event.target.value); }}>
                <option value="">Select a PDF from Library</option>
                {papers.map((paper) => <option key={paper.id} value={paper.id}>{paper.filename}</option>)}
              </select>
              <button type="button" disabled={busy || !paperId} onClick={() => {
                invalidateRequest(); openedFromRouteRef.current = null; setParams({ paper: paperId }, { replace: true });
                setConversationId(null); setMessages([]); setQuestion(""); setAnswerResult(null); setResult(null);
              }}>New conversation</button>
            </div>
            <div className="research-composer-row">
              <button className="research-attach-button" type="button" title="Choose a Library PDF" aria-label="Choose a Library PDF" disabled={busy} onClick={() => { paperSelectRef.current?.focus(); try { paperSelectRef.current?.showPicker?.(); } catch { /* Keyboard focus remains on the selector. */ } }}><WorkspaceIcon name="plus" size={19} /></button>
              <textarea id="research-question" ref={questionRef} value={question} onChange={(event) => { setQuestion(event.target.value); setResult(null); setAnswerResult(null); }}
                onKeyDown={(event) => { if (event.key === "ArrowUp" && !question && messages.at(-1)?.question) {
                  event.preventDefault(); setQuestion(messages.at(-1)!.question); return;
                } if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  if (paperId && question.trim() && !busy && (!canAnswer || retryRemaining === 0)) void submit(canAnswer ? "ask" : "search");
                } }} maxLength={1000} required rows={1} aria-label="Your question" aria-keyshortcuts="Enter ArrowUp" placeholder={paperId ? "Ask a question about this paper…" : "Select a Library PDF to ask a question…"} />
              <button className="research-send-button" type="submit" title={canAnswer ? "Send question" : "Find passages"} aria-label={canAnswer ? "Send question" : "Find passages"} disabled={busy || !paperId || !question.trim() || (canAnswer && retryRemaining > 0)}><WorkspaceIcon name="arrowUp" size={19} /></button>
            </div>
          </form>
        </div>
      </div>
    </main>
  </>;
}
