import { Link } from "react-router-dom";
import Logo from "../components/Logo";

const steps = [
  ["01", "Upload your papers", "Add the PDFs you want CiteWise to work with."],
  ["02", "Ask a question", "Ask anything about the papers and get an answer grounded in your library."],
  ["03", "Verify the evidence", "Open the cited page and see the passage behind the answer."],
];

export default function LandingPage() {
  return (
    <div>
      <header className="site-header">
        <Logo />
        <nav aria-label="Main navigation">
          <a href="#how-it-works">How it works</a>
          <Link to="/login">Log in</Link>
          <Link className="primary-button small" to="/signup">Get started</Link>
        </nav>
      </header>

      <main>
        <section className="hero">
          <div>
            <p className="label">Your research, with receipts.</p>
            <h1>Ask your papers.<br /><span>Verify every answer.</span></h1>
            <p className="hero-text">
              Upload research papers, ask questions, and jump directly to the page and passage that supports each answer.
            </p>
            <div className="hero-actions">
              <Link className="primary-button" to="/signup">Start researching</Link>
              <a className="secondary-button" href="#how-it-works">See how it works</a>
            </div>
          </div>

          <div className="answer-preview">
            <div className="preview-header">
              <span>Research workspace</span>
              <strong>✓ Verified</strong>
            </div>
            <p className="question">What are the limitations of RAG?</p>
            <div className="answer">
              <strong>CiteWise</strong>
              <p>RAG can introduce errors when retrieved context is incomplete or irrelevant. <b>[1]</b></p>
            </div>
            <div className="source">
              <span>1</span>
              <div>
                <strong>Evidence</strong>
                <p>Lewis et al., 2020 <small>Page 3 · Section 2.2</small></p>
                <mark>Retrieved context may be incomplete or irrelevant.</mark>
                <span className="open-paper">Open in paper →</span>
              </div>
            </div>
          </div>
        </section>

        <section className="research-flow" aria-labelledby="research-flow-title">
          <div className="flow-copy">
            <p className="label">Research should not feel like this</p>
            <p className="friction">40 PDFs <span>→</span> 17 tabs <span>→</span> Ctrl+F <span>→</span> ???</p>
          </div>
          <div className="flow-copy" id="research-flow-title">
            <p className="label">CiteWise makes it simple</p>
            <ol>
              <li>Question</li>
              <li>Answer</li>
              <li>Citation</li>
              <li>Exact passage</li>
            </ol>
          </div>
        </section>

        <section className="how-it-works" id="how-it-works">
          <p className="label">From paper to proof</p>
          <h2>Upload. Ask. Verify.</h2>
          <div className="steps">
            {steps.map(([number, title, text]) => (
              <article key={title}>
                <span>{number}</span>
                <h3>{title}</h3>
                <p>{text}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="evidence-statement">
          <p className="label">CiteWise</p>
          <h2>Answers are useful.<br /><span>Evidence is better.</span></h2>
          <p>Ask better questions, then follow every answer back to the source.</p>
          <Link className="primary-button" to="/signup">Start researching</Link>
        </section>
      </main>

      <footer>
        <Logo />
        <p>Ask better questions. Cite the evidence.</p>
      </footer>
    </div>
  );
}
