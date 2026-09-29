import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import Logo from "./Logo";

type Props = {
  title: string;
  description: string;
  footer: ReactNode;
  children: ReactNode;
};

export default function AuthLayout({ title, description, footer, children }: Props) {
  return (
    <main className="auth-page">
      <aside className="auth-intro">
        <Logo />
        <div>
          <p className="label">CiteWise research</p>
          <h2>Don't just get an answer. See where it came from.</h2>
          <p>Ask questions about your papers and follow every citation back to the evidence.</p>
        </div>
        <blockquote>Your papers. Your questions. Your evidence.</blockquote>
      </aside>

      <section className="auth-panel">
        <div className="mobile-logo"><Logo /></div>
        <div className="auth-card">
          <p className="label">Welcome to CiteWise</p>
          <h1>{title}</h1>
          <p className="auth-description">{description}</p>
          {children}
          <p className="auth-footer">{footer}</p>
        </div>
        <Link className="back-link" to="/">← Back to home</Link>
      </section>
    </main>
  );
}
