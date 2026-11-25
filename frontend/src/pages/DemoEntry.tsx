import { useState } from "react";
import { useNavigate } from "react-router-dom";
import AuthLayout from "../components/AuthLayout";
import { enterDemo } from "../services/authApi";

export default function DemoEntry() {
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function start() {
    setBusy(true);
    setError("");
    try {
      const { paper_id } = await enterDemo();
      navigate(`/research?paper=${encodeURIComponent(paper_id)}`, { replace: true });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The demo could not open.");
    } finally {
      setBusy(false);
    }
  }

  return <AuthLayout title="Demo Mode"
    description="Explore a published PLOS ONE article with sample AI answers, PDF citations, and sample external research."
    footer="Your guest workspace is separate from live CiteWise accounts.">
    <div className="auth-form">
      <button className="primary-button" type="button" onClick={() => void start()} disabled={busy}>
        {busy ? "Preparing sample paper…" : "Open demo workspace"}
      </button>
      {error && <p className="oauth-error" role="alert">{error}</p>}
    </div>
  </AuthLayout>;
}
