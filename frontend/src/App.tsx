import { ReactNode, useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import AdminPage from "./pages/AdminPage";
import LandingPage from "./pages/LandingPage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import PapersPage from "./pages/PapersPage";
import HomePage from "./pages/HomePage";
import ResearchPage from "./pages/ResearchPage";
import DemoEntry from "./pages/DemoEntry";
import WorkspaceLayout from "./components/WorkspaceLayout";
import { AuthApiError, getCurrentUser, getDemoMode, SignedInUser } from "./services/authApi";

function GuestOnly({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<SignedInUser | null>(null);
  const [status, setStatus] = useState<"checking" | "guest" | "error">("checking");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    getCurrentUser().then(
      (currentUser) => {
        if (active) setUser(currentUser);
      },
      (error: unknown) => {
        if (active) setStatus(error instanceof AuthApiError && error.status === 401 ? "guest" : "error");
      },
    );
    return () => { active = false; };
  }, [attempt]);

  if (user) {
    return <Navigate to="/home" replace />;
  }
  if (status === "checking") {
    return <main className="session-check" role="status">Checking your session...</main>;
  }
  if (status === "error") {
    return (
      <main className="session-check">
        <p>We could not check your session. Please make sure the backend is running.</p>
        <button className="primary-button" onClick={() => { setStatus("checking"); setAttempt((value) => value + 1); }}>
          Try again
        </button>
      </main>
    );
  }
  return <>{children}</>;
}

function WorkspaceRoute({ demoMode }: { demoMode: boolean }) {
  const [user, setUser] = useState<SignedInUser | null>(null);
  const [status, setStatus] = useState<"checking" | "error" | "expired">("checking");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    getCurrentUser().then(
      (currentUser) => { if (active) setUser(currentUser); },
      (error: unknown) => {
        if (active) setStatus(error instanceof AuthApiError && error.status === 401 ? "expired" : "error");
      },
    );
    return () => { active = false; };
  }, [attempt]);

  if (status === "expired") return <Navigate to="/login" replace />;
  if (user) return <WorkspaceLayout user={user} demoMode={demoMode} />;
  if (status === "error") return <main className="session-check">
    <p>We could not load your workspace. Please make sure the backend is running.</p>
    <button className="primary-button" onClick={() => { setStatus("checking"); setAttempt((value) => value + 1); }}>Try again</button>
  </main>;
  return <main className="session-check" role="status">Loading your workspace...</main>;
}

export default function App() {
  const [demoMode, setDemoMode] = useState<boolean | "checking" | "error">("checking");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    getDemoMode().then((enabled) => { if (active) setDemoMode(enabled); },
      () => { if (active) setDemoMode("error"); });
    return () => { active = false; };
  }, [attempt]);
  if (demoMode === "checking") return <main className="session-check" role="status">Loading CiteWise…</main>;
  if (demoMode === "error") return <main className="session-check">
    <p>We could not connect to CiteWise.</p>
    <button className="primary-button" onClick={() => { setDemoMode("checking"); setAttempt((value) => value + 1); }}>
      Try again
    </button>
  </main>;
  return (
    <Routes>
      <Route path="/" element={demoMode ? <GuestOnly><DemoEntry /></GuestOnly> : <LandingPage />} />
      <Route path="/login" element={<GuestOnly>{demoMode ? <DemoEntry /> : <LoginPage />}</GuestOnly>} />
      <Route path="/signup" element={<GuestOnly>{demoMode ? <DemoEntry /> : <SignupPage />}</GuestOnly>} />
      <Route element={<WorkspaceRoute demoMode={demoMode} />}>
        <Route path="/home" element={<HomePage />} />
        <Route path="/research" element={<ResearchPage />} />
        <Route path="/compare" element={<Navigate to="/home" replace />} />
        <Route path="/evaluation" element={<Navigate to="/research" replace />} />
        <Route path="/admin" element={<AdminPage />} />
        <Route path="/papers" element={<PapersPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
