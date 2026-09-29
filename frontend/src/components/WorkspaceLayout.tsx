import { ReactNode, useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { logOut, SignedInUser } from "../services/authApi";
import Logo from "./Logo";
import WorkspaceIcon from "./WorkspaceIcon";
import "../workspace.css";

export default function WorkspaceLayout({ user, children }: { user: SignedInUser; children: ReactNode }) {
  const navigate = useNavigate();
  const [logoutError, setLogoutError] = useState("");
  const [signingOut, setSigningOut] = useState(false);
  const initials = (user.name || user.email).trim().slice(0, 1).toUpperCase();

  async function handleLogout() {
    setSigningOut(true);
    setLogoutError("");
    try {
      await logOut();
      navigate("/", { replace: true });
    } catch (error) {
      setLogoutError(error instanceof Error ? error.message : "Could not sign out.");
      setSigningOut(false);
    }
  }

  return (
    <div className="workspace-shell">
      <aside className="workspace-sidebar" aria-label="Workspace navigation">
        <Logo to="/home" />
        <nav className="workspace-nav" aria-label="Main workspace">
          <NavLink to="/home" className={({ isActive }) => isActive ? "active" : ""}><WorkspaceIcon name="home" />Dashboard</NavLink>
          <NavLink to="/papers" className={({ isActive }) => isActive ? "active" : ""}><WorkspaceIcon name="library" />Library</NavLink>
          <NavLink to="/research" className={({ isActive }) => `workspace-nav-soon${isActive ? " active" : ""}`}><WorkspaceIcon name="search" />Research <small>Coming soon</small></NavLink>
          <NavLink to="/compare" className={({ isActive }) => `workspace-nav-soon${isActive ? " active" : ""}`}><WorkspaceIcon name="compare" />Compare <small>Coming soon</small></NavLink>
          <NavLink to="/evaluation" className={({ isActive }) => `workspace-nav-soon${isActive ? " active" : ""}`}><WorkspaceIcon name="shield" />Evaluation <small>Coming soon</small></NavLink>
          {user.role === "admin" && <Link to="/admin"><WorkspaceIcon name="user" />Admin</Link>}
        </nav>
        <div className="workspace-sidebar-footer">
          <div className="workspace-profile"><span className="workspace-avatar">{initials}</span><span><strong>{user.name || user.email}</strong><small>{user.role === "admin" ? "Administrator" : "Research workspace"}</small></span></div>
          <button type="button" onClick={() => void handleLogout()} disabled={signingOut}><WorkspaceIcon name="logout" />{signingOut ? "Signing out..." : "Sign out"}</button>
          {logoutError && <p role="alert">{logoutError}</p>}
        </div>
      </aside>
      <div className="workspace-page">{children}</div>
    </div>
  );
}
