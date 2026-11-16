import { useCallback, useEffect, useRef, useState, type CSSProperties, type Dispatch, type PointerEvent, type SetStateAction } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { logOut, SignedInUser } from "../services/authApi";
import WorkspaceIcon from "./WorkspaceIcon";
import { Conversation, listConversations } from "../services/researchApi";
import { listPapers, type Paper } from "../services/papersApi";
import ConversationNavigation from "./ConversationNavigation";
import { clampSidebarWidth, MAX_SIDEBAR_WIDTH, MIN_SIDEBAR_WIDTH, resizedSidebarWidth } from "../utils/sidebarWidth";
import ThemeToggle from "./ThemeToggle";

export type ConversationState = {
  conversations: Conversation[];
  setConversations: Dispatch<SetStateAction<Conversation[]>>;
  refreshConversations: () => Promise<void>;
};

export default function WorkspaceLayout({ user }: { user: SignedInUser }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [logoutError, setLogoutError] = useState("");
  const [signingOut, setSigningOut] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try { return localStorage.getItem(`citewise-sidebar-collapsed:${user.id}`) === "true"; }
    catch { return false; }
  });
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    try { return clampSidebarWidth(localStorage.getItem(`citewise-sidebar-width:${user.id}`)); }
    catch { return clampSidebarWidth(undefined); }
  });
  const [resizingSidebar, setResizingSidebar] = useState(false);
  const resizeOriginRef = useRef<{ x: number; width: number } | null>(null);
  const previousUserSelectRef = useRef("");
  const previousCursorRef = useRef("");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const conversationVersionRef = useRef(0);
  const [papers, setPapers] = useState<Paper[]>([]);
  const sidebarExpanded = !sidebarCollapsed || drawerOpen;
  const initials = (user.name || user.email).trim().slice(0, 1).toUpperCase();
  const updateConversations = useCallback<Dispatch<SetStateAction<Conversation[]>>>((update) => {
    conversationVersionRef.current += 1;
    setConversations(update);
  }, []);
  const refreshConversations = useCallback(async () => {
    const version = conversationVersionRef.current;
    const items = await listConversations();
    if (version === conversationVersionRef.current) {
      conversationVersionRef.current += 1;
      setConversations(items);
    }
  }, []);
  const toggleSidebar = () => {
    if (window.matchMedia("(max-width: 900px)").matches) setDrawerOpen((current) => !current);
    else setSidebarCollapsed((current) => !current);
  };

  useEffect(() => { setDrawerOpen(false); }, [location.pathname]);
  useEffect(() => {
    let active = true;
    const version = conversationVersionRef.current;
    void listConversations().then((items) => {
      if (active && version === conversationVersionRef.current) {
        conversationVersionRef.current += 1;
        setConversations(items);
      }
    }).catch(() => {});
    void listPapers(0, 100, "ready").then((items) => { if (active) setPapers(items); }).catch(() => {});
    return () => { active = false; };
  }, [user.id]);
  useEffect(() => {
    try { localStorage.setItem(`citewise-sidebar-collapsed:${user.id}`, String(sidebarCollapsed)); }
    catch { /* Sidebar remains usable without browser storage. */ }
  }, [sidebarCollapsed, user.id]);
  useEffect(() => {
    try { localStorage.setItem(`citewise-sidebar-width:${user.id}`, String(sidebarWidth)); }
    catch { /* Resizing remains usable without browser storage. */ }
  }, [sidebarWidth, user.id]);
  useEffect(() => () => {
    document.body.style.userSelect = previousUserSelectRef.current;
    document.body.style.cursor = previousCursorRef.current;
  }, []);

  function beginResize(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    resizeOriginRef.current = { x: event.clientX, width: sidebarWidth };
    previousUserSelectRef.current = document.body.style.userSelect;
    previousCursorRef.current = document.body.style.cursor;
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setResizingSidebar(true);
  }

  function moveResize(event: PointerEvent<HTMLDivElement>) {
    const origin = resizeOriginRef.current;
    if (origin === null) return;
    setSidebarWidth(resizedSidebarWidth(origin.width, origin.x, event.clientX));
  }

  function endResize(event: PointerEvent<HTMLDivElement>) {
    if (resizeOriginRef.current === null) return;
    resizeOriginRef.current = null;
    document.body.style.userSelect = previousUserSelectRef.current;
    document.body.style.cursor = previousCursorRef.current;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setResizingSidebar(false);
  }
  useEffect(() => {
    if (!drawerOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [drawerOpen]);

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
    <div className={`workspace-shell${sidebarCollapsed ? " sidebar-collapsed" : ""}${drawerOpen ? " drawer-open" : ""}${resizingSidebar ? " sidebar-resizing" : ""}`}
      style={{ "--sidebar-width": `${sidebarWidth}px` } as CSSProperties}>
      <button className="workspace-toggle workspace-mobile-toggle" type="button"
        aria-label={drawerOpen ? "Close navigation" : "Open navigation"}
        aria-controls="workspace-sidebar" aria-expanded={drawerOpen}
        onClick={toggleSidebar}>
        <WorkspaceIcon name={drawerOpen ? "close" : "menu"} size={20} />
      </button>
      {drawerOpen && <button className="workspace-drawer-backdrop" type="button"
        aria-label="Close navigation" onClick={() => setDrawerOpen(false)} />}
      <aside id="workspace-sidebar" className="workspace-sidebar" aria-label="Workspace navigation">
        <div className="workspace-sidebar-header">
          {sidebarExpanded ? <>
            <svg className="workspace-sidebar-book" viewBox="0 0 36 36" aria-hidden="true">
              <path d="M4 8c5 0 9 1.5 14 5v18c-5-3.5-9-5-14-5V8Z" />
              <path d="M32 8c-5 0-9 1.5-14 5v18c5-3.5 9-5 14-5V8Z" />
            </svg>
            <span className="workspace-sidebar-brand">CiteWise</span>
            <Link to="/research#research-question" className="workspace-toggle workspace-sidebar-search"
              title="Search your papers" aria-label="Search your papers"
              onClick={() => { setDrawerOpen(false); document.getElementById("research-question")?.focus(); }}>
              <WorkspaceIcon name="search" size={19} />
            </Link>
            <button className="workspace-toggle workspace-sidebar-toggle" type="button"
              aria-label={drawerOpen ? "Close navigation" : "Collapse navigation"}
              aria-controls="workspace-sidebar" aria-expanded={true}
              onClick={toggleSidebar}><WorkspaceIcon name="panel" size={20} /></button>
          </> :
            <button className="workspace-toggle workspace-sidebar-logo" type="button"
              aria-label="Expand navigation" aria-controls="workspace-sidebar" aria-expanded={false}
              onClick={toggleSidebar}>
              <svg viewBox="0 0 36 36" aria-hidden="true">
                <path d="M4 8c5 0 9 1.5 14 5v18c-5-3.5-9-5-14-5V8Z" />
                <path d="M32 8c-5 0-9 1.5-14 5v18c5-3.5 9-5 14-5V8Z" />
              </svg>
            </button>}
        </div>
        <nav className="workspace-nav" aria-label="Main workspace" onClick={() => setDrawerOpen(false)}>
          <NavLink to="/home" title="Dashboard" aria-label="Dashboard" className={({ isActive }) => isActive ? "active" : ""}><WorkspaceIcon name="home" /><span className="workspace-nav-label">Dashboard</span></NavLink>
          <NavLink to="/papers" title="Library" aria-label="Library" className={({ isActive }) => isActive ? "active" : ""}><WorkspaceIcon name="library" /><span className="workspace-nav-label">Library</span></NavLink>
          <NavLink to="/research" title="Research" aria-label="Research" className={({ isActive }) => isActive ? "active" : ""}><WorkspaceIcon name="search" /><span className="workspace-nav-label">Research</span></NavLink>
          <NavLink to="/compare" title="Compare" aria-label="Compare" className={({ isActive }) => `workspace-nav-soon${isActive ? " active" : ""}`}><WorkspaceIcon name="compare" /><span className="workspace-nav-label">Compare</span><small>Coming soon</small></NavLink>
          {user.role === "admin" && <NavLink to="/admin" title="Admin" aria-label="Admin" className={({ isActive }) => isActive ? "active" : ""}><WorkspaceIcon name="user" /><span className="workspace-nav-label">Admin</span></NavLink>}
          {user.role === "admin" && <NavLink to="/admin/evaluation" title="Retrieval evaluation" aria-label="Retrieval evaluation" className={({ isActive }) => isActive ? "active" : ""}><WorkspaceIcon name="shield" /><span className="workspace-nav-label">Retrieval evaluation</span></NavLink>}
        </nav>
        <div className="workspace-conversation-navigation">
          <ConversationNavigation conversations={conversations} setConversations={updateConversations} papers={papers}
            onOpen={() => setDrawerOpen(false)} />
        </div>
        <div className="workspace-sidebar-footer">
          <div className="workspace-account">
            <div className="workspace-profile">
              <span className="workspace-avatar">{initials}</span><span className="workspace-profile-text"><strong>{user.name || user.email}</strong><small>{user.role === "admin" ? "Administrator" : "Research workspace"}</small></span>
              <ThemeToggle className="workspace-theme-toggle" />
            </div>
          </div>
          <button type="button" title="Sign out" aria-label="Sign out" onClick={() => void handleLogout()} disabled={signingOut}><WorkspaceIcon name="logout" /><span className="workspace-nav-label">{signingOut ? "Signing out..." : "Sign out"}</span></button>
          {logoutError && <p role="alert">{logoutError}</p>}
        </div>
      </aside>
      {!sidebarCollapsed && <div className="workspace-sidebar-resize" role="separator" aria-label="Resize sidebar"
        aria-orientation="vertical" aria-valuemin={MIN_SIDEBAR_WIDTH} aria-valuemax={MAX_SIDEBAR_WIDTH}
        aria-valuenow={sidebarWidth} tabIndex={0} onPointerDown={beginResize} onPointerMove={moveResize}
        onPointerUp={endResize} onPointerCancel={endResize} onLostPointerCapture={endResize}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") setSidebarWidth((width) => clampSidebarWidth(width - 10));
          else if (event.key === "ArrowRight") setSidebarWidth((width) => clampSidebarWidth(width + 10));
          else if (event.key === "Home") setSidebarWidth(MIN_SIDEBAR_WIDTH);
          else if (event.key === "End") setSidebarWidth(MAX_SIDEBAR_WIDTH);
          else return;
          event.preventDefault();
        }} />}
      <div className="workspace-page">
        <Outlet context={{ conversations, setConversations: updateConversations, refreshConversations } satisfies ConversationState} />
      </div>
    </div>
  );
}
