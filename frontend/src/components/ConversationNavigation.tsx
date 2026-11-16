import { useEffect, useRef, useState, type Dispatch, type MouseEvent, type SetStateAction } from "react";
import { createPortal } from "react-dom";
import { useLocation, useNavigate } from "react-router-dom";
import { Conversation, deleteConversation, updateConversation } from "../services/researchApi";
import type { Paper } from "../services/papersApi";
import { conversationViews } from "../utils/conversationViews";
import WorkspaceIcon from "./WorkspaceIcon";

type Props = {
  conversations: Conversation[];
  papers: Paper[];
  setConversations: Dispatch<SetStateAction<Conversation[]>>;
  onOpen: () => void;
};

export default function ConversationNavigation({ conversations, papers, setConversations, onOpen }: Props) {
  const navigate = useNavigate();
  const location = useLocation();
  const [openFolders, setOpenFolders] = useState<Set<string>>(new Set());
  const [openMenu, setOpenMenu] = useState<{ id: string; view: string; top: number; left: number } | null>(null);
  const [renaming, setRenaming] = useState<{ id: string; view: string; title: string } | null>(null);
  const [deleting, setDeleting] = useState<Conversation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const menuRef = useRef<HTMLDivElement>(null);
  const menuTriggerRef = useRef<HTMLButtonElement>(null);
  const deleteCancelRef = useRef<HTMLButtonElement>(null);
  const selectedId = location.pathname === "/research" ? new URLSearchParams(location.search).get("conversation") : null;
  const { saved, pinned, folders, recents } = conversationViews(conversations, papers);
  const menuConversation = openMenu ? conversations.find((item) => item.id === openMenu.id) : null;

  useEffect(() => {
    if (!openMenu) return;
    menuRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const closeOutside = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)
          && !menuTriggerRef.current?.contains(event.target as Node)) setOpenMenu(null);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setOpenMenu(null); menuTriggerRef.current?.focus(); }
    };
    const closeOnMove = () => setOpenMenu(null);
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    window.addEventListener("scroll", closeOnMove, true);
    window.addEventListener("resize", closeOnMove);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", closeOnEscape);
      window.removeEventListener("scroll", closeOnMove, true);
      window.removeEventListener("resize", closeOnMove);
    };
  }, [openMenu]);

  useEffect(() => { if (deleting) deleteCancelRef.current?.focus(); }, [deleting]);

  async function changeConversation(target: Conversation, update: { title?: string; pinned?: boolean; saved?: boolean }) {
    if (busy) return false;
    setBusy(true);
    setError("");
    try {
      const changed = await updateConversation(target.id, update);
      setConversations((items) => items.map((item) => item.id === changed.id ? changed : item));
      return true;
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Conversation could not be updated.");
      return false;
    } finally { setBusy(false); }
  }

  async function removeConversation(targetId: string) {
    if (busy) return false;
    setBusy(true);
    setError("");
    try {
      await deleteConversation(targetId);
      setConversations((items) => items.filter((item) => item.id !== targetId));
      if (selectedId === targetId) navigate("/research", { replace: true });
      return true;
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Conversation could not be deleted.");
      return false;
    } finally { setBusy(false); }
  }

  function toggleMenu(event: MouseEvent<HTMLButtonElement>, conversation: Conversation, view: string) {
    event.stopPropagation();
    if (openMenu?.id === conversation.id && openMenu.view === view) { setOpenMenu(null); return; }
    const rect = event.currentTarget.getBoundingClientRect();
    const width = 164;
    const height = 160;
    menuTriggerRef.current = event.currentTarget;
    setOpenMenu({ id: conversation.id, view,
      left: Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8)),
      top: rect.bottom + height + 4 > window.innerHeight ? Math.max(8, rect.top - height - 4) : rect.bottom + 4 });
  }

  function row(conversation: Conversation, view: string) {
    const filename = conversation.paper_filename ?? "Paper unavailable";
    return <li key={conversation.id} className={selectedId === conversation.id ? "selected" : ""}>
      {renaming?.id === conversation.id && renaming.view === view ?
        <form className="research-history-rename" onSubmit={(event) => { event.preventDefault();
          const title = renaming.title.trim();
          if (title && title !== conversation.title) void changeConversation(conversation, { title }).then((ok) => { if (ok) setRenaming(null); });
          else if (title) setRenaming(null);
        }}>
          <input aria-label="Conversation title" autoFocus maxLength={120} value={renaming.title}
            onChange={(event) => setRenaming({ ...renaming, title: event.target.value })} />
          <button type="submit" disabled={busy || !renaming.title.trim()}>Save</button>
          <button type="button" onClick={() => setRenaming(null)}>Cancel</button>
        </form> : <>
          <button type="button" disabled={busy} onClick={() => { navigate(`/research?conversation=${encodeURIComponent(conversation.id)}`); onOpen(); }}>
            <strong title={conversation.title}>{conversation.title}</strong>
            <small title={filename}>{filename}</small>
          </button>
          <button className="research-history-menu-trigger" type="button" disabled={busy}
            aria-label={`More options for ${conversation.title}`} aria-haspopup="menu"
            aria-expanded={openMenu?.id === conversation.id && openMenu.view === view}
            onClick={(event) => toggleMenu(event, conversation, view)}>
            <WorkspaceIcon name="more" size={17} />
          </button>
        </>}
    </li>;
  }

  return <>
    <aside className="research-history" aria-label="Conversations">
      <div className="research-history-scroll">
        {error && <p role="alert" className="research-history-error">{error}</p>}
        {folders.size > 0 && <section aria-label="PDF folders" className="research-history-section">
          <h2>PDF FOLDERS</h2>
          {[...folders].map(([paperId, items]) => <div className="research-history-folder" key={paperId}>
            <button type="button" className="research-history-folder-toggle" aria-expanded={openFolders.has(paperId)}
              onClick={() => setOpenFolders((current) => { const next = new Set(current); if (next.has(paperId)) next.delete(paperId); else next.add(paperId); return next; })}>
              <span aria-hidden="true">{openFolders.has(paperId) ? "▾" : "▸"}</span>
              <span title={papers.find((paper) => paper.id === paperId)?.filename ?? items[0]?.paper_filename ?? "Paper unavailable"}>
                {papers.find((paper) => paper.id === paperId)?.filename ?? items[0]?.paper_filename ?? "Paper unavailable"}
              </span>
              <small>{items.length}</small>
            </button>
            {openFolders.has(paperId) && <ul>{items.map((item) => row(item, `folder:${paperId}`))}</ul>}
          </div>)}
        </section>}
        {pinned.length > 0 && <section aria-label="Pinned chats" className="research-history-section">
          <h2>PINNED CHATS</h2><ul>{pinned.map((item) => row(item, "pinned"))}</ul>
        </section>}
        {saved.length > 0 && <section aria-label="Saved chats" className="research-history-section">
          <h2>SAVED CHATS</h2><ul>{saved.map((item) => row(item, "saved"))}</ul>
        </section>}
        {recents.length > 0 && <section aria-label="Recents" className="research-history-section">
          <h2>RECENTS</h2><ul>{recents.map((item) => row(item, "recents"))}</ul>
        </section>}
      </div>
    </aside>
    {openMenu && menuConversation && createPortal(<div ref={menuRef} className="research-conversation-menu" role="menu"
      aria-label={`Options for ${menuConversation.title}`} style={{ top: openMenu.top, left: openMenu.left }}
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
        event.preventDefault();
        const items = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('button[role="menuitem"]')];
        const current = items.indexOf(document.activeElement as HTMLButtonElement);
        items[(current + (event.key === "ArrowDown" ? 1 : items.length - 1)) % items.length]?.focus();
      }}>
      <button type="button" role="menuitem" onClick={() => { setRenaming({ id: menuConversation.id, view: openMenu.view, title: menuConversation.title }); setOpenMenu(null); }}>
        <WorkspaceIcon name="edit" size={16} />Rename</button>
      <button type="button" role="menuitem" onClick={() => { setOpenMenu(null); void changeConversation(menuConversation, { pinned: !menuConversation.pinned }); }}>
        <WorkspaceIcon name="pin" size={16} />{menuConversation.pinned ? "Unpin" : "Pin"}</button>
      <button type="button" role="menuitem" onClick={() => { setOpenMenu(null); void changeConversation(menuConversation, { saved: !menuConversation.saved }); }}>
        <WorkspaceIcon name="library" size={16} />{menuConversation.saved ? "Unsave" : "Save"}</button>
      <button className="research-conversation-menu-delete" type="button" role="menuitem"
        onClick={() => { setDeleting(menuConversation); setOpenMenu(null); }}>
        <WorkspaceIcon name="trash" size={16} />Delete</button>
    </div>, document.body)}
    {deleting && createPortal(<div className="research-delete-backdrop" onClick={() => setDeleting(null)}>
      <div className="research-delete-dialog" role="alertdialog" aria-modal="true"
        aria-labelledby="research-delete-title" aria-describedby="research-delete-description"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === "Escape") setDeleting(null);
          if (event.key === "Tab") {
            const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>("button")];
            if (event.shiftKey && document.activeElement === buttons[0]) { event.preventDefault(); buttons.at(-1)?.focus(); }
            else if (!event.shiftKey && document.activeElement === buttons.at(-1)) { event.preventDefault(); buttons[0]?.focus(); }
          }
        }}>
        <h2 id="research-delete-title">Delete conversation?</h2>
        <p id="research-delete-description">This removes “{deleting.title}” from your chats. The PDF and other conversations stay available.</p>
        <div><button ref={deleteCancelRef} type="button" onClick={() => setDeleting(null)}>Cancel</button>
          <button className="research-delete-confirm" type="button" disabled={busy}
            onClick={() => { void removeConversation(deleting.id).then((ok) => { if (ok) setDeleting(null); }); }}>Delete</button></div>
      </div>
    </div>, document.body)}
  </>;
}
