export function conversationViews(conversations, papers = []) {
  const recents = [...conversations].sort((a, b) =>
    Date.parse(b.updated_at) - Date.parse(a.updated_at) || a.id.localeCompare(b.id));
  const folders = new Map();
  for (const paper of papers) folders.set(paper.id, []);
  for (const conversation of recents) {
    const items = folders.get(conversation.paper_id) ?? [];
    items.push(conversation);
    folders.set(conversation.paper_id, items);
  }
  return { saved: recents.filter((item) => item.saved === true), pinned: recents.filter((item) => item.pinned === true), folders, recents };
}
