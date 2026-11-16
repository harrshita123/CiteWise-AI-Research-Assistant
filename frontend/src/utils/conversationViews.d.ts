import type { Conversation } from "../services/researchApi";
import type { Paper } from "../services/papersApi";

export declare function conversationViews(conversations: Conversation[], papers?: Paper[]): {
  saved: Conversation[];
  pinned: Conversation[];
  folders: Map<string, Conversation[]>;
  recents: Conversation[];
};
