export interface FileNode {
  name: string;
  path: string;
  type: "file" | "dir";
  children?: FileNode[] | null;
}

export interface FileContent {
  path: string;
  content: string;
}

/** Events recorded by the backend agent session. */
export type AgentEvent =
  | { type: "user_message"; content: string; context_files?: string[]; timestamp?: string }
  | { type: "assistant_delta"; id: string; content: string; timestamp?: string }
  | { type: "assistant_message"; id?: string; content: string; timestamp?: string }
  | { type: "tool_call"; id: string; name: string; args: Record<string, unknown>; timestamp?: string }
  | { type: "tool_result"; id: string; name: string; ok: boolean; preview: string; timestamp?: string }
  | {
      type: "approval_request";
      id: string;
      name: "run_shell" | "write_file";
      command?: string;
      path?: string;
      diff?: string;
      is_new_file?: boolean;
      risk?: string;
      risk_reason?: string;
      timestamp?: string;
    }
  | { type: "approval_resolved"; id: string; approved: boolean; timestamp?: string }
  | { type: "files_changed"; paths: string[]; timestamp?: string }
  | { type: "turn_end"; timestamp?: string }
  | { type: "error"; message: string; timestamp?: string };

export interface AgentResponse {
  events: AgentEvent[];
  busy: boolean;
}

/** A stored chat session (persisted in MySQL on the backend). */
export interface SessionInfo {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  busy: boolean;
}

/** Items rendered in the chat transcript. */
export type ChatItem =
  | { kind: "user"; id: string; content: string; contextFiles?: string[] }
  | { kind: "assistant"; id: string; content: string }
  | {
      kind: "tool";
      id: string;
      name: string;
      args: Record<string, unknown>;
      status: "running" | "done" | "error";
      preview?: string;
    }
  | {
      kind: "approval";
      id: string;
      name: "run_shell" | "write_file";
      command?: string;
      path?: string;
      diff?: string;
      isNewFile?: boolean;
      risk?: string;
      riskReason?: string;
      status: "pending" | "approved" | "rejected";
    }
  | { kind: "error"; id: string; content: string };

export interface EditorTab {
  path: string;
  content: string;
  savedContent: string;
}
