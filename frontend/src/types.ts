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
  | { type: "user_message"; content: string; context_files?: string[] }
  | { type: "assistant_message"; content: string }
  | { type: "tool_call"; id: string; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; id: string; name: string; ok: boolean; preview: string }
  | {
      type: "approval_request";
      id: string;
      name: "run_shell" | "write_file";
      command?: string;
      path?: string;
      diff?: string;
      is_new_file?: boolean;
    }
  | { type: "approval_resolved"; id: string; approved: boolean }
  | { type: "files_changed"; paths: string[] }
  | { type: "turn_end" }
  | { type: "error"; message: string };

export interface AgentResponse {
  events: AgentEvent[];
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
      status: "pending" | "approved" | "rejected";
    }
  | { kind: "error"; id: string; content: string };

export interface EditorTab {
  path: string;
  content: string;
  savedContent: string;
}
