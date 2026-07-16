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

export interface UploadedContextFile {
  name: string;
  path: string;
  chars: number;
  kind: "pdf" | "text";
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

export interface LocalModel {
  id: string;
  label: string;
  size: string;
  description: string;
  active: boolean;
}

export interface ModelState {
  current: string;
  models: LocalModel[];
}

export interface AdminOverview {
  totals: Record<string, number>;
  session_count: number;
  sessions: Array<{
    id: string;
    title: string;
    created_at: string;
    updated_at: string;
    busy: boolean;
    counts: Record<string, number>;
  }>;
  recent_events: Array<{
    session_id: string;
    session_title: string;
    type: string;
    timestamp?: string;
    summary: string;
  }>;
}

export interface ContextDashboard {
  file_count: number;
  chunk_count: number;
  uploaded_count: number;
  auto_context_budget: number;
  chunk_chars: number;
  chunk_overlap: number;
  top_terms: string[];
  largest_files: Array<{ path: string; chars: number; chunks: number; summary: string; top_terms: string[] }>;
  recent_uploads: Array<{ path: string; chars: number; chunks: number; summary: string; top_terms: string[] }>;
}

export interface PreviewState {
  running: boolean;
  managed: boolean;
  port: number;
  url: string;
  path: string;
  message: string;
}

export interface TaskRunStep {
  name: string;
  status: string;
  detail: string;
}

export interface TaskRunResponse {
  goal: string;
  mode: string;
  ok: boolean;
  created_files: string[];
  preview_url: string | null;
  steps: TaskRunStep[];
  notes: string[];
}
