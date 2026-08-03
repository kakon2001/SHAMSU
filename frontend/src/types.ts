export interface AuthUser {
  id: string;
  email: string;
  name: string;
}

export interface AuthResponse {
  token: string;
  user: AuthUser;
}
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
  extension?: string;
  summary?: string;
  preview?: string;
}

/** Events recorded by the backend agent session. */
export type AgentEvent =
  | { type: "user_message"; content: string; context_files?: string[]; context_file_labels?: Record<string, string>; timestamp?: string }
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
  | { kind: "user"; id: string; content: string; contextFiles?: string[]; contextFileLabels?: Record<string, string> }
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


export interface ConnectorInfo {
  id: string;
  name: string;
  category: string;
  status: "enabled" | "planned" | "disabled";
  privacy: string;
  description: string;
  capabilities: string[];
  tools: string[];
  setup_required: boolean;
}

export interface ConnectorMarketplace {
  connectors: ConnectorInfo[];
  enabled_count: number;
  planned_count: number;
}
export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  source: string;
}

export interface WebSearchResponse {
  ok: boolean;
  query: string;
  provider: string;
  results: WebSearchResult[];
  error?: string;
  notes?: string[];
}
export interface GitFileStatus {
  code: string;
  path: string;
}

export interface GitStatus {
  ok: boolean;
  repo_root: string;
  branch: string;
  remote: string;
  clean: boolean;
  ahead: number;
  behind: number;
  files: GitFileStatus[];
}

export interface GitCommit {
  hash: string;
  author: string;
  date: string;
  subject: string;
}

export interface GitSearchHit {
  kind: string;
  path?: string | null;
  line?: number | null;
  commit?: string | null;
  text: string;
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


export interface VectorStats {
  file_count: number;
  chunk_count: number;
  dims: number;
  db_path: string;
  ready: boolean;
  updated_at?: number;
}

export interface VectorMatch {
  path: string;
  chunk_index: number;
  start_line: number;
  end_line: number;
  score: number;
  preview: string;
}

export interface VectorSearchResponse {
  query: string;
  matches: VectorMatch[];
}

export interface VectorRebuildResponse extends VectorStats {
  ok: boolean;
  indexed_files: number;
  indexed_chunks: number;
}


export interface ReportColumn {
  name: string;
  type: string;
  not_null: boolean;
  primary_key: boolean;
}

export interface ReportTableSchema {
  name: string;
  row_count: number;
  columns: ReportColumn[];
}

export interface ReportSchema {
  database: string;
  tables: ReportTableSchema[];
}

export interface ReportQueryResult {
  ok: boolean;
  title: string;
  sql: string;
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  limit: number;
  elapsed_ms: number;
  saved: boolean;
  report_id?: string;
}

export interface ReportHistoryItem {
  id: string;
  title: string;
  sql: string;
  row_count: number;
  columns: string[];
  preview: Record<string, unknown>[];
  created_at: number;
}
export interface PreviewState {
  running: boolean;
  managed: boolean;
  port: number;
  url: string;
  path: string;
  message: string;
}


export interface ReliabilityReport {
  phases: string[];
  verification_feedback: string[];
  repair_attempts: number;
  final_status: string;
  next_action: string;
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
  requirements_analysis?: string[];
  clarification_questions?: string[];
  stack?: string[];
  file_plan?: string[];
  workflow_summary?: string;
  reliability?: ReliabilityReport | null;
}





export interface DependencyProject {
  path: string;
  absolute_path: string;
  managers: string[];
  dependencies: Record<string, string[]>;
  scripts: Record<string, string>;
}

export interface DependencyScan {
  root: string;
  projects: DependencyProject[];
  summary: string;
}

export interface DependencySuggestion {
  manager: "npm" | "pip";
  package: string;
  reason: string;
  project_path: string;
  already_installed: string;
}

export interface DependencyPlan {
  prompt: string;
  target: string;
  detected: DependencyScan;
  suggestions: DependencySuggestion[];
  commands: string[];
  workflow: string[];
}

export interface DependencyInstallResult {
  approved: boolean;
  ran: boolean;
  ok?: boolean;
  manager: string;
  packages: string[];
  project_path: string;
  command: string;
  message?: string;
  exit_code?: number;
  output?: string;
  elapsed_ms?: number;
  manifest_updates: string[];
  verification: Array<{ command: string; ok: boolean; exit_code: number | null; output: string }>;
}
