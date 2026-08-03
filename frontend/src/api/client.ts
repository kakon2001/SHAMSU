import type { AgentResponse, FileContent, FileNode, SessionInfo, UploadedContextFile, AdminOverview, ContextDashboard, ModelState, PreviewState, TaskRunResponse, AuthResponse, AuthUser, GitStatus, GitCommit, GitSearchHit, WebSearchResponse, ConnectorMarketplace, VectorRebuildResponse, VectorSearchResponse, VectorStats, ReportSchema, ReportQueryResult, ReportHistoryItem, DependencyScan, DependencyPlan, DependencyInstallResult } from "../types";

export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "http://localhost:8080";

const AUTH_TOKEN_KEY = "shamsu_auth_token";

export function getAuthToken(): string | null {
  return window.localStorage.getItem(AUTH_TOKEN_KEY);
}

export function setAuthToken(token: string | null): void {
  if (token) window.localStorage.setItem(AUTH_TOKEN_KEY, token);
  else window.localStorage.removeItem(AUTH_TOKEN_KEY);
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = getAuthToken();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}
async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Request failed with status ${res.status}`);
  }
  return res.json() as Promise<T>;
}

function post<T>(url: string, body?: unknown): Promise<T> {
  return fetch(`${API_BASE}${url}`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((res) => handle<T>(res));
}


// -------------------------------------------------------------------- auth

export function register(email: string, password: string, name = ""): Promise<AuthResponse> {
  return post("/api/auth/register", { email, password, name });
}

export function login(email: string, password: string): Promise<AuthResponse> {
  return post("/api/auth/login", { email, password });
}

export function getCurrentUser(): Promise<AuthUser> {
  return fetch(`${API_BASE}/api/auth/me`, { headers: authHeaders(), cache: "no-store" }).then((res) => handle<AuthUser>(res));
}

export function logout(): Promise<{ ok: boolean }> {
  return post<{ ok: boolean }>("/api/auth/logout").finally(() => setAuthToken(null));
}

// ---------------------------------------------------------------- sessions

export function listSessions(): Promise<SessionInfo[]> {
  return fetch(`${API_BASE}/api/sessions`, { headers: authHeaders() }).then((res) => handle<SessionInfo[]>(res));
}

export function createSession(title?: string): Promise<SessionInfo> {
  return post("/api/sessions", { title });
}

export function renameSession(sessionId: string, title: string): Promise<SessionInfo> {
  return fetch(`${API_BASE}/api/sessions/${sessionId}`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ title }),
  }).then((res) => handle<SessionInfo>(res));
}

export function deleteSession(sessionId: string): Promise<{ ok: boolean }> {
  return fetch(`${API_BASE}/api/sessions/${sessionId}`, { method: "DELETE", headers: authHeaders() }).then((res) =>
    handle<{ ok: boolean }>(res),
  );
}

// ------------------------------------------------------------------- agent

export function getAgentState(sessionId: string): Promise<AgentResponse> {
  return fetch(`${API_BASE}/api/sessions/${sessionId}/state`, { headers: authHeaders() }).then((res) =>
    handle<AgentResponse>(res),
  );
}

export function postChat(
  sessionId: string,
  message: string,
  contextFiles: string[] = [],
  contextFileLabels: Record<string, string> = {},
): Promise<AgentResponse> {
  return post(`/api/sessions/${sessionId}/chat`, {
    message,
    context_files: contextFiles,
    context_file_labels: contextFileLabels,
  });
}

export function postApproval(
  sessionId: string,
  id: string,
  approved: boolean,
): Promise<AgentResponse> {
  return post(`/api/sessions/${sessionId}/approval`, { id, approved });
}

export function postContinue(sessionId: string): Promise<AgentResponse> {
  return post(`/api/sessions/${sessionId}/continue`);
}

export function postStop(sessionId: string): Promise<AgentResponse> {
  return post(`/api/sessions/${sessionId}/stop`);
}

export function postReset(sessionId: string): Promise<AgentResponse> {
  return post(`/api/sessions/${sessionId}/reset`);
}

// ------------------------------------------------------------------- files

export function getFileTree(): Promise<FileNode> {
  return fetch(`${API_BASE}/api/files`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<FileNode>(res));
}

export function getFileContent(path: string): Promise<FileContent> {
  return fetch(`${API_BASE}/api/files/content?path=${encodeURIComponent(path)}`, { headers: authHeaders() }).then((res) =>
    handle<FileContent>(res),
  );
}

export function saveFileContent(path: string, content: string): Promise<FileContent> {
  return fetch(`${API_BASE}/api/files/content`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ path, content }),
  }).then((res) => handle<FileContent>(res));
}

// ----------------------------------------------------------------- uploads

export function uploadContextFile(file: File): Promise<UploadedContextFile> {
  const body = new FormData();
  body.append("file", file);
  return fetch(`${API_BASE}/api/uploads`, {
    method: "POST",
    headers: authHeaders(),
    body,
  }).then((res) => handle<UploadedContextFile>(res));
}



// ------------------------------------------------------------------ models

export function getModels(): Promise<ModelState> {
  return fetch(`${API_BASE}/api/models`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<ModelState>(res));
}

export function setCurrentModel(modelId: string): Promise<ModelState> {
  return post("/api/models/current", { model_id: modelId });
}

// ------------------------------------------------------------------- admin

export function getAdminOverview(): Promise<AdminOverview> {
  return fetch(`${API_BASE}/api/admin/overview`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<AdminOverview>(res));
}

export function getContextDashboard(): Promise<ContextDashboard> {
  return fetch(`${API_BASE}/api/context/dashboard`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<ContextDashboard>(res));
}


export function getVectorStats(): Promise<VectorStats> {
  return fetch(`${API_BASE}/api/context/vector/stats`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<VectorStats>(res));
}

export function rebuildVectorIndex(limitFiles = 500): Promise<VectorRebuildResponse> {
  return post(`/api/context/vector/rebuild`, { limit_files: limitFiles });
}

export function searchVectorContext(query: string, limit = 8): Promise<VectorSearchResponse> {
  const params = new URLSearchParams({ query, limit: String(limit) });
  return fetch(`${API_BASE}/api/context/vector/search?${params.toString()}`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<VectorSearchResponse>(res));
}





// ------------------------------------------------------------- dependencies

export function scanDependencies(): Promise<DependencyScan> {
  return fetch(`${API_BASE}/api/dependencies/scan`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<DependencyScan>(res));
}

export function planDependencies(prompt: string, target = "auto"): Promise<DependencyPlan> {
  return post("/api/dependencies/plan", { prompt, target });
}

export function installDependencies(manager: string, packages: string[], projectPath: string, approve = false, runVerification = false): Promise<DependencyInstallResult> {
  return post("/api/dependencies/install", { manager, packages, project_path: projectPath, approve, update_manifest: true, run_verification: runVerification });
}
// ------------------------------------------------------------------ reports

export function getReportSchema(): Promise<ReportSchema> {
  return fetch(`${API_BASE}/api/reports/schema`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<ReportSchema>(res));
}

export function runReportQuery(sql: string, title = "", limit = 100, save = true): Promise<ReportQueryResult> {
  return post("/api/reports/query", { sql, title, limit, save });
}

export function getReportHistory(limit = 10): Promise<{ reports: ReportHistoryItem[] }> {
  const params = new URLSearchParams({ limit: String(limit) });
  return fetch(`${API_BASE}/api/reports/history?${params.toString()}`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<{ reports: ReportHistoryItem[] }>(res));
}
// --------------------------------------------------------------- connectors

export function getConnectors(): Promise<ConnectorMarketplace> {
  return fetch(`${API_BASE}/api/connectors`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<ConnectorMarketplace>(res));
}
// ---------------------------------------------------------------------- web

export function searchWeb(query: string, limit = 5): Promise<WebSearchResponse> {
  const params = new URLSearchParams({ query, limit: String(limit) });
  return fetch(`${API_BASE}/api/web/search?${params.toString()}`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<WebSearchResponse>(res));
}
// ---------------------------------------------------------------------- git

export function getGitStatus(): Promise<GitStatus> {
  return fetch(`${API_BASE}/api/git/status`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<GitStatus>(res));
}

export function getGitLog(limit = 8, query = ""): Promise<{ commits: GitCommit[] }> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (query.trim()) params.set("query", query.trim());
  return fetch(`${API_BASE}/api/git/log?${params.toString()}`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<{ commits: GitCommit[] }>(res));
}

export function getGitSearch(query: string, limit = 12): Promise<{ query: string; hits: GitSearchHit[] }> {
  const params = new URLSearchParams({ query, limit: String(limit) });
  return fetch(`${API_BASE}/api/git/search?${params.toString()}`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<{ query: string; hits: GitSearchHit[] }>(res));
}
// ----------------------------------------------------------------- preview

export function getPreviewStatus(path = "", port = 9000): Promise<PreviewState> {
  const params = new URLSearchParams({ path, port: String(port) });
  return fetch(`${API_BASE}/api/preview/status?${params.toString()}`, { cache: "no-store", headers: authHeaders() }).then((res) => handle<PreviewState>(res));
}

export function startPreviewServer(path = "", port = 9000): Promise<PreviewState> {
  return post("/api/preview/start", { path, port });
}

export function stopPreviewServer(): Promise<PreviewState> {
  return post("/api/preview/stop");
}

// -------------------------------------------------------------------- tasks

export function runTaskBuild(prompt: string, preview = true): Promise<TaskRunResponse> {
  return post("/api/tasks/run", { prompt, preview, overwrite: true });
}






