import type { AgentResponse, FileContent, FileNode, SessionInfo } from "../types";

export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "http://localhost:8080";

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
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((res) => handle<T>(res));
}

// ---------------------------------------------------------------- sessions

export function listSessions(): Promise<SessionInfo[]> {
  return fetch(`${API_BASE}/api/sessions`).then((res) => handle<SessionInfo[]>(res));
}

export function createSession(title?: string): Promise<SessionInfo> {
  return post("/api/sessions", { title });
}

export function renameSession(sessionId: string, title: string): Promise<SessionInfo> {
  return fetch(`${API_BASE}/api/sessions/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  }).then((res) => handle<SessionInfo>(res));
}

export function deleteSession(sessionId: string): Promise<{ ok: boolean }> {
  return fetch(`${API_BASE}/api/sessions/${sessionId}`, { method: "DELETE" }).then((res) =>
    handle<{ ok: boolean }>(res),
  );
}

// ------------------------------------------------------------------- agent

export function getAgentState(sessionId: string): Promise<AgentResponse> {
  return fetch(`${API_BASE}/api/sessions/${sessionId}/state`).then((res) =>
    handle<AgentResponse>(res),
  );
}

export function postChat(
  sessionId: string,
  message: string,
  contextFiles: string[] = [],
): Promise<AgentResponse> {
  return post(`/api/sessions/${sessionId}/chat`, { message, context_files: contextFiles });
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
  return fetch(`${API_BASE}/api/files`).then((res) => handle<FileNode>(res));
}

export function getFileContent(path: string): Promise<FileContent> {
  return fetch(`${API_BASE}/api/files/content?path=${encodeURIComponent(path)}`).then((res) =>
    handle<FileContent>(res),
  );
}

export function saveFileContent(path: string, content: string): Promise<FileContent> {
  return fetch(`${API_BASE}/api/files/content`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, content }),
  }).then((res) => handle<FileContent>(res));
}
