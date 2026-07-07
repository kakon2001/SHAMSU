import type { AgentResponse, FileContent, FileNode } from "../types";

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

// ------------------------------------------------------------------- agent

export function getAgentState(): Promise<AgentResponse> {
  return fetch(`${API_BASE}/api/agent/state`).then((res) => handle<AgentResponse>(res));
}

export function postChat(message: string, contextFiles: string[] = []): Promise<AgentResponse> {
  return post("/api/agent/chat", { message, context_files: contextFiles });
}

export function postApproval(id: string, approved: boolean): Promise<AgentResponse> {
  return post("/api/agent/approval", { id, approved });
}

export function postContinue(): Promise<AgentResponse> {
  return post("/api/agent/continue");
}

export function postStop(): Promise<AgentResponse> {
  return post("/api/agent/stop");
}

export function postReset(): Promise<AgentResponse> {
  return post("/api/agent/reset");
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
