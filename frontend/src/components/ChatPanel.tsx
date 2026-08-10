import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { uploadContextFile } from "../api/client";
import type { ChatItem, UploadedContextFile } from "../types";
import { ApprovalCard } from "./ApprovalCard";
import { MessageBubble } from "./MessageBubble";

interface Props {
  items: ChatItem[];
  busy: boolean;
  connected: boolean;
  files: string[];
  activePath: string | null;
  onSend: (text: string, contextFiles: string[], contextFileLabels: Record<string, string>) => void;
  onStop: () => void;
  onRespondApproval: (id: string, approved: boolean) => void;
  onUploaded?: () => void;
}


const STARTER_PROMPTS = [
  "Read my uploaded PRD and summarize the build plan.",
  "Create a multi-file web app and verify it.",
  "Find bugs in the current workspace and suggest fixes.",
];

function promptPreview(text: string): string {
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > 120 ? `${compact.slice(0, 117)}...` : compact;
}

function userPromptNumber(items: ChatItem[], targetId: string): number {
  return items.filter((item) => item.kind === "user").findIndex((item) => item.id === targetId) + 1;
}

export function ChatPanel({
  items,
  busy,
  connected,
  files,
  activePath,
  onSend,
  onStop,
  onRespondApproval,
  onUploaded,
}: Props) {
  const [input, setInput] = useState("");
  const [attached, setAttached] = useState<string[]>([]);
  const [attachmentLabels, setAttachmentLabels] = useState<Record<string, string>>({});
  const [pickerOpen, setPickerOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ kind: "success" | "error"; text: string; file?: UploadedContextFile } | null>(null);
  const [showPromptTracker, setShowPromptTracker] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pickerRef = useRef<HTMLDivElement>(null);
  const uploadRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [items]);

  useEffect(() => {
    if (!pickerOpen) return;
    function onClickOutside(e: MouseEvent) {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setPickerOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [pickerOpen]);

  function submitPrompt() {
    const text = input.trim();
    if (!text || busy || !connected) return;
    onSend(text, attached, attachmentLabels);
    setInput("");
    setAttached([]);
    setAttachmentLabels({});
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    submitPrompt();
  }

  function handleInputKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitPrompt();
    }
  }

  function attach(path: string) {
    setAttached((prev) => (prev.includes(path) ? prev : [...prev, path]));
    setPickerOpen(false);
    setFilter("");
  }

  async function handleUpload(file: File | undefined) {
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);
    try {
      const uploaded = await uploadContextFile(file);
      setAttachmentLabels((prev) => ({ ...prev, [uploaded.path]: uploaded.name }));
      attach(uploaded.path);
      setUploadStatus({
        kind: "success",
        text: `${uploaded.name} attached. Extracted ${uploaded.chars.toLocaleString()} chars, ${(uploaded.words ?? 0).toLocaleString()} words.`,
        file: uploaded,
      });
      onUploaded?.();
    } catch (err) {
      setUploadStatus({ kind: "error", text: err instanceof Error ? err.message : "Upload failed." });
    } finally {
      setUploading(false);
      if (uploadRef.current) uploadRef.current.value = "";
    }
  }

  const hasPendingApproval = items.some((it) => it.kind === "approval" && it.status === "pending");
  const recentActivity = items
    .filter((it) => it.kind === "user" || it.kind === "approval" || it.kind === "error")
    .slice(-8);
  const userItems = items.filter((it) => it.kind === "user");
  const latestUser = userItems[userItems.length - 1];
  const latestPromptNumber = latestUser ? userPromptNumber(items, latestUser.id) : 0;
  const statusText = hasPendingApproval
    ? "Approval needed before SHAMSU continues"
    : busy && latestUser
      ? `Working on prompt ${latestPromptNumber}`
      : connected
        ? "Ready for your next prompt"
        : "Backend disconnected";

  const pickable = files.filter((f) => !attached.includes(f));
  const filtered = filter
    ? pickable.filter((f) => f.toLowerCase().includes(filter.toLowerCase()))
    : pickable;
  // Surface the file open in the editor at the top - "add this file" is the common case.
  const ordered = activePath && filtered.includes(activePath)
    ? [activePath, ...filtered.filter((f) => f !== activePath)]
    : filtered;

  return (
    <div className="chat-panel">
      <div className={`chat-panel__status-strip${busy ? " is-busy" : ""}${hasPendingApproval ? " needs-approval" : ""}`}>
        <strong>{statusText}</strong>
        <span>{latestPromptNumber ? `Prompt ${latestPromptNumber} is newest; earlier prompts stay above it.` : "Start a session by sending a prompt or uploading a file."}</span>
      </div>
      <details className="activity-history" open={showPromptTracker} onToggle={(event) => setShowPromptTracker(event.currentTarget.open)}>
        <summary><span>Prompt history</span><small>{latestPromptNumber ? `${latestPromptNumber} prompts` : "recent prompts"}</small></summary>
        <div className="activity-history__list">
          {recentActivity.length === 0 && <div className="activity-history__empty">No activity yet.</div>}
          {recentActivity.map((item) => {
            if (item.kind === "user") {
              const number = userPromptNumber(items, item.id);
              const latest = item.id === latestUser?.id ? " latest" : "";
              return <div key={item.id} className={`activity-history__row${latest}`}>Prompt {number}: {promptPreview(item.content)}</div>;
            }
            if (item.kind === "approval") {
              return <div key={item.id}>Approval: {item.name} ({item.status})</div>;
            }
            return <div key={item.id}>Error: {item.content}</div>;
          })}
        </div>
      </details>
      <div className="chat-panel__messages" ref={scrollRef}>
        {items.length === 0 && (
          <div className="chat-panel__empty">
            <strong>What should SHAMSU build or inspect?</strong>
            <span>Upload a file, attach workspace context, or ask for a build plan.</span>
            <div className="chat-panel__starters">
              {STARTER_PROMPTS.map((prompt) => (
                <button key={prompt} type="button" onClick={() => setInput(prompt)} disabled={!connected || busy}>
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}
        {items.map((item) => {
          switch (item.kind) {
            case "user":
              return (
                <MessageBubble
                  key={item.id}
                  role="user"
                  content={item.content}
                  badge={item.id === latestUser?.id ? `Latest prompt ${latestPromptNumber}` : undefined}
                  files={item.contextFiles}
                  fileLabels={item.contextFileLabels}
                />
              );
            case "assistant":
              return <MessageBubble key={item.id} role="assistant" content={item.content} />;
            case "tool":
              return null;
            case "approval":
              return <ApprovalCard key={item.id} item={item} onRespond={onRespondApproval} />;
            case "error":
              return (
                <div key={item.id} className="chat-panel__error">
                  {item.content}
                </div>
              );
          }
        })}
        {busy && !hasPendingApproval && <div className="chat-panel__typing"><span />SHAMSU is working...</div>}
        {hasPendingApproval && (
          <div className="chat-panel__waiting">Waiting for your approval above</div>
        )}
      </div>

      {latestUser && (
        <div className="chat-panel__current-prompt">
          <strong>Current prompt {latestPromptNumber}</strong>
          <span>{promptPreview(latestUser.content)}</span>
        </div>
      )}

      {attached.length > 0 && (
        <div className="chat-panel__chips">
          {attached.map((path) => (
            <span key={path} className="context-chip" title={path}>
              {attachmentLabels[path] ? `Uploaded: ${attachmentLabels[path]}` : path.split("/").pop()}
              <button
                type="button"
                className="context-chip__remove"
                onClick={() => {
                  setAttached((prev) => prev.filter((p) => p !== path));
                  setAttachmentLabels((prev) => {
                    const next = { ...prev };
                    delete next[path];
                    return next;
                  });
                }}
              >
                x
              </button>
            </span>
          ))}
        </div>
      )}
      {uploadStatus && (
        <div className={`chat-panel__upload-status chat-panel__upload-status--${uploadStatus.kind}`}>
          <strong>{uploadStatus.text}</strong>
          {uploadStatus.file?.preview && <p>{uploadStatus.file.preview}</p>}
          {uploadStatus.file?.suggested_prompts?.length ? (
            <div className="chat-panel__upload-suggestions">
              {uploadStatus.file.suggested_prompts.map((prompt) => (
                <button key={prompt} type="button" onClick={() => setInput(prompt)}>
                  {prompt}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      )}

      <form className="chat-panel__input" onSubmit={handleSubmit}>
        <div className="chat-panel__plus-wrap" ref={pickerRef}>
          <button
            type="button"
            className="chat-panel__plus"
            title="Add files or context"
            disabled={!connected || busy}
            onClick={() => setPickerOpen((o) => !o)}
            aria-expanded={pickerOpen}
          >
            +
          </button>
          {pickerOpen && (
            <div className="file-picker file-picker--menu">
              <div className="file-picker__section">
                <button
                  type="button"
                  className="file-picker__action"
                  disabled={uploading}
                  onClick={() => uploadRef.current?.click()}
                >
                  <strong>Upload from laptop</strong>
                  <span>Attach a PDF, Word, text, code, or data file.</span>
                </button>
                <button
                  type="button"
                  className="file-picker__action"
                  disabled={!activePath || attached.includes(activePath)}
                  onClick={() => activePath && attach(activePath)}
                >
                  <strong>Attach open workspace file</strong>
                  <span>{activePath ?? "Open a workspace file first."}</span>
                </button>
              </div>
              <div className="file-picker__divider" />
              <input
                autoFocus
                className="file-picker__filter"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Search workspace files..."
              />
              <div className="file-picker__list">
                {ordered.length === 0 && <div className="file-picker__empty">No workspace files available.</div>}
                {ordered.map((path) => (
                  <button
                    key={path}
                    type="button"
                    className="file-picker__item"
                    onClick={() => attach(path)}
                    title={path}
                  >
                    {path}
                    {path === activePath && <span className="file-picker__hint"> (open in editor)</span>}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
        <input
          ref={uploadRef}
          type="file"
          className="chat-panel__upload-input"
          accept=".pdf,.docx,.txt,.md,.csv,.json,.py,.js,.jsx,.ts,.tsx,.html,.css,.yaml,.yml,.log"
          onChange={(e) => void handleUpload(e.target.files?.[0])}
        />
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleInputKeyDown}
          placeholder={connected ? "Message SHAMSU..." : "Connecting to backend..."}
          disabled={!connected || busy}
          rows={1}
        />
        {busy ? (
          <button type="button" className="btn btn--stop" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!connected || !input.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  );
}

