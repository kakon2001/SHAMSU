import { useEffect, useRef, useState, type FormEvent } from "react";
import type { ChatItem } from "../types";
import { ApprovalCard } from "./ApprovalCard";
import { MessageBubble } from "./MessageBubble";
import { ToolChip } from "./ToolChip";

interface Props {
  items: ChatItem[];
  busy: boolean;
  connected: boolean;
  files: string[];
  activePath: string | null;
  onSend: (text: string, contextFiles: string[]) => void;
  onStop: () => void;
  onRespondApproval: (id: string, approved: boolean) => void;
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
}: Props) {
  const [input, setInput] = useState("");
  const [attached, setAttached] = useState<string[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const pickerRef = useRef<HTMLDivElement>(null);

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

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy || !connected) return;
    onSend(text, attached);
    setInput("");
    setAttached([]);
  }

  function attach(path: string) {
    setAttached((prev) => (prev.includes(path) ? prev : [...prev, path]));
    setPickerOpen(false);
    setFilter("");
  }

  const hasPendingApproval = items.some((it) => it.kind === "approval" && it.status === "pending");

  const pickable = files.filter((f) => !attached.includes(f));
  const filtered = filter
    ? pickable.filter((f) => f.toLowerCase().includes(filter.toLowerCase()))
    : pickable;
  // Surface the file open in the editor at the top — "add this file" is the common case.
  const ordered = activePath && filtered.includes(activePath)
    ? [activePath, ...filtered.filter((f) => f !== activePath)]
    : filtered;

  return (
    <div className="chat-panel">
      <div className="chat-panel__messages" ref={scrollRef}>
        {items.length === 0 && (
          <div className="chat-panel__empty">
            Ask the agent to explore, edit or run something in the workspace. Every shell command and
            file edit will wait for your approval here.
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
                  files={item.contextFiles}
                />
              );
            case "assistant":
              return <MessageBubble key={item.id} role="assistant" content={item.content} />;
            case "tool":
              return <ToolChip key={item.id} item={item} />;
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
        {busy && !hasPendingApproval && <div className="chat-panel__typing">Agent is working…</div>}
        {hasPendingApproval && (
          <div className="chat-panel__waiting">Waiting for your approval above ↑</div>
        )}
      </div>

      {attached.length > 0 && (
        <div className="chat-panel__chips">
          {attached.map((path) => (
            <span key={path} className="context-chip" title={path}>
              {path.split("/").pop()}
              <button
                type="button"
                className="context-chip__remove"
                onClick={() => setAttached((prev) => prev.filter((p) => p !== path))}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      <form className="chat-panel__input" onSubmit={handleSubmit}>
        <div className="chat-panel__plus-wrap" ref={pickerRef}>
          <button
            type="button"
            className="chat-panel__plus"
            title="Attach a workspace file as context"
            disabled={!connected || busy || files.length === 0}
            onClick={() => setPickerOpen((o) => !o)}
          >
            +
          </button>
          {pickerOpen && (
            <div className="file-picker">
              <input
                autoFocus
                className="file-picker__filter"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Filter files…"
              />
              <div className="file-picker__list">
                {ordered.length === 0 && <div className="file-picker__empty">No files.</div>}
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
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={connected ? "Ask the agent to do something…" : "Connecting to backend…"}
          disabled={!connected || busy}
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
