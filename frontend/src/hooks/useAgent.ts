import { useCallback, useEffect, useRef, useState } from "react";
import {
  getAgentState,
  postApproval,
  postChat,
  postContinue,
  postReset,
  postStop,
} from "../api/client";
import type { AgentEvent, AgentResponse, ChatItem } from "../types";

let nextId = 0;
const uid = () => `local-${++nextId}`;

interface UseAgent {
  items: ChatItem[];
  connected: boolean;
  busy: boolean;
  sendChat: (content: string, contextFiles?: string[]) => void;
  respondApproval: (id: string, approved: boolean) => void;
  stop: () => void;
  reset: () => void;
}

function eventsToItems(
  prev: ChatItem[],
  events: AgentEvent[],
  includeUser: boolean,
  onFilesChanged: (paths: string[]) => void,
): ChatItem[] {
  let items = prev;
  for (const ev of events) {
    switch (ev.type) {
      case "user_message":
        if (includeUser)
          items = [
            ...items,
            { kind: "user", id: uid(), content: ev.content, contextFiles: ev.context_files },
          ];
        break;
      case "assistant_message":
        if (ev.id && items.some((it) => it.kind === "assistant" && it.id === ev.id)) {
          items = items.map((it) =>
            it.kind === "assistant" && it.id === ev.id ? { ...it, content: ev.content } : it,
          );
        } else {
          items = [...items, { kind: "assistant", id: ev.id ?? uid(), content: ev.content }];
        }
        break;
      case "assistant_delta":
        if (items.some((it) => it.kind === "assistant" && it.id === ev.id)) {
          items = items.map((it) =>
            it.kind === "assistant" && it.id === ev.id
              ? { ...it, content: it.content + ev.content }
              : it,
          );
        } else {
          items = [...items, { kind: "assistant", id: ev.id, content: ev.content }];
        }
        break;
      case "tool_call":
        items = [...items, { kind: "tool", id: ev.id, name: ev.name, args: ev.args, status: "running" }];
        break;
      case "tool_result":
        items = items.map((it) =>
          it.kind === "tool" && it.id === ev.id
            ? { ...it, status: ev.ok ? "done" : "error", preview: ev.preview }
            : it,
        );
        break;
      case "approval_request":
        items = [
          ...items,
          {
            kind: "approval",
            id: ev.id,
            name: ev.name,
            command: ev.command,
            path: ev.path,
            diff: ev.diff,
            isNewFile: ev.is_new_file,
            risk: ev.risk,
            riskReason: ev.risk_reason,
            status: "pending",
          },
        ];
        break;
      case "approval_resolved":
        items = items.map((it) =>
          it.kind === "approval" && it.id === ev.id
            ? { ...it, status: ev.approved ? "approved" : "rejected" }
            : it,
        );
        break;
      case "files_changed":
        onFilesChanged(ev.paths);
        break;
      case "error":
        items = [...items, { kind: "error", id: uid(), content: ev.message }];
        break;
      case "turn_end":
        break;
    }
  }
  return items;
}

/**
 * Drives one backend chat session at a time. Switching `sessionId` reloads the
 * transcript from the backend; a turn still running in another session keeps
 * running server-side and is picked up again when you switch back.
 */
export function useAgent(
  sessionId: string | null,
  onFilesChanged: (paths: string[]) => void,
  onSessionActivity?: () => void,
): UseAgent {
  const [items, setItems] = useState<ChatItem[]>([]);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);

  const onFilesChangedRef = useRef(onFilesChanged);
  onFilesChangedRef.current = onFilesChanged;
  const onSessionActivityRef = useRef(onSessionActivity);
  onSessionActivityRef.current = onSessionActivity;

  // The session the UI is currently showing — late responses from other
  // sessions must not touch the visible transcript.
  const sessionRef = useRef<string | null>(sessionId);
  sessionRef.current = sessionId;
  // Sessions with a pump loop in flight.
  const pumpingRef = useRef<Set<string>>(new Set());
  // Unresolved approval ids per session, tracked synchronously (React state
  // flushes too late for the pump loop's stop condition).
  const pendingRef = useRef<Map<string, Set<string>>>(new Map());

  const pendingOf = useCallback((sid: string): Set<string> => {
    let set = pendingRef.current.get(sid);
    if (!set) {
      set = new Set();
      pendingRef.current.set(sid, set);
    }
    return set;
  }, []);

  const trackPending = useCallback(
    (sid: string, events: AgentEvent[]) => {
      const set = pendingOf(sid);
      for (const ev of events) {
        if (ev.type === "approval_request") set.add(ev.id);
        else if (ev.type === "approval_resolved") set.delete(ev.id);
      }
    },
    [pendingOf],
  );

  const apply = useCallback((events: AgentEvent[], includeUser = false) => {
    setItems((prev) =>
      eventsToItems(prev, events, includeUser, (paths) => onFilesChangedRef.current(paths)),
    );
  }, []);

  const fail = useCallback((err: unknown) => {
    const message = err instanceof Error ? err.message : "Request to the agent backend failed.";
    setItems((prev) => [...prev, { kind: "error", id: uid(), content: message }]);
  }, []);

  // Apply a response, then keep long-polling while the agent is generating
  // with nothing for the user to approve. Stops early if the user switches
  // sessions — the backend turn keeps running without us.
  const pump = useCallback(
    async (sid: string, first: Promise<AgentResponse>, includeUser = false) => {
      if (pumpingRef.current.has(sid)) return;
      pumpingRef.current.add(sid);
      try {
        let res = await first;
        trackPending(sid, res.events);
        if (sessionRef.current === sid) {
          apply(res.events, includeUser);
          setBusy(res.busy);
        }
        while (res.busy && pendingOf(sid).size === 0 && sessionRef.current === sid) {
          res = await postContinue(sid);
          trackPending(sid, res.events);
          if (sessionRef.current === sid) {
            apply(res.events);
            setBusy(res.busy);
          }
        }
        // Turn settled (or paused) — let the app refresh session titles/order.
        onSessionActivityRef.current?.();
      } catch (err) {
        if (sessionRef.current === sid) {
          fail(err);
          setBusy(false);
        }
      } finally {
        pumpingRef.current.delete(sid);
      }
    },
    [apply, fail, pendingOf, trackPending],
  );

  // Load (or reload) the transcript whenever the active session changes.
  useEffect(() => {
    setItems([]);
    setBusy(false);
    if (!sessionId) return;
    getAgentState(sessionId)
      .then((res) => {
        if (sessionRef.current !== sessionId) return;
        setConnected(true);
        pendingRef.current.set(sessionId, new Set());
        trackPending(sessionId, res.events);
        apply(res.events, true);
        setBusy(res.busy);
        if (res.busy && pendingOf(sessionId).size === 0)
          void pump(sessionId, Promise.resolve({ events: [], busy: res.busy }));
      })
      .catch(() => setConnected(false));
  }, [sessionId, apply, pump, pendingOf, trackPending]);

  const sendChat = useCallback(
    (content: string, contextFiles: string[] = []) => {
      const sid = sessionRef.current;
      if (!sid) return;
      setItems((prev) => [...prev, { kind: "user", id: uid(), content, contextFiles }]);
      void pump(sid, postChat(sid, content, contextFiles));
    },
    [pump],
  );

  const respondApproval = useCallback(
    (id: string, approved: boolean) => {
      const sid = sessionRef.current;
      if (!sid) return;
      // Mark resolved immediately so the pump doesn't see a stale pending card.
      pendingOf(sid).delete(id);
      setItems((prev) =>
        prev.map((it) =>
          it.kind === "approval" && it.id === id
            ? { ...it, status: approved ? "approved" : "rejected" }
            : it,
        ),
      );
      void pump(sid, postApproval(sid, id, approved));
    },
    [pump, pendingOf],
  );

  const stop = useCallback(() => {
    const sid = sessionRef.current;
    if (!sid) return;
    postStop(sid)
      .then((res) => {
        if (sessionRef.current !== sid) return;
        apply(res.events);
        setBusy(res.busy);
        onSessionActivityRef.current?.();
      })
      .catch(fail);
  }, [apply, fail]);

  const reset = useCallback(() => {
    const sid = sessionRef.current;
    if (!sid) return;
    postReset(sid)
      .then(() => {
        if (sessionRef.current !== sid) return;
        setItems([]);
        setBusy(false);
        pendingRef.current.set(sid, new Set());
        onSessionActivityRef.current?.();
      })
      .catch(fail);
  }, [fail]);

  return { items, connected, busy, sendChat, respondApproval, stop, reset };
}
