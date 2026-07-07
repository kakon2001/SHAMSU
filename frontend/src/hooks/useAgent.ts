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
        items = [...items, { kind: "assistant", id: uid(), content: ev.content }];
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

export function useAgent(onFilesChanged: (paths: string[]) => void): UseAgent {
  const [items, setItems] = useState<ChatItem[]>([]);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);

  const onFilesChangedRef = useRef(onFilesChanged);
  onFilesChangedRef.current = onFilesChanged;
  const pumpingRef = useRef(false);
  // Unresolved approval ids, tracked synchronously (React state flushes too late
  // for the pump loop's stop condition).
  const pendingIdsRef = useRef<Set<string>>(new Set());

  const trackPending = useCallback((events: AgentEvent[]) => {
    for (const ev of events) {
      if (ev.type === "approval_request") pendingIdsRef.current.add(ev.id);
      else if (ev.type === "approval_resolved") pendingIdsRef.current.delete(ev.id);
    }
  }, []);

  const apply = useCallback((events: AgentEvent[], includeUser = false) => {
    setItems((prev) =>
      eventsToItems(prev, events, includeUser, (paths) => onFilesChangedRef.current(paths)),
    );
  }, []);

  const fail = useCallback((err: unknown) => {
    const message = err instanceof Error ? err.message : "Request to the agent backend failed.";
    setItems((prev) => [...prev, { kind: "error", id: uid(), content: message }]);
  }, []);

  // Apply a response, then keep long-polling while the agent is generating with
  // nothing for the user to approve (e.g. after a page reload mid-turn).
  const pump = useCallback(
    async (first: Promise<AgentResponse>, includeUser = false) => {
      if (pumpingRef.current) return;
      pumpingRef.current = true;
      try {
        let res = await first;
        trackPending(res.events);
        apply(res.events, includeUser);
        setBusy(res.busy);
        while (res.busy && pendingIdsRef.current.size === 0) {
          res = await postContinue();
          trackPending(res.events);
          apply(res.events);
          setBusy(res.busy);
        }
      } catch (err) {
        fail(err);
        setBusy(false);
      } finally {
        pumpingRef.current = false;
      }
    },
    [apply, fail, trackPending],
  );

  useEffect(() => {
    getAgentState()
      .then((res) => {
        setConnected(true);
        setItems([]);
        pendingIdsRef.current = new Set();
        trackPending(res.events);
        apply(res.events, true);
        setBusy(res.busy);
        if (res.busy) void pump(Promise.resolve({ events: [], busy: res.busy }));
      })
      .catch(() => setConnected(false));
  }, [apply, pump, trackPending]);

  const sendChat = useCallback(
    (content: string, contextFiles: string[] = []) => {
      setItems((prev) => [...prev, { kind: "user", id: uid(), content, contextFiles }]);
      void pump(postChat(content, contextFiles));
    },
    [pump],
  );

  const respondApproval = useCallback(
    (id: string, approved: boolean) => {
      // Mark resolved immediately so the pump doesn't see a stale pending card.
      pendingIdsRef.current.delete(id);
      setItems((prev) =>
        prev.map((it) =>
          it.kind === "approval" && it.id === id
            ? { ...it, status: approved ? "approved" : "rejected" }
            : it,
        ),
      );
      void pump(postApproval(id, approved));
    },
    [pump],
  );

  const stop = useCallback(() => {
    postStop()
      .then((res) => {
        apply(res.events);
        setBusy(res.busy);
      })
      .catch(fail);
  }, [apply, fail]);

  const reset = useCallback(() => {
    postReset()
      .then(() => {
        setItems([]);
        setBusy(false);
        pendingIdsRef.current = new Set();
      })
      .catch(fail);
  }, [fail]);

  return { items, connected, busy, sendChat, respondApproval, stop, reset };
}
