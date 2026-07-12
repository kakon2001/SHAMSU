import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import {
  createSession,
  deleteSession,
  getFileContent,
  getFileTree,
  listSessions,
  saveFileContent,
} from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { EditorPane } from "./components/EditorPane";
import { useAgent } from "./hooks/useAgent";
import type { EditorTab, FileNode, SessionInfo } from "./types";

function flattenFiles(node: FileNode | null): string[] {
  if (!node) return [];
  if (node.type === "file") return [node.path];
  return (node.children ?? []).flatMap(flattenFiles);
}

function App() {
  const [fileTree, setFileTree] = useState<FileNode | null>(null);
  const [tabs, setTabs] = useState<EditorTab[]>([]);
  const [activePath, setActivePath] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [refreshingFiles, setRefreshingFiles] = useState(false);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  const tabsRef = useRef(tabs);
  tabsRef.current = tabs;

  const refreshSessions = useCallback(() => {
    listSessions()
      .then(setSessions)
      .catch(() => {
        // Backend unreachable; the connection badge already reflects it.
      });
  }, []);

  // Load stored sessions on startup; make sure at least one exists.
  useEffect(() => {
    listSessions()
      .then(async (list) => {
        if (list.length === 0) {
          const created = await createSession();
          list = [created];
        }
        setSessions(list);
        setActiveSessionId(list[0].id);
      })
      .catch((err: Error) => setNotice(`Could not load sessions: ${err.message}`));
  }, []);

  const newSession = useCallback(() => {
    createSession()
      .then((created) => {
        setSessions((prev) => [created, ...prev]);
        setActiveSessionId(created.id);
      })
      .catch((err: Error) => setNotice(err.message));
  }, []);

  const removeSession = useCallback(() => {
    if (!activeSessionId) return;
    const current = sessions.find((s) => s.id === activeSessionId);
    if (!window.confirm(`Delete session "${current?.title ?? "this session"}"?`)) return;
    deleteSession(activeSessionId)
      .then(async () => {
        const remaining = sessions.filter((s) => s.id !== activeSessionId);
        if (remaining.length === 0) {
          const created = await createSession();
          setSessions([created]);
          setActiveSessionId(created.id);
        } else {
          setSessions(remaining);
          setActiveSessionId(remaining[0].id);
        }
      })
      .catch((err: Error) => setNotice(err.message));
  }, [activeSessionId, sessions]);

  const refreshFileTree = useCallback((showNotice = false) => {
    setRefreshingFiles(true);
    return getFileTree()
      .then((tree) => {
        setFileTree(tree);
        if (showNotice) setNotice("Workspace files refreshed.");
      })
      .catch((err: Error) => setNotice(err.message))
      .finally(() => setRefreshingFiles(false));
  }, []);

  // Reload open tabs that the agent (or a shell command) may have rewritten on disk.
  // Dirty tabs are left alone so user edits are never clobbered.
  const reloadCleanTabs = useCallback(() => {
    for (const tab of tabsRef.current) {
      if (tab.content !== tab.savedContent) continue;
      getFileContent(tab.path)
        .then((file) =>
          setTabs((prev) =>
            prev.map((t) =>
              t.path === file.path && t.content === t.savedContent
                ? { ...t, content: file.content, savedContent: file.content }
                : t,
            ),
          ),
        )
        .catch(() => {
          // File may have been deleted; keep the tab, saving will recreate it.
        });
    }
  }, []);

  const handleFilesChanged = useCallback(() => {
    void refreshFileTree();
    reloadCleanTabs();
  }, [refreshFileTree, reloadCleanTabs]);

  const { items, connected, busy, sendChat, respondApproval, stop, reset } = useAgent(
    activeSessionId,
    handleFilesChanged,
    refreshSessions,
  );

  useEffect(() => {
    void refreshFileTree();
  }, [refreshFileTree]);

  const openFile = useCallback((path: string) => {
    setActivePath(path);
    if (tabsRef.current.some((t) => t.path === path)) return;
    getFileContent(path)
      .then((file) =>
        setTabs((prev) =>
          prev.some((t) => t.path === path)
            ? prev
            : [...prev, { path, content: file.content, savedContent: file.content }],
        ),
      )
      .catch((err: Error) => setNotice(err.message));
  }, []);

  const closeTab = useCallback((path: string) => {
    setTabs((prev) => {
      const next = prev.filter((t) => t.path !== path);
      setActivePath((current) =>
        current === path ? (next.length ? next[next.length - 1].path : null) : current,
      );
      return next;
    });
  }, []);

  const changeTab = useCallback((path: string, content: string) => {
    setTabs((prev) => prev.map((t) => (t.path === path ? { ...t, content } : t)));
  }, []);

  const saveTab = useCallback(
    (path: string) => {
      const tab = tabsRef.current.find((t) => t.path === path);
      if (!tab || tab.content === tab.savedContent) return;
      saveFileContent(path, tab.content)
        .then(() => {
          setTabs((prev) =>
            prev.map((t) => (t.path === path ? { ...t, savedContent: t.content } : t)),
          );
          void refreshFileTree();
        })
        .catch((err: Error) => setNotice(err.message));
    },
    [refreshFileTree],
  );

  const workspaceFiles = flattenFiles(fileTree);

  return (
    <div className="app">
      <header className="app__header">
        <span className="app__title">Local Coding Agent</span>
        <span className={`app__status app__status--${connected ? "on" : "off"}`}>
          {connected ? "connected" : "disconnected"}
        </span>
      </header>

      {notice && (
        <div className="app__notice" onClick={() => setNotice(null)} title="Click to dismiss">
          {notice}
        </div>
      )}

      <div className="app__body">
        <section className="chat-shell">
          <div className="chat-shell__toolbar">
          <select
            className="app__session-select"
            value={activeSessionId ?? ""}
            onChange={(e) => setActiveSessionId(e.target.value)}
            disabled={sessions.length === 0}
            title="Switch session"
          >
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.busy && s.id !== activeSessionId ? "â— " : ""}
                {s.title}
              </option>
            ))}
          </select>
          <button className="app__new-chat" onClick={newSession} title="Start a new session">
            New session
          </button>
          <button className="app__new-chat" onClick={reset} disabled={busy} title="Clear this session's transcript">
            Clear
          </button>
          <button
            className="app__new-chat app__session-delete"
            onClick={removeSession}
            disabled={busy || !activeSessionId}
            title="Delete this session"
          >
            Delete
          </button>
          </div>
          <ChatPanel
            items={items}
            busy={busy}
            connected={connected}
            files={workspaceFiles}
            activePath={activePath}
            onSend={sendChat}
            onStop={stop}
            onRespondApproval={respondApproval}
            onUploaded={() => void refreshFileTree(true)}
          />
        </section>
        <section className="workspace-panel">
          <div className="workspace-panel__toolbar">
            <select
              className="workspace-panel__select"
              value={activePath ?? ""}
              onChange={(e) => {
                if (e.target.value) openFile(e.target.value);
              }}
              disabled={workspaceFiles.length === 0}
              title="Open workspace file"
            >
              <option value="">Open workspace file</option>
              {workspaceFiles.map((path) => (
                <option key={path} value={path}>
                  {path}
                </option>
              ))}
            </select>
            <button
              className="workspace-panel__refresh"
              onClick={() => {
                void refreshFileTree(true);
                reloadCleanTabs();
              }}
              disabled={refreshingFiles}
            >
              {refreshingFiles ? "Refreshing" : "Refresh"}
            </button>
          </div>
          <EditorPane
            tabs={tabs}
            activePath={activePath}
            onSelect={setActivePath}
            onClose={closeTab}
            onChange={changeTab}
            onSave={saveTab}
          />
        </section>
      </div>
    </div>
  );
}

export default App;


