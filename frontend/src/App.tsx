import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import {
  createSession,
  deleteSession,
  getAdminOverview,
  getContextDashboard,
  getFileContent,
  getFileTree,
  getModels,
  listSessions,
  saveFileContent,
  setCurrentModel,
} from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { EditorPane } from "./components/EditorPane";
import { useAgent } from "./hooks/useAgent";
import type { AdminOverview, ContextDashboard, EditorTab, FileNode, ModelState, SessionInfo } from "./types";

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
  const [modelState, setModelState] = useState<ModelState | null>(null);
  const [adminOverview, setAdminOverview] = useState<AdminOverview | null>(null);
  const [contextDashboard, setContextDashboard] = useState<ContextDashboard | null>(null);
  const [dashboardOpen, setDashboardOpen] = useState(false);

  const tabsRef = useRef(tabs);
  tabsRef.current = tabs;
  const activeSessionIdRef = useRef<string | null>(activeSessionId);
  activeSessionIdRef.current = activeSessionId;

  const refreshSessions = useCallback((selectNewestCli = false) => {
    listSessions()
      .then((list) => {
        setSessions(list);
        const newest = list[0];
        if (selectNewestCli && newest?.title.startsWith("CLI ") && newest.id !== activeSessionIdRef.current) {
          setActiveSessionId(newest.id);
        }
      })
      .catch(() => {
        // Backend unreachable; the connection badge already reflects it.
      });
  }, []);

  const refreshDashboard = useCallback(() => {
    getAdminOverview().then(setAdminOverview).catch((err: Error) => setNotice(err.message));
    getContextDashboard().then(setContextDashboard).catch((err: Error) => setNotice(err.message));
  }, []);

  const refreshModels = useCallback(() => {
    getModels().then(setModelState).catch((err: Error) => setNotice(err.message));
  }, []);

  const changeModel = useCallback((modelId: string) => {
    setCurrentModel(modelId)
      .then((state) => {
        setModelState(state);
        setNotice(`Model switched to ${state.current}.`);
      })
      .catch((err: Error) => setNotice(err.message));
  }, []);

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
    refreshDashboard();
  }, [refreshFileTree, reloadCleanTabs, refreshDashboard]);

  const { items, connected, busy, sendChat, respondApproval, stop, reset } = useAgent(
    activeSessionId,
    handleFilesChanged,
    refreshSessions,
  );

  useEffect(() => {
    void refreshFileTree();
  }, [refreshFileTree]);

  useEffect(() => {
    refreshModels();
    refreshDashboard();
  }, [refreshDashboard, refreshModels]);

  
  // CLI-created sessions are recorded by the backend; poll so the browser catches them.
  useEffect(() => {
    const timer = window.setInterval(() => {
      refreshSessions(true);
      void refreshFileTree();
      if (dashboardOpen) refreshDashboard();
    }, 4000);
    return () => window.clearInterval(timer);
  }, [dashboardOpen, refreshDashboard, refreshFileTree, refreshSessions]);

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
        <div className="app__model-switcher">
          <span>Model</span>
          <select
            value={modelState?.current ?? ""}
            onChange={(e) => changeModel(e.target.value)}
            disabled={!modelState}
          >
            {modelState?.models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label}
              </option>
            ))}
          </select>
        </div>
        <button
          className="app__dashboard-toggle"
          onClick={() => {
            setDashboardOpen((open) => !open);
            refreshDashboard();
          }}
        >
          Admin / Context
        </button>
      </header>

      {notice && (
        <div className="app__notice" onClick={() => setNotice(null)} title="Click to dismiss">
          {notice}
        </div>
      )}

      {dashboardOpen && (
        <section className="dashboard-panel">
          <div className="dashboard-panel__header">
            <strong>Admin and Context Dashboard</strong>
            <button className="btn" onClick={refreshDashboard}>Refresh dashboard</button>
          </div>
          <div className="dashboard-grid">
            <div className="dashboard-card"><span className="dashboard-card__label">Sessions</span><strong>{adminOverview?.session_count ?? 0}</strong></div>
            <div className="dashboard-card"><span className="dashboard-card__label">Prompts</span><strong>{adminOverview?.totals.user_message ?? 0}</strong></div>
            <div className="dashboard-card"><span className="dashboard-card__label">Approvals</span><strong>{adminOverview?.totals.approval_request ?? 0}</strong></div>
            <div className="dashboard-card"><span className="dashboard-card__label">Indexed Files</span><strong>{contextDashboard?.file_count ?? 0}</strong></div>
            <div className="dashboard-card"><span className="dashboard-card__label">Context Chunks</span><strong>{contextDashboard?.chunk_count ?? 0}</strong></div>
            <div className="dashboard-card"><span className="dashboard-card__label">Uploads</span><strong>{contextDashboard?.uploaded_count ?? 0}</strong></div>
          </div>
          <div className="dashboard-columns">
            <div>
              <h3>Recent Activity</h3>
              <div className="dashboard-list">
                {(adminOverview?.recent_events ?? []).slice(0, 8).map((event, index) => (
                  <div key={`${event.timestamp}-${index}`}>{event.type}: {event.summary}</div>
                ))}
              </div>
            </div>
            <div>
              <h3>Context Terms</h3>
              <div className="dashboard-terms">
                {(contextDashboard?.top_terms ?? []).map((term) => <span key={term}>{term}</span>)}
              </div>
              <h3>Largest Indexed Files</h3>
              <div className="dashboard-list">
                {(contextDashboard?.largest_files ?? []).slice(0, 5).map((file) => (
                  <div key={file.path}>{file.path} ({file.chars.toLocaleString()} chars)</div>
                ))}
              </div>
            </div>
          </div>
        </section>
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
                  {s.busy && s.id !== activeSessionId ? "* " : ""}
                  {s.title}
                </option>
              ))}
            </select>
            <button className="app__new-chat" onClick={newSession} title="Start a new session">New session</button>
            <button className="app__new-chat" onClick={reset} disabled={busy} title="Clear this session's transcript">Clear</button>
            <button className="app__new-chat app__session-delete" onClick={removeSession} disabled={busy || !activeSessionId} title="Delete this session">Delete</button>
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
            onUploaded={() => {
              void refreshFileTree(true);
              refreshDashboard();
            }}
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
                <option key={path} value={path}>{path}</option>
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


