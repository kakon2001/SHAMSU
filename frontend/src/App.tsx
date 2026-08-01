import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import "./App.css";
import {
  createSession,
  getCurrentUser,
  deleteSession,
  getAdminOverview,
  getContextDashboard,
  getFileContent,
  getFileTree,
  getGitLog,
  getGitSearch,
  getGitStatus,
  searchWeb,
  getModels,
  listSessions,
  login,
  logout,
  register,
  runTaskBuild,
  saveFileContent,
  setAuthToken,
  setCurrentModel,
} from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { EditorPane } from "./components/EditorPane";
import { useAgent } from "./hooks/useAgent";
import type { AdminOverview, AuthUser, ChatItem, ContextDashboard, EditorTab, FileNode, GitCommit, GitSearchHit, GitStatus, ModelState, SessionInfo, WebSearchResult } from "./types";

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
  const [gitStatus, setGitStatus] = useState<GitStatus | null>(null);
  const [gitCommits, setGitCommits] = useState<GitCommit[]>([]);
  const [gitSearchQuery, setGitSearchQuery] = useState("");
  const [gitSearchHits, setGitSearchHits] = useState<GitSearchHit[]>([]);
  const [webSearchQuery, setWebSearchQuery] = useState("");
  const [webSearchResults, setWebSearchResults] = useState<WebSearchResult[]>([]);
  const [webSearchMessage, setWebSearchMessage] = useState("");
  const [dashboardOpen, setDashboardOpen] = useState(false);
  const [autoBuildBusy, setAutoBuildBusy] = useState(false);
  const [autoBuildItems, setAutoBuildItems] = useState<ChatItem[]>([]);
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authName, setAuthName] = useState("");
  const [authBusy, setAuthBusy] = useState(false);

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
    getGitStatus().then(setGitStatus).catch((err: Error) => setNotice(err.message));
    getGitLog().then((result) => setGitCommits(result.commits)).catch((err: Error) => setNotice(err.message));
  }, []);

  const refreshModels = useCallback(() => {
    getModels().then(setModelState).catch((err: Error) => setNotice(err.message));
  }, []);


  const runGitSearch = useCallback(() => {
    const query = gitSearchQuery.trim();
    if (!query) {
      setGitSearchHits([]);
      return;
    }
    getGitSearch(query)
      .then((result) => setGitSearchHits(result.hits))
      .catch((err: Error) => setNotice(err.message));
  }, [gitSearchQuery]);

  const runWebSearch = useCallback(() => {
    const query = webSearchQuery.trim();
    if (!query) {
      setWebSearchResults([]);
      setWebSearchMessage("");
      return;
    }
    setWebSearchMessage("Searching the web...");
    searchWeb(query)
      .then((result) => {
        setWebSearchResults(result.results ?? []);
        setWebSearchMessage(result.ok ? `${result.results.length} sourced result(s).` : result.error ?? "Web search failed.");
      })
      .catch((err: Error) => {
        setWebSearchResults([]);
        setWebSearchMessage(err.message);
      });
  }, [webSearchQuery]);

  const changeModel = useCallback((modelId: string) => {
    setCurrentModel(modelId)
      .then((state) => {
        setModelState(state);
        setNotice(`Model switched to ${state.current}.`);
      })
      .catch((err: Error) => setNotice(err.message));
  }, []);



  useEffect(() => {
    getCurrentUser()
      .then(setAuthUser)
      .catch(() => setAuthUser(null));
  }, []);

  useEffect(() => {
    if (!authUser) return;
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
  }, [authUser]);

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


  const runAutoBuild = useCallback((prompt: string) => {
    const id = `build-${Date.now()}`;
    setAutoBuildItems((prev) => [
      ...prev,
      { kind: "user", id: `${id}-user`, content: prompt },
      { kind: "assistant", id: `${id}-working`, content: "SHAMSU accepted this build request and is working on it..." },
    ]);
    setNotice("SHAMSU is working on your build request...");
    setAutoBuildBusy(true);
    runTaskBuild(prompt)
      .then((result) => {
        void refreshFileTree();
        reloadCleanTabs();
        refreshDashboard();
        if (result.preview_url) {
          window.open(result.preview_url, "_blank", "noopener,noreferrer");
        }
        const fileText = result.created_files.length ? `Created files: ${result.created_files.join(", ")}.` : "No files were created.";
        const section = (title: string, rows?: string[]) => rows?.length ? `\n\n${title}:\n${rows.map((row) => `- ${row}`).join("\n")}` : "";
        const workflowText = result.workflow_summary ? `\n\nWorkflow:\n${result.workflow_summary}` : "";
        const stepText = result.steps.length
          ? `\n\nVerification steps:\n${result.steps.slice(0, 8).map((step) => `- ${step.name}: ${step.status} - ${step.detail}`).join("\n")}`
          : "";
        const noteText = result.notes.length ? `\n\nNotes:\n${result.notes.map((note) => `- ${note}`).join("\n")}` : "";
        const reliability = result.reliability;
        const reliabilityText = reliability
          ? `\n\nReliability loop:\n${reliability.phases.map((phase) => `- ${phase}`).join("\n")}\n- final: ${reliability.final_status}\n- repair attempts: ${reliability.repair_attempts}\n- next: ${reliability.next_action}`
          : "";
        const summary = [
          result.ok ? "Build completed." : "Build needs follow-up.",
          `Mode: ${result.mode}.`,
          fileText,
          result.preview_url ? `Preview: ${result.preview_url}` : "Preview: not available until verification passes.",
        ].join("\n")
          + section("Requirement analysis", result.requirements_analysis)
          + section("Clarification questions", result.clarification_questions)
          + section("Chosen stack", result.stack)
          + section("File plan", result.file_plan)
          + workflowText
          + stepText
          + reliabilityText
          + noteText;
        setAutoBuildItems((prev) => prev.map((item) => item.id === `${id}-working` ? { ...item, content: summary } : item));
        setNotice(`${result.ok ? "Autonomous build completed." : "Autonomous build needs follow-up."} ${fileText}`);
      })
      .catch((err: Error) => {
        setAutoBuildItems((prev) => prev.map((item) => item.id === `${id}-working` ? { ...item, content: `Build failed: ${err.message}` } : item));
        setNotice(err.message);
      })
      .finally(() => setAutoBuildBusy(false));
  }, [refreshDashboard, refreshFileTree, reloadCleanTabs]);

  const { items, connected, busy, sendChat, respondApproval, stop, reset } = useAgent(
    activeSessionId,
    handleFilesChanged,
    refreshSessions,
  );
  const sendOrBuild = useCallback((text: string, contextFiles: string[]) => {
    const lower = text.toLowerCase();
    const shouldAutoBuild = contextFiles.length === 0 && /\b(make|build|create|generate|develop|implement|write)\b/.test(lower) && /\b(game|app|application|website|web page|html|system|tool|program|project|calculator|todo|quiz|crm|management|dashboard|portal|inventory|student|library|os|operating system)\b/.test(lower);
    if (shouldAutoBuild) {
      runAutoBuild(text);
      return;
    }
    sendChat(text, contextFiles);
  }, [runAutoBuild, sendChat]);

  useEffect(() => {
    void refreshFileTree();
  }, [refreshFileTree]);

  useEffect(() => {
    refreshModels();
    refreshDashboard();
  }, [refreshDashboard, refreshModels]);

  
  // CLI-created sessions are recorded by the backend; poll so the browser catches them.
  useEffect(() => {
    if (!authUser) return;
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


  const submitAuth = useCallback((event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setAuthBusy(true);
    const action = authMode === "register" ? register(authEmail, authPassword, authName) : login(authEmail, authPassword);
    action
      .then((response) => {
        setAuthToken(response.token);
        setAuthUser(response.user);
        setAuthPassword("");
        setNotice(`Signed in as ${response.user.email}.`);
      })
      .catch((err: Error) => setNotice(err.message))
      .finally(() => setAuthBusy(false));
  }, [authEmail, authMode, authName, authPassword]);

  const signOut = useCallback(() => {
    logout().finally(() => {
      setAuthUser(null);
      setSessions([]);
      setActiveSessionId(null);
      setAutoBuildItems([]);
      setTabs([]);
      setActivePath(null);
      setNotice("Signed out.");
    });
  }, []);
  const workspaceFiles = flattenFiles(fileTree);

  if (!authUser) {
    return (
      <div className="auth-page">
        <form className="auth-card" onSubmit={submitAuth}>
          <h1>SHAMSU</h1>
          <p>Sign in with your email to open your own chats, history, uploads, and workspace sessions.</p>
          {authMode === "register" && (
            <label>
              Name
              <input value={authName} onChange={(e) => setAuthName(e.target.value)} placeholder="Your name" />
            </label>
          )}
          <label>
            Email
            <input type="email" value={authEmail} onChange={(e) => setAuthEmail(e.target.value)} placeholder="you@example.com" required />
          </label>
          <label>
            Password
            <input type="password" value={authPassword} onChange={(e) => setAuthPassword(e.target.value)} placeholder="At least 6 characters" required />
          </label>
          <button className="auth-card__submit" type="submit" disabled={authBusy}>{authBusy ? "Please wait" : authMode === "register" ? "Create account" : "Sign in"}</button>
          <button className="auth-card__switch" type="button" onClick={() => setAuthMode(authMode === "register" ? "login" : "register")}>
            {authMode === "register" ? "Already have an account? Sign in" : "New to SHAMSU? Create account"}
          </button>
          {notice && <div className="auth-card__notice">{notice}</div>}
        </form>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app__header">
        <span className="app__title">SHAMSU</span>
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
        <div className="app__user-pill" title={authUser.email}>{authUser.email}</div>
        <button className="app__logout" onClick={signOut}>Logout</button>
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
          <div className="dashboard-git">
            <div className="dashboard-panel__header">
              <strong>Git Dashboard</strong>
              <span>{gitStatus ? `${gitStatus.branch} · ${gitStatus.clean ? "clean" : `${gitStatus.files.length} changed`}` : "not loaded"}</span>
            </div>
            <div className="dashboard-columns">
              <div>
                <h3>Changed Files</h3>
                <div className="dashboard-list">
                  {(gitStatus?.files ?? []).slice(0, 10).map((file) => <div key={`${file.code}-${file.path}`}>{file.code} {file.path}</div>)}
                  {gitStatus && gitStatus.files.length === 0 && <div>Working tree clean.</div>}
                </div>
                <div className="git-search-row">
                  <input value={gitSearchQuery} onChange={(e) => setGitSearchQuery(e.target.value)} placeholder="Search code or commits" />
                  <button className="btn" onClick={runGitSearch}>Search Git</button>
                </div>
                <div className="dashboard-list">
                  {gitSearchHits.slice(0, 8).map((hit, index) => <div key={`${hit.kind}-${index}`}>{hit.kind}: {hit.path ? `${hit.path}${hit.line ? `:${hit.line}` : ""}` : hit.commit?.slice(0, 7)} {hit.text}</div>)}
                </div>
              </div>
              <div>
                <h3>Recent Commits</h3>
                <div className="dashboard-list">
                  {gitCommits.slice(0, 8).map((commit) => <div key={commit.hash}>{commit.hash.slice(0, 7)} {commit.date} {commit.author}: {commit.subject}</div>)}
                </div>
              </div>
            </div>
          </div>
          <div className="dashboard-git">
            <div className="dashboard-panel__header">
              <strong>Web Search</strong>
              <span>read-only sourced lookup</span>
            </div>
            <div className="web-search-row">
              <input
                value={webSearchQuery}
                onChange={(e) => setWebSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") runWebSearch();
                }}
                placeholder="Search current docs, news, or references"
              />
              <button className="btn" onClick={runWebSearch}>Search Web</button>
            </div>
            {webSearchMessage && <div className="web-search-message">{webSearchMessage}</div>}
            <div className="web-search-results">
              {webSearchResults.slice(0, 5).map((result) => (
                <a key={result.url} href={result.url} target="_blank" rel="noreferrer" className="web-result-card">
                  <strong>{result.title}</strong>
                  <span>{result.url}</span>
                  {result.snippet && <p>{result.snippet}</p>}
                </a>
              ))}
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
            items={[...items, ...autoBuildItems]}
            busy={busy || autoBuildBusy}
            connected={connected}
            files={workspaceFiles}
            activePath={activePath}
            onSend={sendOrBuild}
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































