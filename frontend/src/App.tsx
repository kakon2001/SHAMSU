import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import "./App.css";
import {
  createSession,
  getCurrentUser,
  deleteSession,
  getAdminOverview,
  getContextDashboard,
  getVectorStats,
  getConnectors,
  getFileContent,
  getFileTree,
  getGitLog,
  getGitSearch,
  getGitStatus,
  getReportHistory,
  getReportSchema,
  searchVectorContext,
  runReportQuery,
  searchWeb,
  getModels,
  listSessions,
  login,
  logout,
  register,
  rebuildVectorIndex,
  runTaskBuild,
  saveFileContent,
  setAuthToken,
  setCurrentModel,
} from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { EditorPane } from "./components/EditorPane";
import { useAgent } from "./hooks/useAgent";
import type { AdminOverview, AuthUser, ChatItem, ConnectorMarketplace, ContextDashboard, EditorTab, FileNode, GitCommit, GitSearchHit, GitStatus, ModelState, SessionInfo, WebSearchResult, VectorMatch, VectorStats, ReportHistoryItem, ReportQueryResult, ReportSchema } from "./types";

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
  const [connectorMarketplace, setConnectorMarketplace] = useState<ConnectorMarketplace | null>(null);
  const [gitStatus, setGitStatus] = useState<GitStatus | null>(null);
  const [gitCommits, setGitCommits] = useState<GitCommit[]>([]);
  const [gitSearchQuery, setGitSearchQuery] = useState("");
  const [gitSearchHits, setGitSearchHits] = useState<GitSearchHit[]>([]);
  const [webSearchQuery, setWebSearchQuery] = useState("");
  const [webSearchResults, setWebSearchResults] = useState<WebSearchResult[]>([]);
  const [webSearchMessage, setWebSearchMessage] = useState("");
  const [vectorStats, setVectorStats] = useState<VectorStats | null>(null);
  const [vectorQuery, setVectorQuery] = useState("");
  const [vectorMatches, setVectorMatches] = useState<VectorMatch[]>([]);
  const [vectorMessage, setVectorMessage] = useState("");
  const [vectorBusy, setVectorBusy] = useState(false);
  const [reportSchema, setReportSchema] = useState<ReportSchema | null>(null);
  const [reportSql, setReportSql] = useState("SELECT title, updated_at, owner_id FROM sessions ORDER BY updated_at DESC");
  const [reportTitle, setReportTitle] = useState("Recent sessions");
  const [reportResult, setReportResult] = useState<ReportQueryResult | null>(null);
  const [reportHistory, setReportHistory] = useState<ReportHistoryItem[]>([]);
  const [reportMessage, setReportMessage] = useState("");
  const [reportBusy, setReportBusy] = useState(false);
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
    getVectorStats().then(setVectorStats).catch((err: Error) => setNotice(err.message));
    getConnectors().then(setConnectorMarketplace).catch((err: Error) => setNotice(err.message));
    getGitStatus().then(setGitStatus).catch((err: Error) => setNotice(err.message));
    getGitLog().then((result) => setGitCommits(result.commits)).catch((err: Error) => setNotice(err.message));
    getReportSchema().then(setReportSchema).catch((err: Error) => setNotice(err.message));
    getReportHistory().then((result) => setReportHistory(result.reports)).catch((err: Error) => setNotice(err.message));
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


  const rebuildSemanticIndex = useCallback(() => {
    setVectorBusy(true);
    setVectorMessage("Rebuilding vector index...");
    rebuildVectorIndex()
      .then((result) => {
        setVectorStats(result);
        setVectorMessage(`Indexed ${result.indexed_files} file(s) into ${result.indexed_chunks} vector chunk(s).`);
        refreshDashboard();
      })
      .catch((err: Error) => setVectorMessage(err.message))
      .finally(() => setVectorBusy(false));
  }, [refreshDashboard]);

  const runVectorSearch = useCallback(() => {
    const query = vectorQuery.trim();
    if (!query) {
      setVectorMessage("Enter a semantic query first.");
      return;
    }
    setVectorBusy(true);
    setVectorMessage("Searching vector index...");
    searchVectorContext(query)
      .then((result) => {
        setVectorMatches(result.matches);
        setVectorMessage(result.matches.length ? `${result.matches.length} semantic match(es).` : "No vector matches found.");
      })
      .catch((err: Error) => {
        setVectorMatches([]);
        setVectorMessage(err.message);
      })
      .finally(() => setVectorBusy(false));
  }, [vectorQuery]);


  const runSqlReport = useCallback(() => {
    const sql = reportSql.trim();
    if (!sql) {
      setReportMessage("Enter a SELECT query first.");
      return;
    }
    setReportBusy(true);
    setReportMessage("Running read-only SQL report...");
    runReportQuery(sql, reportTitle.trim(), 100, true)
      .then((result) => {
        setReportResult(result);
        setReportMessage(`Returned ${result.row_count} row(s) in ${result.elapsed_ms} ms.`);
        return getReportHistory();
      })
      .then((history) => setReportHistory(history.reports))
      .catch((err: Error) => {
        setReportResult(null);
        setReportMessage(err.message);
      })
      .finally(() => setReportBusy(false));
  }, [reportSql, reportTitle]);
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
          "SHAMSU advisory build plan",
          `Request: ${result.goal}`,
          `Mode: ${result.mode}.`,
        ].join("\n")
          + section("Requirement analysis", result.requirements_analysis)
          + section("What may need clarification later", result.clarification_questions)
          + section("Recommended first-version setup", result.stack)
          + section("File plan", result.file_plan)
          + workflowText
          + "\n\nBuild result:\n"
          + [
            result.ok ? "- Build completed and verified." : "- Build needs follow-up.",
            `- ${fileText}`,
            result.preview_url ? `- Preview: ${result.preview_url}` : "- Preview: not available until verification passes.",
          ].join("\n")
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
  const sendOrBuild = useCallback((text: string, contextFiles: string[], contextFileLabels: Record<string, string> = {}) => {
    const lower = text.toLowerCase();
    const shouldAutoBuild = contextFiles.length === 0 && /\b(make|build|create|generate|develop|implement|write)\b/.test(lower) && /\b(game|app|application|website|web page|html|system|tool|program|project|calculator|todo|quiz|crm|management|dashboard|portal|inventory|student|library|os|operating system)\b/.test(lower);
    if (shouldAutoBuild) {
      runAutoBuild(text);
      return;
    }
    sendChat(text, contextFiles, contextFileLabels);
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
              <strong>Vector Context</strong>
              <span>{vectorStats ? `${vectorStats.chunk_count} chunks / ${vectorStats.dims} dims` : "not built"}</span>
            </div>
            <div className="vector-actions">
              <button className="btn" onClick={rebuildSemanticIndex} disabled={vectorBusy}>Rebuild Vector Index</button>
              <div>{vectorStats?.ready ? "semantic index ready" : "semantic index not built yet"}</div>
            </div>
            <div className="vector-search-row">
              <input
                value={vectorQuery}
                onChange={(e) => setVectorQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") runVectorSearch();
                }}
                placeholder="Ask by meaning, for example: login sessions, game controls, database history"
              />
              <button className="btn" onClick={runVectorSearch} disabled={vectorBusy}>Semantic Search</button>
            </div>
            {vectorMessage && <div className="web-search-message">{vectorMessage}</div>}
            <div className="vector-results">
              {vectorMatches.slice(0, 6).map((match) => (
                <button key={`${match.path}-${match.chunk_index}`} className="vector-result-card" onClick={() => openFile(match.path)}>
                  <strong>{match.path}:{match.start_line}-{match.end_line}</strong>
                  <span>score {match.score.toFixed(4)}</span>
                  <p>{match.preview}</p>
                </button>
              ))}
            </div>
          </div>
          <div className="dashboard-git">
            <div className="dashboard-panel__header">
              <strong>SQL Reporting</strong>
              <span>{reportSchema ? `${reportSchema.tables.length} tables` : "not loaded"}</span>
            </div>
            <div className="reporting-layout">
              <div className="reporting-schema">
                <h3>Reporting Database</h3>
                <p>{reportSchema?.database ?? "History database not loaded yet."}</p>
                <div className="dashboard-list reporting-table-list">
                  {(reportSchema?.tables ?? []).map((table) => (
                    <button key={table.name} className="report-history-item" onClick={() => setReportSql(`SELECT * FROM ${table.name} ORDER BY rowid DESC`)}>
                      <strong>{table.name}</strong>
                      <span>{table.row_count} rows</span>
                    </button>
                  ))}
                </div>
                <h3>Saved Reports</h3>
                <div className="dashboard-list reporting-table-list">
                  {reportHistory.slice(0, 8).map((item) => (
                    <button key={item.id} className="report-history-item" onClick={() => { setReportTitle(item.title); setReportSql(item.sql); }}>
                      <strong>{item.title}</strong>
                      <span>{item.row_count} rows</span>
                    </button>
                  ))}
                  {reportHistory.length === 0 && <div>No saved reports yet.</div>}
                </div>
              </div>
              <div>
                <h3>Run Read-Only SQL</h3>
                <input className="report-title-input" value={reportTitle} onChange={(e) => setReportTitle(e.target.value)} placeholder="Report title" />
                <textarea className="report-sql-input" value={reportSql} onChange={(e) => setReportSql(e.target.value)} spellCheck={false} />
                <div className="vector-actions">
                  <button className="btn" onClick={runSqlReport} disabled={reportBusy}>{reportBusy ? "Running..." : "Run Report"}</button>
                  <span>{reportMessage || "Only SELECT/WITH queries are allowed."}</span>
                </div>
                {reportResult && (
                  <div className="report-table-wrap">
                    <table className="report-table">
                      <thead>
                        <tr>{reportResult.columns.map((column) => <th key={column}>{column}</th>)}</tr>
                      </thead>
                      <tbody>
                        {reportResult.rows.map((row, index) => (
                          <tr key={index}>{reportResult.columns.map((column) => <td key={column}>{String(row[column] ?? "")}</td>)}</tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="dashboard-git">
            <div className="dashboard-panel__header">
              <strong>Git Dashboard</strong>
              <span>{gitStatus ? `${gitStatus.branch} - ${gitStatus.clean ? "clean" : `${gitStatus.files.length} changed`}` : "not loaded"}</span>
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
              <strong>Tool Marketplace</strong>
              <span>{connectorMarketplace ? `${connectorMarketplace.enabled_count} enabled · ${connectorMarketplace.planned_count} planned` : "not loaded"}</span>
            </div>
            <div className="connector-grid">
              {(connectorMarketplace?.connectors ?? []).map((connector) => (
                <article className={`connector-card connector-card--${connector.status}`} key={connector.id}>
                  <div className="connector-card__top">
                    <strong>{connector.name}</strong>
                    <span>{connector.status}</span>
                  </div>
                  <p>{connector.description}</p>
                  <div className="connector-card__meta">
                    <span>{connector.category}</span>
                    <span>{connector.privacy}</span>
                    {connector.setup_required && <span>setup needed</span>}
                  </div>
                  <div className="connector-card__tools">
                    {connector.capabilities.slice(0, 4).map((capability) => <span key={capability}>{capability}</span>)}
                  </div>
                </article>
              ))}
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

































