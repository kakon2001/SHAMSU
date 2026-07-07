import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { getFileContent, getFileTree, saveFileContent } from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { EditorPane } from "./components/EditorPane";
import { FileTree } from "./components/FileTree";
import { useAgent } from "./hooks/useAgent";
import type { EditorTab, FileNode } from "./types";

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

  const tabsRef = useRef(tabs);
  tabsRef.current = tabs;

  const refreshFileTree = useCallback(() => {
    getFileTree()
      .then(setFileTree)
      .catch((err: Error) => setNotice(err.message));
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
    refreshFileTree();
    reloadCleanTabs();
  }, [refreshFileTree, reloadCleanTabs]);

  const { items, connected, busy, sendChat, respondApproval, stop, reset } =
    useAgent(handleFilesChanged);

  useEffect(() => {
    refreshFileTree();
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
          refreshFileTree();
        })
        .catch((err: Error) => setNotice(err.message));
    },
    [refreshFileTree],
  );

  return (
    <div className="app">
      <header className="app__header">
        <span className="app__title">Local Coding Agent</span>
        <span className={`app__status app__status--${connected ? "on" : "off"}`}>
          {connected ? "connected" : "disconnected"}
        </span>
        <div className="app__header-right">
          <button className="app__new-chat" onClick={reset} disabled={busy}>
            New chat
          </button>
        </div>
      </header>

      {notice && (
        <div className="app__notice" onClick={() => setNotice(null)} title="Click to dismiss">
          {notice}
        </div>
      )}

      <div className="app__body">
        <FileTree
          root={fileTree}
          activePath={activePath}
          onOpenFile={openFile}
          onRefresh={refreshFileTree}
        />
        <EditorPane
          tabs={tabs}
          activePath={activePath}
          onSelect={setActivePath}
          onClose={closeTab}
          onChange={changeTab}
          onSave={saveTab}
        />
        <ChatPanel
          items={items}
          busy={busy}
          connected={connected}
          files={flattenFiles(fileTree)}
          activePath={activePath}
          onSend={sendChat}
          onStop={stop}
          onRespondApproval={respondApproval}
        />
      </div>
    </div>
  );
}

export default App;
