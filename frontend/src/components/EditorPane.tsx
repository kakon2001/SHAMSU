import Editor, { type OnMount } from "@monaco-editor/react";
import { useEffect, useRef, useState } from "react";
import type { EditorTab } from "../types";

const LANGUAGES: Record<string, string> = {
  ts: "typescript",
  tsx: "typescript",
  js: "javascript",
  jsx: "javascript",
  mjs: "javascript",
  cjs: "javascript",
  py: "python",
  json: "json",
  html: "html",
  htm: "html",
  css: "css",
  scss: "scss",
  md: "markdown",
  yml: "yaml",
  yaml: "yaml",
  xml: "xml",
  sh: "shell",
  ps1: "powershell",
  sql: "sql",
  rs: "rust",
  go: "go",
  java: "java",
  c: "c",
  h: "c",
  cpp: "cpp",
  cs: "csharp",
  toml: "ini",
  ini: "ini",
  txt: "plaintext",
};

function languageFor(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  return LANGUAGES[ext] ?? "plaintext";
}

interface Props {
  tabs: EditorTab[];
  activePath: string | null;
  onSelect: (path: string) => void;
  onClose: (path: string) => void;
  onChange: (path: string, content: string) => void;
  onSave: (path: string) => void;
}

export function EditorPane({ tabs, activePath, onSelect, onClose, onChange, onSave }: Props) {
  const active = tabs.find((t) => t.path === activePath) ?? null;
  const [dark, setDark] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);
  const saveRef = useRef<() => void>(() => {});
  saveRef.current = () => {
    if (activePath) onSave(activePath);
  };

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = (e: MediaQueryListEvent) => setDark(e.matches);
    mq.addEventListener("change", listener);
    return () => mq.removeEventListener("change", listener);
  }, []);

  const handleMount: OnMount = (editor, monaco) => {
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => saveRef.current());
  };

  return (
    <div className="editor-pane">
      <div className="editor-pane__tabs">
        {tabs.map((tab) => {
          const dirty = tab.content !== tab.savedContent;
          return (
            <div
              key={tab.path}
              className={`editor-tab${tab.path === activePath ? " editor-tab--active" : ""}`}
              onClick={() => onSelect(tab.path)}
              title={tab.path}
            >
              <span>
                {dirty ? "● " : ""}
                {tab.path.split("/").pop()}
              </span>
              <button
                className="editor-tab__close"
                onClick={(e) => {
                  e.stopPropagation();
                  onClose(tab.path);
                }}
              >
                ×
              </button>
            </div>
          );
        })}
        {active && (
          <button
            className="editor-pane__save"
            disabled={active.content === active.savedContent}
            onClick={() => saveRef.current()}
            title="Save (Ctrl+S)"
          >
            Save
          </button>
        )}
      </div>
      <div className="editor-pane__body">
        {active ? (
          <Editor
            path={active.path}
            language={languageFor(active.path)}
            value={active.content}
            theme={dark ? "vs-dark" : "light"}
            onChange={(value) => onChange(active.path, value ?? "")}
            onMount={handleMount}
            options={{
              minimap: { enabled: false },
              fontSize: 13,
              automaticLayout: true,
              scrollBeyondLastLine: false,
            }}
          />
        ) : (
          <div className="editor-pane__empty">
            Open a file from the tree to view or edit it. Agent edits show up here after you approve
            them.
          </div>
        )}
      </div>
    </div>
  );
}
