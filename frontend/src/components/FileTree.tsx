import { useState } from "react";
import type { FileNode } from "../types";

interface TreeNodeProps {
  node: FileNode;
  depth: number;
  activePath: string | null;
  onOpenFile: (path: string) => void;
}

function TreeNode({ node, depth, activePath, onOpenFile }: TreeNodeProps) {
  const [expanded, setExpanded] = useState(depth === 0);

  if (node.type === "file") {
    const active = node.path === activePath;
    return (
      <button
        className={`tree-node tree-node--file${active ? " tree-node--active" : ""}`}
        style={{ paddingLeft: 10 + depth * 14 }}
        onClick={() => onOpenFile(node.path)}
        title={node.path}
      >
        {node.name}
      </button>
    );
  }

  return (
    <div>
      {depth > 0 && (
        <button
          className="tree-node tree-node--dir"
          style={{ paddingLeft: 10 + depth * 14 }}
          onClick={() => setExpanded((e) => !e)}
        >
          <span className="tree-node__arrow">{expanded ? "▾" : "▸"}</span> {node.name}
        </button>
      )}
      {expanded &&
        node.children?.map((child) => (
          <TreeNode
            key={child.path}
            node={child}
            depth={depth + 1}
            activePath={activePath}
            onOpenFile={onOpenFile}
          />
        ))}
    </div>
  );
}

interface FileTreeProps {
  root: FileNode | null;
  activePath: string | null;
  onOpenFile: (path: string) => void;
  onRefresh: () => void;
}

export function FileTree({ root, activePath, onOpenFile, onRefresh }: FileTreeProps) {
  return (
    <div className="file-tree">
      <div className="file-tree__title">
        <span>Workspace</span>
        <button className="file-tree__refresh" onClick={onRefresh} title="Refresh">
          ↻
        </button>
      </div>
      <div className="file-tree__nodes">
        {root ? (
          root.children && root.children.length > 0 ? (
            <TreeNode node={root} depth={0} activePath={activePath} onOpenFile={onOpenFile} />
          ) : (
            <div className="file-tree__empty">Workspace is empty.</div>
          )
        ) : (
          <div className="file-tree__empty">Loading…</div>
        )}
      </div>
    </div>
  );
}
