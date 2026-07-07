import type { ChatItem } from "../types";

type ToolItem = Extract<ChatItem, { kind: "tool" }>;

const STATUS_ICON: Record<ToolItem["status"], string> = {
  running: "⏳",
  done: "✓",
  error: "✗",
};

function argsSummary(args: Record<string, unknown>): string {
  const parts = Object.entries(args)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${k}: ${String(v).slice(0, 80)}`);
  return parts.join(", ");
}

export function ToolChip({ item }: { item: ToolItem }) {
  return (
    <details className={`tool-chip tool-chip--${item.status}`}>
      <summary>
        <span className="tool-chip__icon">{STATUS_ICON[item.status]}</span>
        <span className="tool-chip__name">{item.name}</span>
        <span className="tool-chip__args">{argsSummary(item.args)}</span>
      </summary>
      {item.preview && <pre className="tool-chip__preview">{item.preview}</pre>}
    </details>
  );
}
