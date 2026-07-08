import type { ChatItem } from "../types";
import { DiffView } from "./DiffView";

type ApprovalItem = Extract<ChatItem, { kind: "approval" }>;

interface Props {
  item: ApprovalItem;
  onRespond: (id: string, approved: boolean) => void;
}

export function ApprovalCard({ item, onRespond }: Props) {
  const pending = item.status === "pending";

  return (
    <div className={`approval-card approval-card--${item.status}`}>
      <div className="approval-card__header">
        <span className="approval-card__title">
          {item.name === "run_shell" ? "Run shell command" : item.isNewFile ? "Create file" : "Edit file"}
        </span>
        <span className={`approval-card__badge approval-card__badge--${item.status}`}>{item.status}</span>
      </div>

      {item.name === "run_shell" ? (
        <>
          {item.risk && item.risk !== "normal" && (
            <div className="approval-card__risk">
              {item.risk.toUpperCase()}: {item.riskReason}
            </div>
          )}
          <pre className="approval-card__command">{item.command}</pre>
        </>
      ) : (
        <>
          <div className="approval-card__path">{item.path}</div>
          <DiffView diff={item.diff ?? ""} />
        </>
      )}

      {pending && (
        <div className="approval-card__actions">
          <button className="btn btn--approve" onClick={() => onRespond(item.id, true)}>
            Approve
          </button>
          <button className="btn btn--reject" onClick={() => onRespond(item.id, false)}>
            Reject
          </button>
        </div>
      )}
    </div>
  );
}
