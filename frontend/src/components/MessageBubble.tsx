interface Props {
  role: "user" | "assistant";
  content: string;
  files?: string[];
}

export function MessageBubble({ role, content, files }: Props) {
  return (
    <div className={`message message--${role}`}>
      <div className="message__role">{role === "user" ? "You" : "Agent"}</div>
      {files && files.length > 0 && (
        <div className="message__files">
          {files.map((path) => (
            <span key={path} className="context-chip context-chip--static" title={path}>
              {path.split("/").pop()}
            </span>
          ))}
        </div>
      )}
      <div className="message__content">{content}</div>
    </div>
  );
}
