interface Props {
  role: "user" | "assistant";
  content: string;
  files?: string[];
  fileLabels?: Record<string, string>;
}

function attachmentLabel(path: string, labels?: Record<string, string>): string {
  if (labels?.[path]) return `Uploaded: ${labels[path]}`;
  if (path.startsWith("uploads/")) return `Uploaded: ${path.split("/").pop()}`;
  return path.split("/").pop() ?? path;
}

export function MessageBubble({ role, content, files, fileLabels }: Props) {
  return (
    <div className={`message message--${role}`}>
      <div className="message__role">{role === "user" ? "You" : "Agent"}</div>
      {files && files.length > 0 && (
        <div className="message__files">
          {files.map((path) => (
            <span key={path} className="context-chip context-chip--static" title={path}>
              {attachmentLabel(path, fileLabels)}
            </span>
          ))}
        </div>
      )}
      <div className="message__content">{content}</div>
    </div>
  );
}
