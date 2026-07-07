interface Props {
  diff: string;
}

function lineClass(line: string): string {
  if (line.startsWith("+++") || line.startsWith("---")) return "diff-line diff-line--meta";
  if (line.startsWith("@@")) return "diff-line diff-line--hunk";
  if (line.startsWith("+")) return "diff-line diff-line--add";
  if (line.startsWith("-")) return "diff-line diff-line--remove";
  return "diff-line";
}

export function DiffView({ diff }: Props) {
  return (
    <pre className="diff-view">
      {diff.split("\n").map((line, i) => (
        <div key={i} className={lineClass(line)}>
          {line || " "}
        </div>
      ))}
    </pre>
  );
}
