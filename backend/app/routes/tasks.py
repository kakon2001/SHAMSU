from __future__ import annotations

import json
import re
from typing import Any

import ollama
from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import model_registry
from ..agent import tools
from ..config import settings
from .preview import PreviewStartRequest, start_preview

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

ALLOWED_GENERATED_SUFFIXES = {".html", ".css", ".js", ".py", ".c", ".h", ".md", ".txt", ".json"}
MAX_GENERATED_FILES = 8
MAX_GENERATED_CHARS = 200_000
MAX_REPAIR_ATTEMPTS = 2


class TaskPlanRequest(BaseModel):
    prompt: str


class TaskPlanResponse(BaseModel):
    goal: str
    mode: str
    steps: list[str]
    suggested_files: list[dict[str, str]]
    verify_commands: list[str]
    notes: list[str]
    requirements_analysis: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    stack: list[str] = Field(default_factory=list)
    file_plan: list[str] = Field(default_factory=list)
    workflow_summary: str = ""


class TaskRunRequest(BaseModel):
    prompt: str
    preview: bool = True
    overwrite: bool = True


class TaskRunStep(BaseModel):
    name: str
    status: str
    detail: str


class ReliabilityReport(BaseModel):
    phases: list[str]
    verification_feedback: list[str]
    repair_attempts: int
    final_status: str
    next_action: str


class TaskRunResponse(BaseModel):
    goal: str
    mode: str
    ok: bool
    created_files: list[str]
    preview_url: str | None
    steps: list[TaskRunStep]
    notes: list[str]
    requirements_analysis: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    stack: list[str] = Field(default_factory=list)
    file_plan: list[str] = Field(default_factory=list)
    workflow_summary: str = ""
    reliability: ReliabilityReport | None = None


@router.post("/plan", response_model=TaskPlanResponse)
async def plan_task(body: TaskPlanRequest) -> TaskPlanResponse:
    return build_plan(body.prompt)


@router.post("/run", response_model=TaskRunResponse)
async def run_task(body: TaskRunRequest) -> TaskRunResponse:
    """Run the autonomous build loop: plan -> write -> verify -> preview.

    Known safe templates run deterministically. Unknown build prompts fall back to
    a strict JSON file generator powered by the selected local Ollama model.
    """
    plan = build_plan(body.prompt)
    steps: list[TaskRunStep] = [TaskRunStep(name="plan", status="ok", detail=f"Selected {plan.mode} workflow.")]
    notes = list(plan.notes)

    if not plan.suggested_files and _looks_like_build_request(body.prompt):
        generated = await _generate_json_file_plan(body.prompt, steps)
        if generated:
            plan = _make_generated_plan(body.prompt, generated)
            notes = list(plan.notes)

    created_files = _write_suggested_files(plan, body.overwrite, steps)
    if not plan.suggested_files:
        steps.append(
            TaskRunStep(
                name="write",
                status="skipped",
                detail="No deterministic template or valid JSON file plan was available for this prompt.",
            )
        )
        return TaskRunResponse(
            goal=plan.goal,
            mode=plan.mode,
            ok=False,
            created_files=created_files,
            preview_url=None,
            steps=steps,
            notes=notes + ["Try a more concrete build prompt, or use chat mode to ask for code/instructions."],
            requirements_analysis=plan.requirements_analysis,
            clarification_questions=plan.clarification_questions,
            stack=plan.stack,
            file_plan=plan.file_plan,
            workflow_summary=plan.workflow_summary,
            reliability=_build_reliability_report(steps, created_files, False),
        )

    verify_ok, plan, created_files = await _verify_and_repair_loop(body.prompt, plan, created_files, steps)
    preview_url = await _maybe_start_preview(body.preview, created_files, verify_ok, steps)

    return TaskRunResponse(
        goal=plan.goal,
        mode=plan.mode,
        ok=verify_ok,
        created_files=created_files,
        preview_url=preview_url,
        steps=steps,
        notes=notes,
        requirements_analysis=plan.requirements_analysis,
        clarification_questions=plan.clarification_questions,
        stack=plan.stack,
        file_plan=plan.file_plan,
        workflow_summary=plan.workflow_summary,
        reliability=_build_reliability_report(steps, created_files, verify_ok),
    )



def _build_reliability_report(steps: list[TaskRunStep], created_files: list[str], ok: bool) -> ReliabilityReport:
    phase_order = ["plan", "generate", "validate", "write", "verify", "feedback", "repair", "repair-generate", "preview"]
    phases: list[str] = []
    for phase in phase_order:
        phase_steps = [step for step in steps if step.name == phase]
        if not phase_steps:
            continue
        status = _phase_status(phase_steps)
        detail = phase_steps[-1].detail
        phases.append(f"{phase}: {status} - {detail}")

    feedback = [step.detail for step in steps if step.name == "feedback"]
    repair_attempts = len([step for step in steps if step.name in {"repair", "repair-generate"} and step.status != "skipped"])
    if ok:
        next_action = "Open the preview or inspect the generated files, then ask SHAMSU for changes if needed."
        final_status = "verified"
    elif created_files:
        next_action = "Read the verification feedback, repair the smallest affected file or block, then run verification again."
        final_status = "needs-repair"
    else:
        next_action = "Clarify the request or ask for a smaller first version so SHAMSU can create a valid file plan."
        final_status = "blocked"
    return ReliabilityReport(
        phases=phases,
        verification_feedback=feedback,
        repair_attempts=repair_attempts,
        final_status=final_status,
        next_action=next_action,
    )


def _phase_status(steps: list[TaskRunStep]) -> str:
    statuses = {step.status for step in steps}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    if "ok" in statuses:
        return "ok"
    if "skipped" in statuses:
        return "skipped"
    return steps[-1].status
def build_plan(prompt: str) -> TaskPlanResponse:
    prompt = prompt.strip()
    lower = prompt.lower()
    if "brick" in lower and ("breaker" in lower or "game" in lower):
        return _brick_breaker_plan(prompt)
    if "snake" in lower and "game" in lower:
        return _snake_game_plan(prompt)
    if "pong" in lower and "game" in lower:
        return _pong_game_plan(prompt)
    if ("tic" in lower and "toe" in lower) or "tic-tac-toe" in lower:
        return _tic_tac_toe_plan(prompt)
    if "quiz" in lower and ("app" in lower or "game" in lower):
        return _quiz_app_plan(prompt)
    if "todo" in lower or "to-do" in lower:
        return _todo_app_plan(prompt)
    if "calculator" in lower and ("app" in lower or "tool" in lower or "program" in lower):
        return _calculator_app_plan(prompt)
    if "bouncing" in lower and "ball" in lower:
        return _bouncing_ball_plan(prompt)
    if "single html" in lower and ("crm" in lower or "management system" in lower or "management app" in lower):
        return _management_system_plan(prompt)
    if ("crm" in lower or "management system" in lower or "management app" in lower) and _wants_database_backing(lower):
        return _database_backed_management_app_plan(prompt)
    if "crm" in lower or "management system" in lower or "management app" in lower:
        return _multi_file_management_app_plan(prompt)
    if _looks_like_website_request(lower):
        return _website_prototype_plan(prompt)
    if _looks_like_system_request(lower):
        return _starter_system_plan(prompt)
    if any(word in lower for word in ["bug", "fix", "error", "traceback", "failing"]):
        return _bugfix_plan(prompt)
    if any(word in lower for word in ["large file", "100000", "100,000", "huge file"]):
        return _large_file_plan(prompt)
    if _looks_like_build_request(prompt):
        return _generated_task_plan(prompt)
    return _general_plan(prompt)


def _looks_like_build_request(prompt: str) -> bool:
    text = prompt.lower()
    return bool(re.search(r"\b(make|build|create|generate|write|develop|implement)\b", text)) and bool(
        re.search(r"\b(game|app|application|website|web page|html|system|tool|program|project|calculator|todo|quiz|crm|management|dashboard|portal|inventory|student|library|os|operating system)\b", text)
    )


def _looks_like_website_request(text: str) -> bool:
    return bool(re.search(r"\b(website|web page|landing page|site)\b", text))


def _looks_like_system_request(text: str) -> bool:
    return bool(re.search(r"\b(system|dashboard|portal|application|app)\b", text)) and not any(
        word in text for word in ["game", "calculator", "todo", "quiz"]
    )


def _wants_database_backing(text: str) -> bool:
    return bool(re.search(r"\b(database|db|sqlite|backend|api|full[- ]?stack|persistent|server)\b", text))


def _slug_from_prompt(prompt: str, fallback: str, suffix: str) -> str:
    text = prompt.lower()
    text = re.sub(r"\b(make|build|create|generate|write|develop|implement|a|an|the|for|me|one|single|simple|website|web|page|site|system|app|application|dashboard|portal)\b", " ", text)
    words = re.findall(r"[a-z0-9]+", text)[:5]
    stem = "_".join(words) if words else fallback
    if not stem.endswith(suffix):
        stem = f"{stem}_{suffix}"
    return f"{stem}.html"


async def _generate_json_file_plan(prompt: str, steps: list[TaskRunStep]) -> list[dict[str, str]]:
    system = (
        "You generate small coding project files. Return ONLY valid JSON with this exact shape: "
        "{\"files\":[{\"path\":\"relative/path.ext\",\"content\":\"full file content\"}],\"notes\":[\"...\"]}. "
        "Use relative paths only. Do not use markdown fences. Prefer one runnable HTML file for browser games, "
        "or a small Python/C project when requested. Keep the project compact."
    )
    user = f"Create the files for this request: {prompt}"
    try:
        response = await ollama.AsyncClient(host=settings.ollama_host).chat(
            model=model_registry.get_current_model(),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            stream=False,
            think=False,
            options={"temperature": 0.1, "num_ctx": settings.model_num_ctx, "num_predict": settings.max_model_output_tokens},
        )
    except Exception as exc:
        steps.append(TaskRunStep(name="generate", status="error", detail=f"Local model generation failed: {exc}"))
        return []

    content = (response.get("message") or {}).get("content") or ""
    payload = _extract_json_payload(content)
    if not payload:
        steps.append(TaskRunStep(name="generate", status="error", detail="Model did not return valid JSON."))
        return []
    files = _validated_generated_files(payload, steps)
    if files:
        steps.append(TaskRunStep(name="generate", status="ok", detail=f"Generated {len(files)} file(s) from JSON plan."))
    return files


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _validated_generated_files(payload: dict[str, Any], steps: list[TaskRunStep]) -> list[dict[str, str]]:
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        steps.append(TaskRunStep(name="validate", status="error", detail="JSON payload does not contain a files list."))
        return []
    files: list[dict[str, str]] = []
    total_chars = 0
    for raw in raw_files[:MAX_GENERATED_FILES]:
        if not isinstance(raw, dict):
            continue
        path = str(raw.get("path") or "").strip().replace("\\", "/")
        content = raw.get("content")
        if not path or not isinstance(content, str):
            continue
        suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if suffix not in ALLOWED_GENERATED_SUFFIXES:
            steps.append(TaskRunStep(name="validate", status="skipped", detail=f"Skipped unsupported generated file type: {path}"))
            continue
        try:
            tools.resolve_in_workspace(path)
        except ValueError as exc:
            steps.append(TaskRunStep(name="validate", status="skipped", detail=str(exc)))
            continue
        total_chars += len(content)
        if total_chars > MAX_GENERATED_CHARS:
            steps.append(TaskRunStep(name="validate", status="error", detail="Generated project exceeded size limit."))
            break
        files.append({"path": path, "content": content})
    return files


def _make_generated_plan(prompt: str, files: list[dict[str, str]], mode: str = "json-generated-task", note: str | None = None) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt.strip(),
        mode=mode,
        steps=[
            "Generate a JSON file plan.",
            "Write generated files.",
            "Run lightweight verification.",
            "Repair from verification feedback if needed.",
            "Open preview for HTML output.",
        ],
        suggested_files=files,
        verify_commands=[],
        notes=[note or "Files were generated by the local model in strict JSON format and validated before writing."],
    )


async def _verify_and_repair_loop(
    prompt: str,
    plan: TaskPlanResponse,
    created_files: list[str],
    steps: list[TaskRunStep],
) -> tuple[bool, TaskPlanResponse, list[str]]:
    verify_ok = _verify_created_files(plan, created_files, steps)
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        if verify_ok:
            break
        feedback = _verification_feedback(created_files)
        steps.append(TaskRunStep(name="feedback", status="warning", detail="; ".join(feedback[:4]) or "Verification failed without detailed feedback."))

        repaired_by_rule = _repair_created_files(created_files, steps)
        if repaired_by_rule:
            verify_ok = _verify_created_files(plan, created_files, steps)
            if verify_ok:
                break
            feedback = _verification_feedback(created_files)

        if plan.mode not in {"json-generated-task", "json-repaired-task"}:
            break

        repaired_files = await _generate_json_repair_plan(prompt, plan.suggested_files, feedback, steps, attempt)
        if not repaired_files:
            break
        plan = _make_generated_plan(
            prompt,
            repaired_files,
            mode="json-repaired-task",
            note=f"Files were repaired by the local model using verification feedback on attempt {attempt}.",
        )
        created_files = _write_suggested_files(plan, overwrite=True, steps=steps)
        verify_ok = _verify_created_files(plan, created_files, steps)
    return verify_ok, plan, created_files


async def _generate_json_repair_plan(
    prompt: str,
    current_files: list[dict[str, str]],
    feedback: list[str],
    steps: list[TaskRunStep],
    attempt: int,
) -> list[dict[str, str]]:
    system = (
        "You repair small coding project files. Return ONLY valid JSON with this exact shape: "
        "{\"files\":[{\"path\":\"relative/path.ext\",\"content\":\"full corrected file content\"}],\"notes\":[\"...\"]}. "
        "Use relative paths only. Return complete corrected file contents, not patches. Keep the project compact. "
        "Fix the verification errors while preserving the user's requested app."
    )
    file_context = []
    for item in current_files[:MAX_GENERATED_FILES]:
        content = str(item.get("content") or "")
        file_context.append({"path": item.get("path"), "content": _truncate_for_model(content, 12000)})
    user = json.dumps(
        {
            "original_prompt": prompt,
            "verification_feedback": feedback,
            "current_files": file_context,
        },
        ensure_ascii=False,
    )
    try:
        response = await ollama.AsyncClient(host=settings.ollama_host).chat(
            model=model_registry.get_current_model(),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            stream=False,
            think=False,
            options={"temperature": 0.05, "num_ctx": settings.model_num_ctx, "num_predict": settings.max_model_output_tokens},
        )
    except Exception as exc:
        steps.append(TaskRunStep(name="repair-generate", status="error", detail=f"Repair attempt {attempt} failed: {exc}"))
        return []

    content = (response.get("message") or {}).get("content") or ""
    payload = _extract_json_payload(content)
    if not payload:
        steps.append(TaskRunStep(name="repair-generate", status="error", detail=f"Repair attempt {attempt} did not return valid JSON."))
        return []
    files = _validated_generated_files(payload, steps)
    if files:
        steps.append(TaskRunStep(name="repair-generate", status="ok", detail=f"Repair attempt {attempt} generated {len(files)} corrected file(s)."))
    return files


def _truncate_for_model(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    half = limit // 2
    return content[:half] + "\n... truncated for repair feedback ...\n" + content[-half:]

def _write_suggested_files(plan: TaskPlanResponse, overwrite: bool, steps: list[TaskRunStep]) -> list[str]:
    created_files: list[str] = []
    for item in plan.suggested_files:
        path = item["path"]
        content = item["content"]
        existing = tools.read_file(path)
        if not overwrite and not existing.startswith("Error:"):
            steps.append(TaskRunStep(name="write", status="skipped", detail=f"{path} already exists."))
            continue
        result = tools.write_file(path, content)
        created_files.append(path)
        status = "error" if result.startswith("Error") else "ok"
        steps.append(TaskRunStep(name="write", status=status, detail=result))
    return created_files


def _verify_created_files(plan: TaskPlanResponse, created_files: list[str], steps: list[TaskRunStep]) -> bool:
    if not created_files:
        steps.append(TaskRunStep(name="verify", status="error", detail="No files were created."))
        return False
    ok = True
    for path in created_files:
        content = tools.read_file(path)
        if content.startswith("Error:") or not content.strip():
            steps.append(TaskRunStep(name="verify", status="error", detail=f"{path} is missing or empty."))
            ok = False
            continue
        if path.endswith(".py"):
            try:
                compile(content, path, "exec")
                steps.append(TaskRunStep(name="verify", status="ok", detail=f"{path} Python syntax verified."))
            except SyntaxError as exc:
                steps.append(TaskRunStep(name="verify", status="error", detail=f"{path} syntax error: {exc}"))
                ok = False
        elif path.endswith(".html"):
            lower = content.lower()
            html_ok = "<html" in lower or "<canvas" in lower or "<script" in lower or "<main" in lower
            script_feedback = _basic_script_feedback(content) if "<script" in lower else None
            if script_feedback:
                steps.append(TaskRunStep(name="verify", status="error", detail=f"{path} {script_feedback}"))
                ok = False
            else:
                steps.append(TaskRunStep(name="verify", status="ok" if html_ok else "warning", detail=f"{path} HTML structure checked."))
                ok = ok and html_ok
        elif path.endswith(".c"):
            c_ok = "main(" in content or "main (" in content
            steps.append(TaskRunStep(name="verify", status="ok" if c_ok else "warning", detail=f"{path} C entry point checked."))
        else:
            steps.append(TaskRunStep(name="verify", status="ok", detail=f"{path} written and readable."))
    return ok


def _verification_feedback(created_files: list[str]) -> list[str]:
    if not created_files:
        return ["No files were created."]
    feedback: list[str] = []
    for path in created_files:
        content = tools.read_file(path)
        lower = content.lower()
        if content.startswith("Error:") or not content.strip():
            feedback.append(f"{path}: file is missing or empty.")
            continue
        if path.endswith(".py"):
            try:
                compile(content, path, "exec")
            except SyntaxError as exc:
                feedback.append(f"{path}: Python syntax error: {exc}.")
        elif path.endswith(".html"):
            if not any(token in lower for token in ["<html", "<body", "<canvas", "<script", "<main"]):
                feedback.append(f"{path}: HTML appears to have no runnable or visible document structure.")
            if "<script" in lower:
                script_feedback = _basic_script_feedback(content)
                if script_feedback:
                    feedback.append(f"{path}: {script_feedback}")
        elif path.endswith(".c") and "main(" not in content and "main (" not in content:
            feedback.append(f"{path}: C file has no main function.")
    return feedback


def _basic_script_feedback(content: str) -> str | None:
    opens = content.count("{")
    closes = content.count("}")
    if opens != closes:
        return f"JavaScript block braces look unbalanced ({opens} opening, {closes} closing)."
    if "getContext('2d')" in content and "requestAnimationFrame" not in content and "setInterval" not in content:
        return "Canvas code has no animation loop such as requestAnimationFrame or setInterval."
    return None

def _repair_created_files(created_files: list[str], steps: list[TaskRunStep]) -> bool:
    repaired = False
    for path in created_files:
        content = tools.read_file(path)
        if content.startswith("Error:") or not content.strip():
            continue
        if path.endswith(".html"):
            fixed = _repair_html_content(path, content)
            if fixed is not None:
                result = tools.write_file(path, fixed)
                status = "error" if result.startswith("Error") else "ok"
                steps.append(TaskRunStep(name="repair", status=status, detail=f"Repaired HTML shell for {path}." if status == "ok" else result))
                repaired = repaired or status == "ok"
    if not repaired:
        steps.append(TaskRunStep(name="repair", status="skipped", detail="No automatic safe repair was available."))
    return repaired


def _repair_html_content(path: str, content: str) -> str | None:
    lower = content.lower()
    if "<html" in lower and ("<body" in lower or "<canvas" in lower or "<script" in lower):
        return None
    title = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title() or "SHAMSU Generated App"
    body = content.strip()
    if "<script" not in lower and re.search(r"\b(function|const|let|var|document\.)\b", body):
        body = '<canvas id="game" width="640" height="420"></canvas>\n<script>\n' + body + '\n</script>'
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{title}</title>
  <style>body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#111827;color:#f9fafb;font-family:Arial,sans-serif}}main{{width:min(900px,92vw)}}canvas{{max-width:100%;background:#f8fafc;border:2px solid #334155}}</style>
</head>
<body>
  <main>
{body}
  </main>
</body>
</html>
"""

async def _maybe_start_preview(preview: bool, created_files: list[str], verify_ok: bool, steps: list[TaskRunStep]) -> str | None:
    if not preview or not verify_ok:
        return None
    html_file = next((path for path in created_files if path.endswith(".html")), None)
    if not html_file:
        return None
    preview_state = await start_preview(PreviewStartRequest(path=html_file, port=9000))
    steps.append(TaskRunStep(name="preview", status="ok", detail=f"Preview ready at {preview_state.url}"))
    return preview_state.url


def _website_prototype_plan(prompt: str) -> TaskPlanResponse:
    project = _slug_from_prompt(prompt, "shamsu", "website").removesuffix(".html")
    title = project.replace("_", " ").title()
    base = project
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <header class="hero">
    <nav><strong>__TITLE__</strong><a href="#features">Features</a><a href="#contact">Contact</a></nav>
    <section><h1>__TITLE__</h1><p>A multi-file website prototype generated by SHAMSU with separate HTML, CSS, JavaScript, and workflow documentation.</p><button id="ctaButton">Get started</button></section>
  </header>
  <main>
    <section id="features" class="grid">
      <article><h2>Responsive Layout</h2><p>Works on desktop and mobile.</p></article>
      <article><h2>Editable Sections</h2><p>Content is split into clean files for easier future edits.</p></article>
      <article><h2>Verified Preview</h2><p>SHAMSU checks the generated files and opens this page locally.</p></article>
    </section>
    <section id="contact" class="panel"><h2>Contact</h2><form id="contactForm"><input placeholder="Name" required /><input placeholder="Email" required /><textarea placeholder="Message" required></textarea><button type="submit">Send</button></form><p id="message"></p></section>
  </main>
  <script src="app.js"></script>
</body>
</html>
""".replace("__TITLE__", title)
    css = """body{margin:0;font-family:Arial,sans-serif;background:#f5f7fb;color:#172033}.hero{background:#123047;color:white;padding:28px 36px 58px}nav{display:flex;gap:18px;align-items:center;flex-wrap:wrap}nav strong{margin-right:auto;font-size:22px}nav a{color:#d9eefb;text-decoration:none}.hero section{max-width:820px;margin-top:48px}.hero h1{font-size:44px;line-height:1.08;margin:0 0 12px}.hero p{font-size:18px;color:#d9eefb;line-height:1.6}button{border:0;border-radius:6px;background:#0f766e;color:white;padding:12px 16px;font-weight:800;cursor:pointer}main{max-width:1120px;margin:0 auto;padding:28px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.grid article,.panel{background:white;border:1px solid #dbe4ef;border-radius:8px;padding:20px}.panel{margin-top:28px}form{display:grid;grid-template-columns:1fr 1fr;gap:12px}input,textarea{box-sizing:border-box;width:100%;border:1px solid #cbd5e1;border-radius:6px;padding:11px;font:inherit}textarea{grid-column:span 2;min-height:110px}#message{color:#0f766e;font-weight:700}@media(max-width:780px){.hero h1{font-size:34px}.grid,form{grid-template-columns:1fr}textarea{grid-column:auto}}"""
    js = """document.getElementById('ctaButton').addEventListener('click', () => document.getElementById('contact').scrollIntoView({behavior:'smooth'}));
document.getElementById('contactForm').addEventListener('submit', (event) => {
  event.preventDefault();
  document.getElementById('message').textContent = 'Demo message saved locally. Add a backend API for production.';
  event.target.reset();
});
"""
    workflow = f"""# {title} Workflow

## Requirement Analysis
- Build a presentable website from the prompt: {prompt}
- Keep the first version local, previewable, and easy to edit.

## Clarification Questions
- What exact brand colors, copy, and pages should be used next?
- Should this stay static or become a React/full-stack site?

## Chosen Stack
- HTML for structure
- CSS for responsive styling
- JavaScript for interaction
- Python preview server for local demo

## File Plan
- index.html: page structure and sections
- styles.css: layout and visual design
- app.js: contact form and scroll interaction
- WORKFLOW.md: explanation for faculty/demo

## Verification
- Open index.html through the preview URL.
- Submit the contact form and confirm the success message appears.
"""
    return TaskPlanResponse(
        goal=prompt,
        mode="website-generator",
        steps=["Analyze requirements.", "Ask/record clarification questions when details are missing.", "Choose a simple static website stack.", "Create a multi-file project.", "Verify generated files.", "Start preview server and explain workflow."],
        suggested_files=[
            {"path": f"{base}/index.html", "content": html},
            {"path": f"{base}/styles.css", "content": css},
            {"path": f"{base}/app.js", "content": js},
            {"path": f"{base}/WORKFLOW.md", "content": workflow},
        ],
        verify_commands=[f"Open http://127.0.0.1:9000/{base}/index.html"],
        notes=["SHAMSU created a Claude-like multi-file website starter. Ask for a React/full-stack upgrade when you want backend, routing, auth, or database."],
        requirements_analysis=["The prompt requests a website that should be visible in a browser.", "A local static prototype is the safest first build for demo and iteration."],
        clarification_questions=["What brand/content should replace the generated placeholder copy?", "Should the next version be static, React, or full-stack?"],
        stack=["HTML", "CSS", "JavaScript", "Local preview server"],
        file_plan=[f"{base}/index.html", f"{base}/styles.css", f"{base}/app.js", f"{base}/WORKFLOW.md"],
        workflow_summary="prompt -> requirement analysis -> clarification questions -> stack choice -> file plan -> multi-file creation -> verification -> preview -> explanation",
    )


def _starter_system_plan(prompt: str) -> TaskPlanResponse:
    project = _slug_from_prompt(prompt, "starter", "system").removesuffix(".html")
    title = project.replace("_", " ").title()
    base = project
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <div class="shell">
    <aside><h1>__TITLE__</h1><button data-view="overview">Overview</button><button data-view="records">Records</button><button data-view="workflow">Workflow</button></aside>
    <main>
      <section id="overview"><h2>Overview</h2><div class="metrics"><article><span>Total Records</span><strong id="totalCount">0</strong></article><article><span>Open Items</span><strong id="openCount">0</strong></article><article><span>Done</span><strong id="doneCount">0</strong></article></div></section>
      <section id="records"><h2>Records</h2><div class="toolbar"><input id="recordInput" placeholder="Add a record" /><button id="addButton">Add</button></div><table><thead><tr><th>Name</th><th>Status</th><th>Actions</th></tr></thead><tbody id="recordRows"></tbody></table></section>
      <section id="workflow" class="hidden"><h2>Workflow</h2><ol><li>Capture requirements.</li><li>Create starter UI.</li><li>Verify generated files.</li><li>Expand into backend/database/auth when requested.</li></ol></section>
    </main>
  </div>
  <script src="app.js"></script>
</body>
</html>
""".replace("__TITLE__", title)
    css = """body{margin:0;font-family:Arial,sans-serif;background:#eef3f8;color:#172033}.shell{display:grid;grid-template-columns:240px 1fr;min-height:100vh}aside{background:#102a43;color:white;padding:22px}aside h1{font-size:24px}aside button{display:block;width:100%;text-align:left;margin:8px 0;padding:10px;border:0;border-radius:6px;background:#1d4ed8;color:white;cursor:pointer}main{padding:24px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metrics article,section{background:white;border:1px solid #d7e0ea;border-radius:8px;padding:18px;margin-bottom:18px}.metrics strong{display:block;font-size:30px;color:#0f766e}.toolbar{display:flex;gap:10px;margin:18px 0}.toolbar input{flex:1;border:1px solid #cbd5e1;border-radius:6px;padding:10px;font:inherit}button{background:#7c3aed;color:white;border:0;border-radius:6px;padding:10px 14px;font-weight:800;cursor:pointer}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:11px}.status{padding:4px 8px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:12px;font-weight:700}.hidden{display:none}@media(max-width:820px){.shell{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}aside button{display:inline-block;width:auto;margin-right:8px}}"""
    js = """const storageKey = 'shamsu-system-records';
let records = JSON.parse(localStorage.getItem(storageKey) || 'null') || [{name:'Demo setup',status:'Open'},{name:'Faculty review',status:'Done'}];
function save(){ localStorage.setItem(storageKey, JSON.stringify(records)); }
function showView(id){ ['overview','records','workflow'].forEach((name) => document.getElementById(name).classList.toggle('hidden', name !== id)); }
function addRecord(){ const input = document.getElementById('recordInput'); const name = input.value.trim(); if(!name) return; records.push({name,status:'Open'}); input.value=''; save(); renderRecords(); }
function toggleRecord(index){ records[index].status = records[index].status === 'Open' ? 'Done' : 'Open'; save(); renderRecords(); }
function deleteRecord(index){ records.splice(index,1); save(); renderRecords(); }
function renderRecords(){ document.getElementById('recordRows').innerHTML = records.map((record,index) => `<tr><td>${record.name}</td><td><span class="status">${record.status}</span></td><td><button onclick="toggleRecord(${index})">Toggle</button> <button onclick="deleteRecord(${index})">Delete</button></td></tr>`).join(''); document.getElementById('totalCount').textContent = records.length; document.getElementById('openCount').textContent = records.filter((r) => r.status === 'Open').length; document.getElementById('doneCount').textContent = records.filter((r) => r.status === 'Done').length; }
document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => showView(button.dataset.view)));
document.getElementById('addButton').addEventListener('click', addRecord);
renderRecords();
"""
    workflow = f"""# {title} Workflow

## Requirement Analysis
- Build a local management/system prototype from the prompt: {prompt}
- Include a dashboard, records area, workflow explanation, and persistent local state.

## Clarification Questions
- Which user roles, data fields, and reports are required?
- Should the next iteration use FastAPI/MySQL authentication and APIs?

## Chosen Stack
- HTML/CSS/JavaScript for the first local UI
- localStorage for demo persistence
- Future upgrade path: React + FastAPI + MySQL

## File Plan
- index.html: dashboard structure
- styles.css: UI layout and responsive styling
- app.js: CRUD-style record behavior
- WORKFLOW.md: explanation and next implementation steps

## Verification
- Open index.html through the preview URL.
- Add, toggle, and delete a record.
- Refresh and confirm records persist locally.
"""
    return TaskPlanResponse(
        goal=prompt,
        mode="system-prototype-generator",
        steps=["Analyze requirements.", "Ask/record clarification questions when details are missing.", "Choose a local starter stack.", "Create a multi-file project.", "Verify generated files.", "Start preview server and explain workflow."],
        suggested_files=[
            {"path": f"{base}/index.html", "content": html},
            {"path": f"{base}/styles.css", "content": css},
            {"path": f"{base}/app.js", "content": js},
            {"path": f"{base}/WORKFLOW.md", "content": workflow},
        ],
        verify_commands=[f"Open http://127.0.0.1:9000/{base}/index.html"],
        notes=["SHAMSU created a Claude-like multi-file system starter. Ask for the production upgrade to add FastAPI routes, MySQL tables, authentication, and tests."],
        requirements_analysis=["The prompt requests an application/system, so SHAMSU builds a working dashboard first.", "A multi-file local UI keeps the first version testable while leaving a clear full-stack path."],
        clarification_questions=["Which exact fields, roles, and reports should the system support?", "Should SHAMSU upgrade this to React/FastAPI/MySQL next?"],
        stack=["HTML", "CSS", "JavaScript", "localStorage", "Local preview server"],
        file_plan=[f"{base}/index.html", f"{base}/styles.css", f"{base}/app.js", f"{base}/WORKFLOW.md"],
        workflow_summary="prompt -> requirement analysis -> clarification questions -> stack choice -> file plan -> multi-file creation -> verification -> preview -> explanation",
    )


def _database_backed_management_app_plan(prompt: str) -> TaskPlanResponse:
    lower = prompt.lower()
    if "student" in lower:
        title, entity, category, contact = "Student Management System", "Student", "Program", "Student ID"
        seeds = [
            {"name": "Ayesha Rahman", "category": "CSE", "contact": "2026-001", "status": "Active", "notes": "Registered for summer semester."},
            {"name": "Tanvir Islam", "category": "EEE", "contact": "2026-014", "status": "Probation", "notes": "Advisor meeting needed."},
        ]
    elif "inventory" in lower:
        title, entity, category, contact = "Inventory Management System", "Item", "Category", "SKU"
        seeds = [
            {"name": "Wireless Mouse", "category": "Accessories", "contact": "SKU-1001", "status": "In Stock", "notes": "45 units available."},
            {"name": "Laptop Charger", "category": "Hardware", "contact": "SKU-1020", "status": "Low Stock", "notes": "Reorder before demo week."},
        ]
    elif "library" in lower:
        title, entity, category, contact = "Library Management System", "Book", "Section", "ISBN"
        seeds = [
            {"name": "Clean Code", "category": "Programming", "contact": "9780132350884", "status": "Available", "notes": "Popular CSE reference."},
            {"name": "Database Systems", "category": "Academic", "contact": "DB-204", "status": "Issued", "notes": "Due next week."},
        ]
    else:
        title, entity, category, contact = "CRM System", "Customer", "Company", "Contact"
        seeds = [
            {"name": "Northwind Traders", "category": "Retail", "contact": "sales@northwind.test", "status": "Lead", "notes": "Needs follow-up this week."},
            {"name": "BluePeak Software", "category": "SaaS", "contact": "hello@bluepeak.test", "status": "Proposal", "notes": "Interested in annual plan."},
        ]

    base = f"{_slug_from_title(title)}_database"
    seed_json = json.dumps(seeds, indent=4)
    config_json = json.dumps({"title": title, "entity": entity, "categoryLabel": category, "contactLabel": contact}, indent=2)
    requirements = "fastapi\nuvicorn[standard]\npydantic\n"
    package_json = json.dumps(
        {
            "name": base,
            "version": "0.1.0",
            "private": True,
            "scripts": {
                "dev": "python -m uvicorn backend.main:app --reload --port 8090",
                "smoke": "python tests/smoke_test.py",
            },
        },
        indent=2,
    )

    database_py = '''from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "app.db"
SEED_RECORDS = __SEED_JSON__


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                contact TEXT NOT NULL,
                status TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT ''
            )
            """
        )
        count = connection.execute("SELECT COUNT(*) AS total FROM records").fetchone()["total"]
        if count == 0:
            connection.executemany(
                "INSERT INTO records (name, category, contact, status, notes) VALUES (:name, :category, :contact, :status, :notes)",
                SEED_RECORDS,
            )
        connection.commit()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def list_records() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    return [row_to_dict(row) for row in rows]


def create_record(data: dict[str, str]) -> dict[str, Any]:
    with connect() as connection:
        cursor = connection.execute(
            "INSERT INTO records (name, category, contact, status, notes) VALUES (?, ?, ?, ?, ?)",
            (data["name"], data["category"], data["contact"], data["status"], data.get("notes", "")),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM records WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return row_to_dict(row)


def update_record(record_id: int, data: dict[str, str]) -> dict[str, Any] | None:
    with connect() as connection:
        connection.execute(
            "UPDATE records SET name = ?, category = ?, contact = ?, status = ?, notes = ? WHERE id = ?",
            (data["name"], data["category"], data["contact"], data["status"], data.get("notes", ""), record_id),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    return row_to_dict(row) if row else None


def delete_record(record_id: int) -> bool:
    with connect() as connection:
        cursor = connection.execute("DELETE FROM records WHERE id = ?", (record_id,))
        connection.commit()
    return cursor.rowcount > 0
'''.replace("__SEED_JSON__", seed_json)

    main_py = '''from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .database import create_record, delete_record, init_db, list_records, update_record

APP_CONFIG = __CONFIG_JSON__


class RecordIn(BaseModel):
    name: str
    category: str
    contact: str
    status: str = "Lead"
    notes: str = ""


app = FastAPI(title=APP_CONFIG["title"])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/config")
def config() -> dict:
    return APP_CONFIG


@app.get("/api/records")
def records() -> list[dict]:
    return list_records()


@app.post("/api/records", status_code=201)
def create(data: RecordIn) -> dict:
    return create_record(data.model_dump())


@app.put("/api/records/{record_id}")
def update(record_id: int, data: RecordIn) -> dict:
    record = update_record(record_id, data.model_dump())
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.delete("/api/records/{record_id}")
def remove(record_id: int) -> dict[str, bool]:
    if not delete_record(record_id):
        raise HTTPException(status_code=404, detail="Record not found")
    return {"ok": True}


frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
'''.replace("__CONFIG_JSON__", config_json)

    html = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <aside class="sidebar"><h1>__TITLE__</h1><button data-view="dashboard">Dashboard</button><button data-view="records">Records</button><button data-view="api">API</button></aside>
  <main class="content"><section id="dashboard"></section><section id="records"></section><section id="api" class="hidden"><h2>API Workflow</h2><pre>GET /api/records\nPOST /api/records\nPUT /api/records/{id}\nDELETE /api/records/{id}</pre></section></main>
  <script src="app.js"></script>
</body>
</html>
'''.replace("__TITLE__", title)

    app_js = '''let config = null;
let records = [];
let editingId = null;
const views = ["dashboard", "records", "api"];

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function load() {
  config = await api("/api/config");
  records = await api("/api/records");
  render();
}

function render() {
  renderDashboard();
  renderRecords();
}

function renderDashboard() {
  const open = records.filter((record) => !["Done", "Won", "Archived"].includes(record.status)).length;
  document.getElementById("dashboard").innerHTML = `<h2>Dashboard</h2><div class="metrics"><article><span>Total</span><strong>${records.length}</strong></article><article><span>Open</span><strong>${open}</strong></article><article><span>Closed</span><strong>${records.length - open}</strong></article></div>`;
}

function renderRecords() {
  const rows = records.map((record) => `<tr><td>${record.name}</td><td>${record.category}</td><td>${record.contact}</td><td>${record.status}</td><td>${record.notes || ""}</td><td><button data-edit="${record.id}">Edit</button><button data-delete="${record.id}">Delete</button></td></tr>`).join("");
  document.getElementById("records").innerHTML = `<h2>${config.entity} Records</h2><form id="recordForm" class="record-form"><input id="name" placeholder="${config.entity} name" required /><input id="category" placeholder="${config.categoryLabel}" required /><input id="contact" placeholder="${config.contactLabel}" required /><select id="status"><option>Lead</option><option>Active</option><option>Proposal</option><option>Won</option><option>Done</option><option>Archived</option></select><input id="notes" placeholder="Notes" /><button>Save to SQLite</button></form><table><thead><tr><th>Name</th><th>${config.categoryLabel}</th><th>${config.contactLabel}</th><th>Status</th><th>Notes</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table>`;
  bindRecordEvents();
}

function formData() {
  return Object.fromEntries(["name", "category", "contact", "status", "notes"].map((id) => [id, document.getElementById(id).value.trim()]));
}

function bindRecordEvents() {
  document.getElementById("recordForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = formData();
    if (editingId === null) await api("/api/records", { method: "POST", body: JSON.stringify(data) });
    else await api(`/api/records/${editingId}`, { method: "PUT", body: JSON.stringify(data) });
    editingId = null;
    records = await api("/api/records");
    render();
  });
  document.querySelectorAll("[data-delete]").forEach((button) => button.addEventListener("click", async () => {
    await api(`/api/records/${button.dataset.delete}`, { method: "DELETE" });
    records = await api("/api/records");
    render();
  }));
  document.querySelectorAll("[data-edit]").forEach((button) => button.addEventListener("click", () => {
    editingId = Number(button.dataset.edit);
    const record = records.find((item) => item.id === editingId);
    ["name", "category", "contact", "status", "notes"].forEach((id) => { document.getElementById(id).value = record[id] || ""; });
  }));
}

document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
  views.forEach((view) => document.getElementById(view).classList.toggle("hidden", view !== button.dataset.view));
}));

load().catch((error) => { document.body.innerHTML = `<main class="content"><h1>Could not load app</h1><pre>${error.message}</pre></main>`; });
'''

    css = '''body{margin:0;font-family:Arial,sans-serif;background:#f3f6fb;color:#172033}.sidebar{position:fixed;inset:0 auto 0 0;width:245px;background:#102a43;color:white;padding:24px;box-sizing:border-box}.sidebar h1{font-size:24px;line-height:1.2}.sidebar button{display:block;width:100%;margin:8px 0;padding:11px;border:0;border-radius:6px;background:#2563eb;color:white;text-align:left;font-weight:800;cursor:pointer}.content{margin-left:245px;padding:26px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metrics article,section{background:white;border:1px solid #d8e1ec;border-radius:8px;padding:18px;margin-bottom:18px}.metrics strong{display:block;font-size:32px;color:#0f766e}.record-form{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:16px}input,select{border:1px solid #cbd5e1;border-radius:6px;padding:10px;font:inherit}button{border:0;border-radius:6px;background:#7c3aed;color:white;padding:10px 12px;font-weight:800;cursor:pointer}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:10px}.hidden{display:none}pre{background:#0f172a;color:#e2e8f0;padding:16px;overflow:auto}@media(max-width:840px){.sidebar{position:static;width:auto}.content{margin-left:0}.record-form,.metrics{grid-template-columns:1fr}table{font-size:14px}}'''

    readme = f"""# {title} With SQLite Backend

Generated by SHAMSU as a database-backed local system.

## Run
```powershell
cd C:\\Users\\HP\\Desktop\\CSE327\\workspace\\{base}
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8090
```

Open `http://127.0.0.1:8090`.

## Verify
```powershell
python tests/smoke_test.py
```

## Data
The API creates `app.db` automatically using SQLite when the backend starts.
"""
    workflow = f"""# Workflow

Prompt: {prompt}

1. Requirement analysis: the prompt asks for a real {title} with persistent data.
2. Stack choice: FastAPI backend, SQLite database, vanilla JS frontend.
3. File plan: backend API, database layer, frontend, docs, smoke test.
4. Verification: Python syntax and smoke test check project structure.
5. Preview: run uvicorn and open the local URL.
"""
    smoke = '''from pathlib import Path
import py_compile

root = Path(__file__).resolve().parents[1]
required = [
    "backend/__init__.py",
    "backend/main.py",
    "backend/database.py",
    "frontend/index.html",
    "frontend/app.js",
    "frontend/styles.css",
    "requirements.txt",
    "README.md",
    "WORKFLOW.md",
]
missing = [path for path in required if not (root / path).exists()]
assert not missing, f"Missing files: {missing}"
py_compile.compile(str(root / "backend" / "main.py"), doraise=True)
py_compile.compile(str(root / "backend" / "database.py"), doraise=True)
assert "sqlite3" in (root / "backend" / "database.py").read_text(encoding="utf-8")
assert "/api/records" in (root / "frontend" / "app.js").read_text(encoding="utf-8")
print("database smoke passed")
'''

    files = [
        {"path": f"{base}/package.json", "content": package_json + "\n"},
        {"path": f"{base}/requirements.txt", "content": requirements},
        {"path": f"{base}/README.md", "content": readme},
        {"path": f"{base}/WORKFLOW.md", "content": workflow},
        {"path": f"{base}/backend/__init__.py", "content": "# Backend package for the SHAMSU-generated database system.\n"},
        {"path": f"{base}/backend/database.py", "content": database_py},
        {"path": f"{base}/backend/main.py", "content": main_py},
        {"path": f"{base}/frontend/index.html", "content": html},
        {"path": f"{base}/frontend/app.js", "content": app_js},
        {"path": f"{base}/frontend/styles.css", "content": css},
        {"path": f"{base}/tests/smoke_test.py", "content": smoke},
    ]
    return TaskPlanResponse(
        goal=prompt,
        mode="database-backed-system-generator",
        steps=["Analyze persistence requirements.", "Choose FastAPI + SQLite + vanilla JS stack.", "Create backend API, database layer, frontend, docs, and smoke test.", "Run Python syntax and file verification.", "Serve with uvicorn for local testing."],
        suggested_files=files,
        verify_commands=[f"python {base}/tests/smoke_test.py", f"cd {base} && python -m uvicorn backend.main:app --reload --port 8090"],
        notes=["SHAMSU generated a database-backed local system, not only a static mockup.", "SQLite keeps the demo self-contained while matching a real backend/API workflow."],
        requirements_analysis=[f"The prompt asks for a {title}-style system with persistent records.", "A backend API and SQLite database make create/update/delete survive page reloads."],
        clarification_questions=["Which user roles, authentication rules, and reports should be added next?", "Should the SQLite schema be upgraded to MySQL for final deployment?"],
        stack=["FastAPI", "SQLite", "Pydantic", "Vanilla JavaScript", "HTML", "CSS", "Python smoke test"],
        file_plan=[item["path"] for item in files],
        workflow_summary="prompt -> requirement analysis -> choose FastAPI/SQLite stack -> create backend/database/frontend -> smoke verification -> run uvicorn -> iterate",
    )
def _multi_file_management_app_plan(prompt: str) -> TaskPlanResponse:
    lower = prompt.lower()
    if "student" in lower:
        title, entity, category, contact = "Student Management System", "Student", "Program", "Student ID"
        seeds = [
            {"name": "Ayesha Rahman", "category": "CSE", "contact": "2026-001", "status": "Active", "notes": "Registered for summer semester."},
            {"name": "Tanvir Islam", "category": "EEE", "contact": "2026-014", "status": "Probation", "notes": "Advisor meeting needed."},
        ]
    elif "inventory" in lower:
        title, entity, category, contact = "Inventory Management System", "Item", "Category", "SKU"
        seeds = [
            {"name": "Wireless Mouse", "category": "Accessories", "contact": "SKU-1001", "status": "In Stock", "notes": "45 units available."},
            {"name": "Laptop Charger", "category": "Hardware", "contact": "SKU-1020", "status": "Low Stock", "notes": "Reorder before demo week."},
        ]
    elif "library" in lower:
        title, entity, category, contact = "Library Management System", "Book", "Section", "ISBN"
        seeds = [
            {"name": "Clean Code", "category": "Programming", "contact": "9780132350884", "status": "Available", "notes": "Popular CSE reference."},
            {"name": "Database Systems", "category": "Academic", "contact": "DB-204", "status": "Issued", "notes": "Due next week."},
        ]
    else:
        title, entity, category, contact = "CRM System", "Customer", "Company", "Contact"
        seeds = [
            {"name": "Northwind Traders", "category": "Retail", "contact": "sales@northwind.test", "status": "Lead", "notes": "Needs follow-up this week."},
            {"name": "BluePeak Software", "category": "SaaS", "contact": "hello@bluepeak.test", "status": "Proposal", "notes": "Interested in annual plan."},
        ]
    base = _slug_from_title(title)
    package_json = json.dumps({"name": base, "version": "0.1.0", "private": True, "type": "module", "scripts": {"preview": "python -m http.server 9000", "smoke": "python tests/smoke_test.py"}}, indent=2)
    html = f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>{title}</title><link rel="stylesheet" href="src/styles.css" /></head>
<body><div id="app"></div><script type="module" src="src/main.js"></script></body>
</html>
'''
    data_js = f'''export const appConfig = {{ title: {json.dumps(title)}, entity: {json.dumps(entity)}, categoryLabel: {json.dumps(category)}, contactLabel: {json.dumps(contact)}, statuses: ["Lead", "Active", "Proposal", "Won", "Open", "Done", "Archived"] }};
export const seedRecords = {json.dumps(seeds, indent=2)};
'''
    state_js = f'''import {{ seedRecords }} from './data.js';
const storageKey = '{base}-records';
export function loadRecords() {{ return JSON.parse(localStorage.getItem(storageKey) || 'null') || seedRecords; }}
export function saveRecords(records) {{ localStorage.setItem(storageKey, JSON.stringify(records)); }}
export function addRecord(records, record) {{ return [...records, {{ ...record, id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) }}]; }}
export function updateRecord(records, index, record) {{ return records.map((item, currentIndex) => currentIndex === index ? {{ ...item, ...record }} : item); }}
export function deleteRecord(records, index) {{ return records.filter((_, currentIndex) => currentIndex !== index); }}
'''
    views_js = '''import { appConfig } from './data.js';
export function layout() { return `<aside class="sidebar"><h1>${appConfig.title}</h1><button data-view="dashboard">Dashboard</button><button data-view="records">Records</button><button data-view="workflow">Workflow</button></aside><main class="content"><section id="dashboardView"></section><section id="recordsView"></section><section id="workflowView" class="hidden"><h2>Workflow</h2><ol><li>Requirement analysis</li><li>Stack choice</li><li>Multi-file project scaffold</li><li>Smoke verification</li><li>Preview and iterate</li></ol></section></main>`; }
export function dashboard(records) { const openCount = records.filter((record) => !['Done', 'Won', 'Archived'].includes(record.status)).length; return `<h2>Dashboard</h2><div class="metrics"><article><span>Total</span><strong>${records.length}</strong></article><article><span>Open</span><strong>${openCount}</strong></article><article><span>Completed</span><strong>${records.length - openCount}</strong></article></div>`; }
export function recordsTable(records) { const rows = records.map((record, index) => `<tr><td>${record.name}</td><td>${record.category}</td><td>${record.contact}</td><td><span>${record.status}</span></td><td>${record.notes || ''}</td><td><button data-edit="${index}">Edit</button><button data-delete="${index}">Delete</button></td></tr>`).join(''); return `<h2>${appConfig.entity} Records</h2><form id="recordForm" class="record-form"><input id="name" placeholder="${appConfig.entity} name" required /><input id="category" placeholder="${appConfig.categoryLabel}" required /><input id="contact" placeholder="${appConfig.contactLabel}" required /><select id="status">${appConfig.statuses.map((status) => `<option>${status}</option>`).join('')}</select><input id="notes" placeholder="Notes" /><button type="submit">Save</button></form><table><thead><tr><th>Name</th><th>${appConfig.categoryLabel}</th><th>${appConfig.contactLabel}</th><th>Status</th><th>Notes</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table>`; }
'''
    main_js = '''import { addRecord, deleteRecord, loadRecords, saveRecords, updateRecord } from './state.js';
import { dashboard, layout, recordsTable } from './views.js';
let records = loadRecords(); let editingIndex = null; const app = document.getElementById('app'); app.innerHTML = layout();
function render() { document.getElementById('dashboardView').innerHTML = dashboard(records); document.getElementById('recordsView').innerHTML = recordsTable(records); bindRecordEvents(); }
function showView(name) { document.getElementById('dashboardView').classList.toggle('hidden', name !== 'dashboard'); document.getElementById('recordsView').classList.toggle('hidden', name !== 'records'); document.getElementById('workflowView').classList.toggle('hidden', name !== 'workflow'); }
function bindRecordEvents() { document.getElementById('recordForm').addEventListener('submit', (event) => { event.preventDefault(); const record = Object.fromEntries(['name', 'category', 'contact', 'status', 'notes'].map((id) => [id, document.getElementById(id).value.trim()])); records = editingIndex === null ? addRecord(records, record) : updateRecord(records, editingIndex, record); editingIndex = null; saveRecords(records); render(); }); document.querySelectorAll('[data-delete]').forEach((button) => button.addEventListener('click', () => { records = deleteRecord(records, Number(button.dataset.delete)); saveRecords(records); render(); })); document.querySelectorAll('[data-edit]').forEach((button) => button.addEventListener('click', () => { editingIndex = Number(button.dataset.edit); const record = records[editingIndex]; ['name', 'category', 'contact', 'status', 'notes'].forEach((id) => { document.getElementById(id).value = record[id] || ''; }); })); }
document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => showView(button.dataset.view))); render(); showView('dashboard');
'''
    css = '''body{margin:0;font-family:Arial,sans-serif;background:#f3f6fb;color:#172033}.sidebar{position:fixed;inset:0 auto 0 0;width:240px;background:#102a43;color:white;padding:24px;box-sizing:border-box}.sidebar h1{font-size:24px;line-height:1.2}.sidebar button{display:block;width:100%;margin:8px 0;padding:11px;border:0;border-radius:6px;background:#2563eb;color:white;text-align:left;font-weight:800;cursor:pointer}.content{margin-left:240px;padding:26px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metrics article,section{background:white;border:1px solid #d8e1ec;border-radius:8px;padding:18px;margin-bottom:18px}.metrics strong{display:block;font-size:32px;color:#0f766e}.record-form{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:16px}input,select{border:1px solid #cbd5e1;border-radius:6px;padding:10px;font:inherit}button{border:0;border-radius:6px;background:#7c3aed;color:white;padding:10px 12px;font-weight:800;cursor:pointer}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:10px}.hidden{display:none}@media(max-width:840px){.sidebar{position:static;width:auto}.content{margin-left:0}.record-form,.metrics{grid-template-columns:1fr}table{font-size:14px}}'''
    readme = f"""# {title}\n\nGenerated by SHAMSU as a multi-file project scaffold.\n\n## Run\n```powershell\ncd C:\\Users\\HP\\Desktop\\CSE327\\workspace\\{base}\npython -m http.server 9000\n```\n\nOpen `http://127.0.0.1:9000/index.html`.\n\n## Verify\n```powershell\npython tests/smoke_test.py\n```\n"""
    workflow = f"""# Workflow\n\nPrompt: {prompt}\n\n1. Requirement analysis: build a usable {title} prototype.\n2. Stack choice: static modular JS first, upgradeable to React/FastAPI/MySQL.\n3. File plan: separate data, state, views, styles, docs, smoke test.\n4. Verification: generated files are readable and smoke test checks required files.\n5. Preview: browser preview opens `index.html`.\n"""
    smoke = f'''from pathlib import Path
root = Path(__file__).resolve().parents[1]
required = ["index.html", "src/main.js", "src/views.js", "src/state.js", "src/data.js", "src/styles.css", "README.md", "WORKFLOW.md"]
missing = [path for path in required if not (root / path).exists()]
assert not missing, f"Missing files: {{missing}}"
assert "{title}" in (root / "index.html").read_text(encoding="utf-8")
assert "localStorage" in (root / "src/state.js").read_text(encoding="utf-8")
print("smoke passed")
'''
    files = [
        {"path": f"{base}/package.json", "content": package_json + "\n"},
        {"path": f"{base}/README.md", "content": readme},
        {"path": f"{base}/WORKFLOW.md", "content": workflow},
        {"path": f"{base}/index.html", "content": html},
        {"path": f"{base}/src/main.js", "content": main_js},
        {"path": f"{base}/src/views.js", "content": views_js},
        {"path": f"{base}/src/state.js", "content": state_js},
        {"path": f"{base}/src/data.js", "content": data_js},
        {"path": f"{base}/src/styles.css", "content": css},
        {"path": f"{base}/tests/smoke_test.py", "content": smoke},
    ]
    return TaskPlanResponse(
        goal=prompt,
        mode="multi-file-project-generator",
        steps=["Analyze requirements.", "Choose modular browser app stack.", "Create folder scaffold with data/state/views/styles/tests/docs.", "Run lightweight file verification.", "Start preview server."],
        suggested_files=files,
        verify_commands=[f"python {base}/tests/smoke_test.py", f"Open http://127.0.0.1:9000/{base}/index.html"],
        notes=["SHAMSU generated a multi-file project instead of a single HTML file so the app can be extended like a Claude-created project."],
        requirements_analysis=[f"The prompt asks for a {title}-style system with multiple responsibilities.", "A modular file structure makes future edits safer and easier to explain."],
        clarification_questions=["Which roles, reports, and database fields should the production version include?", "Should SHAMSU upgrade this scaffold to React/FastAPI/MySQL next?"],
        stack=["HTML", "CSS", "JavaScript modules", "localStorage", "Python smoke test", "Local preview server"],
        file_plan=[item["path"] for item in files],
        workflow_summary="prompt -> requirement analysis -> stack choice -> multi-file file plan -> scaffold -> smoke verification -> preview -> next iteration",
    )


def _slug_from_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "shamsu_project"
def _management_system_plan(prompt: str) -> TaskPlanResponse:
    lower = prompt.lower()
    presets = {
        "crm": {
            "title": "CRM System",
            "file": "crm_system.html",
            "entity": "Customer",
            "category": "Company",
            "statuses": ["Lead", "Contacted", "Proposal", "Won", "Lost"],
            "seed": [
                {"name": "Northwind Traders", "category": "Retail", "contact": "sales@northwind.test", "status": "Lead", "notes": "Needs follow-up this week."},
                {"name": "BluePeak Software", "category": "SaaS", "contact": "hello@bluepeak.test", "status": "Proposal", "notes": "Interested in annual plan."},
            ],
        },
        "student": {
            "title": "Student Management System",
            "file": "student_management_system.html",
            "entity": "Student",
            "category": "Program",
            "statuses": ["Active", "Probation", "Graduated", "Dropped"],
            "seed": [
                {"name": "Ayesha Rahman", "category": "CSE", "contact": "2026-001", "status": "Active", "notes": "Registered for summer semester."},
                {"name": "Tanvir Islam", "category": "EEE", "contact": "2026-014", "status": "Probation", "notes": "Advisor meeting needed."},
            ],
        },
        "inventory": {
            "title": "Inventory Management System",
            "file": "inventory_management_system.html",
            "entity": "Item",
            "category": "Category",
            "statuses": ["In Stock", "Low Stock", "Ordered", "Discontinued"],
            "seed": [
                {"name": "Wireless Mouse", "category": "Accessories", "contact": "SKU-1001", "status": "In Stock", "notes": "45 units available."},
                {"name": "Laptop Charger", "category": "Hardware", "contact": "SKU-1020", "status": "Low Stock", "notes": "Reorder before demo week."},
            ],
        },
        "library": {
            "title": "Library Management System",
            "file": "library_management_system.html",
            "entity": "Book",
            "category": "Author",
            "statuses": ["Available", "Checked Out", "Reserved", "Lost"],
            "seed": [
                {"name": "Clean Code", "category": "Robert C. Martin", "contact": "BK-001", "status": "Available", "notes": "Software engineering shelf."},
                {"name": "Database Systems", "category": "Elmasri", "contact": "BK-018", "status": "Checked Out", "notes": "Due next Monday."},
            ],
        },
        "generic": {
            "title": "Management System",
            "file": "management_system.html",
            "entity": "Record",
            "category": "Category",
            "statuses": ["New", "In Progress", "Complete", "Archived"],
            "seed": [
                {"name": "Demo Record", "category": "General", "contact": "REF-001", "status": "New", "notes": "Created by SHAMSU."},
                {"name": "Follow-up Task", "category": "Operations", "contact": "REF-002", "status": "In Progress", "notes": "Track progress here."},
            ],
        },
    }
    key = "generic"
    for candidate in ["crm", "student", "inventory", "library"]:
        if candidate in lower:
            key = candidate
            break
    preset = presets[key]
    status_options = "\n".join(f'          <option value="{status}">{status}</option>' for status in preset["statuses"])
    seed_records = json.dumps(preset["seed"], indent=6)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root{font-family:Arial,sans-serif;color:#172033;background:#f3f6fb}body{margin:0}header{background:#103b57;color:white;padding:22px 32px}main{max-width:1120px;margin:0 auto;padding:24px}.top{display:flex;justify-content:space-between;gap:16px;align-items:center}.subtitle{color:#d6e9f5;margin:6px 0 0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.metric{background:white;border:1px solid #dde5ef;border-radius:8px;padding:16px}.metric strong{display:block;font-size:28px;color:#0f766e}.panel{background:white;border:1px solid #dde5ef;border-radius:8px;padding:18px;margin-bottom:18px}.form{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.form input,.form select,.form textarea,.search{width:100%;box-sizing:border-box;border:1px solid #cbd5e1;border-radius:6px;padding:10px;font:inherit}.form textarea{grid-column:span 2;min-height:42px;resize:vertical}.actions{display:flex;gap:8px}.btn{border:0;border-radius:6px;padding:10px 14px;font-weight:700;cursor:pointer}.primary{background:#7c3aed;color:white}.secondary{background:#e2e8f0;color:#172033}.danger{background:#fee2e2;color:#b91c1c}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:12px;vertical-align:top}th{color:#475569;font-size:13px;text-transform:uppercase}.status{display:inline-block;border-radius:999px;background:#dcfce7;color:#166534;padding:4px 9px;font-size:12px;font-weight:700}.empty{text-align:center;color:#64748b;padding:26px}@media(max-width:820px){.grid,.form{grid-template-columns:1fr}.form textarea{grid-column:auto}.top{align-items:flex-start;flex-direction:column}}
  </style>
</head>
<body>
  <header>
    <div class="top">
      <div><h1>__TITLE__</h1><p class="subtitle">Local prototype with search, CRUD actions, status tracking, and saved browser data.</p></div>
      <button class="btn secondary" id="resetBtn">Reset Demo Data</button>
    </div>
  </header>
  <main>
    <section class="grid" id="metrics"></section>
    <section class="panel">
      <h2 id="formTitle">Add __ENTITY__</h2>
      <form class="form" id="recordForm">
        <input id="name" placeholder="__ENTITY__ name" required />
        <input id="category" placeholder="__CATEGORY__" required />
        <input id="contact" placeholder="Email, ID, SKU, or phone" required />
        <select id="status">__STATUS_OPTIONS__
        </select>
        <textarea id="notes" placeholder="Notes"></textarea>
        <div class="actions"><button class="btn primary" type="submit">Save</button><button class="btn secondary" type="button" id="cancelBtn">Cancel</button></div>
      </form>
    </section>
    <section class="panel">
      <input class="search" id="search" placeholder="Search records..." />
      <table aria-label="Records table">
        <thead><tr><th>Name</th><th>__CATEGORY__</th><th>Contact/ID</th><th>Status</th><th>Notes</th><th>Actions</th></tr></thead>
        <tbody id="recordsBody"></tbody>
      </table>
    </section>
  </main>
  <script>
    const localStorageKey = '__FILE__::records';
    const seedRecords = __SEED_RECORDS__;
    let records = JSON.parse(localStorage.getItem(localStorageKey) || 'null') || seedRecords;
    let editingIndex = null;
    const fields = ['name', 'category', 'contact', 'status', 'notes'];
    const form = document.getElementById('recordForm');
    const recordsBody = document.getElementById('recordsBody');
    const search = document.getElementById('search');
    const formTitle = document.getElementById('formTitle');
    function saveRecords(){ localStorage.setItem(localStorageKey, JSON.stringify(records)); }
    function filteredRecords(){ const term = search.value.toLowerCase(); return records.map((record, index) => ({record, index})).filter(({record}) => Object.values(record).join(' ').toLowerCase().includes(term)); }
    function renderMetrics(){
      const counts = records.reduce((acc, record) => { acc[record.status] = (acc[record.status] || 0) + 1; return acc; }, {});
      document.getElementById('metrics').innerHTML = `<article class="metric"><span>Total</span><strong>${records.length}</strong></article>` + Object.entries(counts).map(([status, count]) => `<article class="metric"><span>${status}</span><strong>${count}</strong></article>`).join('');
    }
    function renderRecords(){
      const rows = filteredRecords();
      recordsBody.innerHTML = rows.length ? rows.map(({record, index}) => `<tr><td>${record.name}</td><td>${record.category}</td><td>${record.contact}</td><td><span class="status">${record.status}</span></td><td>${record.notes || ''}</td><td><div class="actions"><button class="btn secondary" onclick="editRecord(${index})">Edit</button><button class="btn danger" onclick="deleteRecord(${index})">Delete</button></div></td></tr>`).join('') : `<tr><td class="empty" colspan="6">No records found.</td></tr>`;
      renderMetrics();
    }
    function editRecord(index){ editingIndex = index; const record = records[index]; fields.forEach((field) => document.getElementById(field).value = record[field] || ''); formTitle.textContent = 'Edit __ENTITY__'; window.scrollTo({top:0,behavior:'smooth'}); }
    function deleteRecord(index){ records.splice(index, 1); saveRecords(); renderRecords(); }
    function resetForm(){ editingIndex = null; form.reset(); formTitle.textContent = 'Add __ENTITY__'; }
    form.addEventListener('submit', (event) => { event.preventDefault(); const record = Object.fromEntries(fields.map((field) => [field, document.getElementById(field).value.trim()])); if (editingIndex === null) records.push(record); else records[editingIndex] = record; saveRecords(); resetForm(); renderRecords(); });
    document.getElementById('cancelBtn').addEventListener('click', resetForm);
    document.getElementById('resetBtn').addEventListener('click', () => { records = seedRecords; saveRecords(); resetForm(); renderRecords(); });
    search.addEventListener('input', renderRecords);
    renderRecords();
  </script>
</body>
</html>
"""
    html = html.replace("__TITLE__", str(preset["title"]))
    html = html.replace("__ENTITY__", str(preset["entity"]))
    html = html.replace("__CATEGORY__", str(preset["category"]))
    html = html.replace("__STATUS_OPTIONS__", status_options)
    html = html.replace("__SEED_RECORDS__", seed_records)
    html = html.replace("__FILE__", str(preset["file"]))
    return TaskPlanResponse(
        goal=prompt,
        mode="management-system-generator",
        steps=[
            "Classify the requested management system.",
            "Create a localStorage CRUD prototype.",
            "Verify the generated HTML structure.",
            "Start preview server for browser testing.",
        ],
        suggested_files=[{"path": str(preset["file"]), "content": html}],
        verify_commands=[f"Open http://127.0.0.1:9000/{preset['file']}"],
        notes=[
            "This is a Claude-style prototype-first build: SHAMSU creates a working local UI before suggesting a larger production architecture.",
            "For a production system, the next layer would be authentication, a backend API, and a database.",
        ],
    )

def _brick_breaker_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Brick Breaker</title>
  <style>
    body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0f172a;color:#f8fafc;font-family:Arial,sans-serif}
    main{text-align:center} canvas{background:#f8fafc;border:3px solid #1e293b;box-shadow:0 16px 40px rgba(0,0,0,.35);outline:none}
    .hud{display:flex;justify-content:center;gap:28px;margin:8px 0 14px;font-size:18px}.hint{color:#cbd5e1}
  </style>
</head>
<body>
  <main>
    <h1>Brick Breaker</h1>
    <div class="hud"><span>Score: <strong id="score">0</strong></span><span>Lives: <strong id="lives">3</strong></span></div>
    <canvas id="game" width="640" height="420" tabindex="0"></canvas>
    <p class="hint">Fixed version: use mouse or Left/Right arrows. Press Space to launch.</p>
  </main>
  <script>
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const scoreEl = document.getElementById('score');
    const livesEl = document.getElementById('lives');
    const paddle = { w: 104, h: 14, x: 268, y: 388, speed: 8 };
    const ball = { x: 320, y: 360, r: 8, dx: 3.4, dy: -3.8, stuck: true };
    const keys = { left: false, right: false };
    const rows = 5, cols = 9, brickW = 58, brickH = 22, gap = 8, brickTop = 50, brickLeft = 26;
    let bricks = [];
    let score = 0;
    let lives = 3;
    let won = false;
    let over = false;

    function makeBricks() {
      bricks = [];
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          bricks.push({ x: brickLeft + col * (brickW + gap), y: brickTop + row * (brickH + gap), alive: true, color: ['#ef4444','#f97316','#eab308','#22c55e','#3b82f6'][row] });
        }
      }
    }

    function reset(full = true) {
      if (full) { score = 0; lives = 3; won = false; over = false; makeBricks(); }
      paddle.x = (canvas.width - paddle.w) / 2;
      ball.x = paddle.x + paddle.w / 2;
      ball.y = paddle.y - ball.r - 3;
      ball.dx = 3.4 * (Math.random() > 0.5 ? 1 : -1);
      ball.dy = -3.8;
      ball.stuck = true;
      scoreEl.textContent = score;
      livesEl.textContent = lives;
      draw();
      canvas.focus();
    }

    function update() {
      if (over || won) return;
      if (keys.left) paddle.x -= paddle.speed;
      if (keys.right) paddle.x += paddle.speed;
      paddle.x = Math.max(0, Math.min(canvas.width - paddle.w, paddle.x));
      if (ball.stuck) { ball.x = paddle.x + paddle.w / 2; ball.y = paddle.y - ball.r - 3; return; }
      ball.x += ball.dx; ball.y += ball.dy;
      if (ball.x <= ball.r || ball.x >= canvas.width - ball.r) ball.dx *= -1;
      if (ball.y <= ball.r) ball.dy *= -1;
      if (ball.y > canvas.height + ball.r) {
        lives -= 1;
        livesEl.textContent = lives;
        if (lives <= 0) over = true;
        else reset(false);
      }
      const paddleHit = ball.y + ball.r >= paddle.y && ball.y - ball.r <= paddle.y + paddle.h && ball.x >= paddle.x && ball.x <= paddle.x + paddle.w;
      if (paddleHit && ball.dy > 0) {
        const hit = (ball.x - (paddle.x + paddle.w / 2)) / (paddle.w / 2);
        ball.dx = hit * 5;
        ball.dy = -Math.abs(ball.dy);
      }
      for (const brick of bricks) {
        if (!brick.alive) continue;
        const hitBrick = ball.x + ball.r > brick.x && ball.x - ball.r < brick.x + brickW && ball.y + ball.r > brick.y && ball.y - ball.r < brick.y + brickH;
        if (hitBrick) {
          brick.alive = false;
          ball.dy *= -1;
          score += 10;
          scoreEl.textContent = score;
          if (bricks.every(b => !b.alive)) won = true;
          break;
        }
      }
    }

    function draw() {
      ctx.fillStyle = '#e2e8f0';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (const brick of bricks) {
        if (!brick.alive) continue;
        ctx.fillStyle = brick.color;
        ctx.fillRect(brick.x, brick.y, brickW, brickH);
        ctx.strokeStyle = 'rgba(15,23,42,.22)';
        ctx.strokeRect(brick.x, brick.y, brickW, brickH);
      }
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(paddle.x, paddle.y, paddle.w, paddle.h);
      ctx.beginPath();
      ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2);
      ctx.fillStyle = '#7c3aed';
      ctx.fill();
      if (ball.stuck && !over && !won) message('Press Space to launch', 18);
      if (over) message('Game Over - Press Space', 26);
      if (won) message('You Win - Press Space', 26);
    }

    function message(text, size) {
      ctx.fillStyle = 'rgba(15,23,42,.82)';
      ctx.fillRect(0, 180, canvas.width, 58);
      ctx.fillStyle = '#fff';
      ctx.font = size + 'px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(text, canvas.width / 2, 216);
    }

    function loop() { update(); draw(); requestAnimationFrame(loop); }
    function handleKey(event, pressed) {
      if (['ArrowLeft', 'ArrowRight', 'Space'].includes(event.code) || ['ArrowLeft', 'ArrowRight'].includes(event.key)) event.preventDefault();
      if (event.key === 'ArrowLeft') keys.left = pressed;
      if (event.key === 'ArrowRight') keys.right = pressed;
      if (pressed && event.code === 'Space') { if (over || won) reset(true); else ball.stuck = false; }
    }
    document.addEventListener('keydown', event => handleKey(event, true));
    document.addEventListener('keyup', event => handleKey(event, false));
    canvas.addEventListener('mousemove', event => {
      const rect = canvas.getBoundingClientRect();
      paddle.x = Math.max(0, Math.min(canvas.width - paddle.w, event.clientX - rect.left - paddle.w / 2));
      if (ball.stuck) { ball.x = paddle.x + paddle.w / 2; ball.y = paddle.y - ball.r - 3; }
    });
    canvas.addEventListener('click', () => canvas.focus());
    reset(true);
    loop();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(
        goal=prompt,
        mode="game-generator",
        steps=["Create a Brick Breaker HTML canvas game.", "Verify the HTML game structure.", "Start preview server."],
        suggested_files=[{"path": "brick_breaker.html", "content": html}],
        verify_commands=["Open http://127.0.0.1:9000/brick_breaker.html"],
        notes=["This deterministic template is used for brick breaker prompts so demos do not depend on JSON fallback reliability."],
    )
def _snake_game_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Snake Game</title>
  <style>
    body{margin:0;min-height:100vh;display:grid;place-items:center;background:#111827;color:#f9fafb;font-family:Arial,sans-serif}
    main{text-align:center} canvas{background:#f8fafc;border:3px solid #1f2937;outline:none}.score{font-size:20px;margin:8px 0 14px}
  </style>
</head>
<body>
  <main>
    <h1>Snake Game</h1>
    <div class="score">Score: <span id="score">0</span></div>
    <canvas id="game" width="400" height="400" tabindex="0"></canvas>
    <p>Use arrow keys to move. Press Space to restart after game over.</p>
  </main>
  <script>
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const scoreEl = document.getElementById('score');
    const size = 20;
    const cells = canvas.width / size;
    let snake, food, direction, queuedDirection, score, gameOver, timer;

    function reset() {
      snake = [{ x: 8, y: 10 }, { x: 7, y: 10 }, { x: 6, y: 10 }];
      food = { x: 14, y: 10 };
      direction = { x: 1, y: 0 };
      queuedDirection = { x: 1, y: 0 };
      score = 0;
      gameOver = false;
      scoreEl.textContent = score;
      draw();
      canvas.focus();
    }

    function placeFood() {
      do {
        food = { x: Math.floor(Math.random() * cells), y: Math.floor(Math.random() * cells) };
      } while (snake.some(part => part.x === food.x && part.y === food.y));
    }

    function step() {
      if (gameOver) return draw();
      direction = queuedDirection;
      const head = { x: snake[0].x + direction.x, y: snake[0].y + direction.y };
      if (head.x < 0 || head.x >= cells || head.y < 0 || head.y >= cells || snake.some(part => part.x === head.x && part.y === head.y)) {
        gameOver = true;
        return draw();
      }
      snake.unshift(head);
      if (head.x === food.x && head.y === food.y) {
        score += 10;
        scoreEl.textContent = score;
        placeFood();
      } else {
        snake.pop();
      }
      draw();
    }

    function draw() {
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ef4444';
      ctx.fillRect(food.x * size, food.y * size, size, size);
      snake.forEach((part, index) => {
        ctx.fillStyle = index === 0 ? '#15803d' : '#22c55e';
        ctx.fillRect(part.x * size + 1, part.y * size + 1, size - 2, size - 2);
      });
      if (gameOver) {
        ctx.fillStyle = 'rgba(17,24,39,0.82)';
        ctx.fillRect(0, 160, canvas.width, 80);
        ctx.fillStyle = '#fff';
        ctx.font = '26px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Game Over', canvas.width / 2, 195);
        ctx.font = '15px Arial';
        ctx.fillText('Press Space to restart', canvas.width / 2, 222);
      }
    }

    function setDirection(next) {
      if (next.x + direction.x === 0 && next.y + direction.y === 0) return;
      queuedDirection = next;
    }

    document.addEventListener('keydown', (event) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(event.code)) event.preventDefault();
      if (event.key === 'ArrowUp') setDirection({ x: 0, y: -1 });
      if (event.key === 'ArrowDown') setDirection({ x: 0, y: 1 });
      if (event.key === 'ArrowLeft') setDirection({ x: -1, y: 0 });
      if (event.key === 'ArrowRight') setDirection({ x: 1, y: 0 });
      if (event.code === 'Space' && gameOver) reset();
    });
    canvas.addEventListener('click', () => canvas.focus());

    reset();
    timer = setInterval(step, 185);
  </script>
</body>
</html>
"""
    return TaskPlanResponse(
        goal=prompt,
        mode="game-generator",
        steps=["Create a Snake HTML canvas game.", "Verify the HTML game structure.", "Start preview server."],
        suggested_files=[{"path": "snake_game.html", "content": html}],
        verify_commands=["Open http://127.0.0.1:9000/snake_game.html"],
        notes=["This deterministic template is used for snake game prompts so demos do not depend on JSON fallback reliability."],
    )
def _bouncing_ball_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Bouncing Ball</title>
  <style>body{margin:0;background:#101820;color:white;font-family:sans-serif}canvas{display:block;margin:24px auto;background:#f7f7f7;border:2px solid #333}</style>
</head>
<body>
  <canvas id=\"game\" width=\"640\" height=\"360\"></canvas>
  <script>
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const ball = { x: 80, y: 80, vx: 4, vy: 3, r: 18 };
    function tick() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ball.x += ball.vx; ball.y += ball.vy;
      if (ball.x < ball.r || ball.x > canvas.width - ball.r) ball.vx *= -1;
      if (ball.y < ball.r || ball.y > canvas.height - ball.r) ball.vy *= -1;
      ctx.beginPath(); ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2);
      ctx.fillStyle = '#7c3aed'; ctx.fill();
      requestAnimationFrame(tick);
    }
    tick();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(
        goal=prompt,
        mode="game-generator",
        steps=["Create an HTML canvas game file in the workspace.", "Verify the game structure.", "Start preview server."],
        suggested_files=[{"path": "bouncing_ball.html", "content": html}],
        verify_commands=["Open http://127.0.0.1:9000/bouncing_ball.html"],
        notes=["This deterministic template is used for bouncing ball prompts."],
    )


def _pong_game_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Pong</title>
  <style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#101827;color:#f8fafc;font-family:Arial,sans-serif}main{text-align:center}canvas{background:#0f172a;border:3px solid #334155;outline:none}.hud{display:flex;justify-content:center;gap:32px;margin:8px 0 14px;font-size:20px}</style>
</head>
<body>
  <main>
    <h1>Pong</h1>
    <div class="hud"><span>You: <strong id="playerScore">0</strong></span><span>AI: <strong id="aiScore">0</strong></span></div>
    <canvas id="game" width="720" height="420" tabindex="0"></canvas>
    <p>Use Up/Down arrows or mouse. First to 7 wins. Press Space to restart.</p>
  </main>
  <script>
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const playerScoreEl = document.getElementById('playerScore');
    const aiScoreEl = document.getElementById('aiScore');
    const player = { x: 24, y: 160, w: 14, h: 92, speed: 7 };
    const aiPaddle = { x: 682, y: 160, w: 14, h: 92, speed: 4.2 };
    const ball = { x: 360, y: 210, r: 9, dx: 4.4, dy: 2.8 };
    const keys = { up: false, down: false };
    let playerScore = 0, aiScore = 0, message = 'Press Space to start', running = false;
    function resetBall(direction) { ball.x = canvas.width / 2; ball.y = canvas.height / 2; ball.dx = 4.4 * direction; ball.dy = (Math.random() > 0.5 ? 1 : -1) * 2.8; }
    function restart() { playerScore = 0; aiScore = 0; player.y = 160; aiPaddle.y = 160; resetBall(1); message = ''; running = true; updateScore(); canvas.focus(); }
    function updateScore() { playerScoreEl.textContent = playerScore; aiScoreEl.textContent = aiScore; }
    function update() {
      if (!running) return;
      if (keys.up) player.y -= player.speed;
      if (keys.down) player.y += player.speed;
      player.y = Math.max(0, Math.min(canvas.height - player.h, player.y));
      const aiTarget = ball.y - aiPaddle.h / 2;
      aiPaddle.y += Math.max(-aiPaddle.speed, Math.min(aiPaddle.speed, aiTarget - aiPaddle.y));
      aiPaddle.y = Math.max(0, Math.min(canvas.height - aiPaddle.h, aiPaddle.y));
      ball.x += ball.dx; ball.y += ball.dy;
      if (ball.y <= ball.r || ball.y >= canvas.height - ball.r) ball.dy *= -1;
      for (const paddle of [player, aiPaddle]) {
        const hit = ball.x + ball.r > paddle.x && ball.x - ball.r < paddle.x + paddle.w && ball.y + ball.r > paddle.y && ball.y - ball.r < paddle.y + paddle.h;
        if (hit) { const offset = (ball.y - (paddle.y + paddle.h / 2)) / (paddle.h / 2); ball.dx = Math.abs(ball.dx) * (paddle === player ? 1 : -1) * 1.04; ball.dy = offset * 5; }
      }
      if (ball.x < -20) { aiScore++; resetBall(1); updateScore(); }
      if (ball.x > canvas.width + 20) { playerScore++; resetBall(-1); updateScore(); }
      if (playerScore >= 7 || aiScore >= 7) { running = false; message = playerScore >= 7 ? 'You win! Space to restart' : 'AI wins. Space to restart'; }
    }
    function draw() {
      ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#334155'; ctx.setLineDash([8, 10]); ctx.beginPath(); ctx.moveTo(canvas.width/2, 0); ctx.lineTo(canvas.width/2, canvas.height); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#f8fafc'; ctx.fillRect(player.x, player.y, player.w, player.h); ctx.fillRect(aiPaddle.x, aiPaddle.y, aiPaddle.w, aiPaddle.h);
      ctx.beginPath(); ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2); ctx.fill();
      if (message) { ctx.font = '24px Arial'; ctx.textAlign = 'center'; ctx.fillText(message, canvas.width/2, canvas.height/2); }
    }
    function loop(){ update(); draw(); requestAnimationFrame(loop); }
    document.addEventListener('keydown', e => { if (['ArrowUp','ArrowDown','Space'].includes(e.code)) e.preventDefault(); if (e.key === 'ArrowUp') keys.up = true; if (e.key === 'ArrowDown') keys.down = true; if (e.code === 'Space') restart(); });
    document.addEventListener('keyup', e => { if (e.key === 'ArrowUp') keys.up = false; if (e.key === 'ArrowDown') keys.down = false; });
    canvas.addEventListener('mousemove', e => { const rect = canvas.getBoundingClientRect(); player.y = Math.max(0, Math.min(canvas.height - player.h, e.clientY - rect.top - player.h / 2)); });
    canvas.addEventListener('click', () => canvas.focus());
    loop();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(goal=prompt, mode="game-generator", steps=["Create a Pong HTML game.", "Verify the HTML game structure.", "Start preview server."], suggested_files=[{"path": "pong.html", "content": html}], verify_commands=["Open http://127.0.0.1:9000/pong.html"], notes=["This deterministic template is used for Pong prompts."])


def _tic_tac_toe_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Tic Tac Toe</title>
  <style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f8fafc;color:#0f172a;font-family:Arial,sans-serif}main{text-align:center}.board{display:grid;grid-template-columns:repeat(3,110px);gap:8px}.cell{width:110px;height:110px;font-size:52px;font-weight:700;border:2px solid #334155;background:white;cursor:pointer}.cell:hover{background:#e0f2fe}button{margin-top:18px;padding:10px 18px}</style>
</head>
<body>
  <main>
    <h1>Tic Tac Toe</h1>
    <p id="status">Player X turn</p>
    <div id="board" class="board"></div>
    <button id="reset">Restart</button>
  </main>
  <script>
    const boardEl = document.getElementById('board');
    const statusEl = document.getElementById('status');
    const resetEl = document.getElementById('reset');
    let board, current, locked;
    const wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
    function checkWinner() { for (const [a,b,c] of wins) if (board[a] && board[a] === board[b] && board[a] === board[c]) return board[a]; return board.every(Boolean) ? 'Draw' : null; }
    function render() { boardEl.innerHTML = ''; board.forEach((value, index) => { const cell = document.createElement('button'); cell.className = 'cell'; cell.textContent = value; cell.onclick = () => play(index); boardEl.appendChild(cell); }); }
    function play(index) { if (locked || board[index]) return; board[index] = current; const result = checkWinner(); if (result) { locked = true; statusEl.textContent = result === 'Draw' ? 'Draw game' : 'Player ' + result + ' wins'; } else { current = current === 'X' ? 'O' : 'X'; statusEl.textContent = 'Player ' + current + ' turn'; } render(); }
    function reset() { board = Array(9).fill(''); current = 'X'; locked = false; statusEl.textContent = 'Player X turn'; render(); }
    resetEl.onclick = reset; reset();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(goal=prompt, mode="game-generator", steps=["Create a Tic Tac Toe HTML game.", "Verify the HTML game structure.", "Start preview server."], suggested_files=[{"path": "tic_tac_toe.html", "content": html}], verify_commands=["Open http://127.0.0.1:9000/tic_tac_toe.html"], notes=["This deterministic template is used for Tic Tac Toe prompts."])


def _quiz_app_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Quiz App</title>
  <style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#eef2ff;color:#111827;font-family:Arial,sans-serif}.panel{width:min(620px,92vw);background:white;border:1px solid #c7d2fe;padding:24px}.answers{display:grid;gap:10px;margin:18px 0}button{padding:10px 14px;text-align:left}.primary{text-align:center;background:#4f46e5;color:white;border:0}</style>
</head>
<body>
  <main class="panel">
    <h1>Quiz App</h1>
    <p id="progress"></p>
    <h2 id="question"></h2>
    <div id="answers" class="answers"></div>
    <p id="feedback"></p>
    <button id="next" class="primary">Next</button>
  </main>
  <script>
    const questions = [
      { q: 'What does HTML describe?', a: ['Page structure','Database schema','CPU speed'], correct: 0 },
      { q: 'Which language runs in the browser?', a: ['JavaScript','SQL','Bash'], correct: 0 },
      { q: 'What does CSS control?', a: ['Visual style','Network routing','File permissions'], correct: 0 }
    ];
    let index = 0, score = 0, answered = false;
    const questionEl = document.getElementById('question'), answersEl = document.getElementById('answers'), feedbackEl = document.getElementById('feedback'), progressEl = document.getElementById('progress'), nextEl = document.getElementById('next');
    function showQuestion() { answered = false; feedbackEl.textContent = ''; const item = questions[index]; progressEl.textContent = 'Question ' + (index + 1) + ' of ' + questions.length + ' | Score ' + score; questionEl.textContent = item.q; answersEl.innerHTML = ''; item.a.forEach((answer, choice) => { const btn = document.createElement('button'); btn.textContent = answer; btn.onclick = () => choose(choice); answersEl.appendChild(btn); }); nextEl.textContent = index === questions.length - 1 ? 'Finish' : 'Next'; }
    function choose(choice) { if (answered) return; answered = true; const correct = choice === questions[index].correct; if (correct) score++; feedbackEl.textContent = correct ? 'Correct.' : 'Not quite. Correct answer: ' + questions[index].a[questions[index].correct]; }
    nextEl.onclick = () => { if (!answered) { feedbackEl.textContent = 'Choose an answer first.'; return; } if (index < questions.length - 1) { index++; showQuestion(); } else { questionEl.textContent = 'Final score: ' + score + '/' + questions.length; answersEl.innerHTML = ''; progressEl.textContent = 'Quiz complete'; feedbackEl.textContent = ''; nextEl.textContent = 'Restart'; nextEl.onclick = () => location.reload(); } };
    showQuestion();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(goal=prompt, mode="app-generator", steps=["Create a Quiz HTML app.", "Verify the HTML structure.", "Start preview server."], suggested_files=[{"path": "quiz_app.html", "content": html}], verify_commands=["Open http://127.0.0.1:9000/quiz_app.html"], notes=["This deterministic template is used for quiz app prompts."])


def _todo_app_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Todo App</title>
  <style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#ecfeff;color:#164e63;font-family:Arial,sans-serif}.app{width:min(620px,92vw);background:white;border:1px solid #a5f3fc;padding:24px}form{display:flex;gap:8px}input{flex:1;padding:10px}button{padding:10px 14px}li{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #e5e7eb}</style>
</head>
<body>
  <main class="app">
    <h1>Todo App</h1>
    <form id="form"><input id="task" placeholder="Add a task" /><button>Add</button></form>
    <ul id="list"></ul>
  </main>
  <script>
    const form = document.getElementById('form');
    const taskInput = document.getElementById('task');
    const list = document.getElementById('list');
    let todos = JSON.parse(localStorage.getItem('shamsuTodos') || '[]');
    function save() { localStorage.setItem('shamsuTodos', JSON.stringify(todos)); }
    function renderTodos() { list.innerHTML = ''; todos.forEach((todo, index) => { const li = document.createElement('li'); const label = document.createElement('label'); const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = todo.done; checkbox.onchange = () => { todo.done = checkbox.checked; save(); renderTodos(); }; label.append(checkbox, ' ' + todo.text); if (todo.done) label.style.textDecoration = 'line-through'; const del = document.createElement('button'); del.textContent = 'Delete'; del.onclick = () => { todos.splice(index, 1); save(); renderTodos(); }; li.append(label, del); list.appendChild(li); }); }
    form.onsubmit = event => { event.preventDefault(); const text = taskInput.value.trim(); if (!text) return; todos.push({ text, done: false }); taskInput.value = ''; save(); renderTodos(); };
    renderTodos();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(goal=prompt, mode="app-generator", steps=["Create a Todo HTML app.", "Verify the HTML structure.", "Start preview server."], suggested_files=[{"path": "todo_app.html", "content": html}], verify_commands=["Open http://127.0.0.1:9000/todo_app.html"], notes=["This deterministic template is used for todo app prompts."])


def _calculator_app_plan(prompt: str) -> TaskPlanResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Calculator App</title>
  <style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f1f5f9;color:#0f172a;font-family:Arial,sans-serif}.calc{width:300px;background:white;border:1px solid #cbd5e1;padding:18px}.display{height:54px;border:1px solid #cbd5e1;margin-bottom:12px;padding:12px;text-align:right;font-size:28px}.keys{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}button{padding:14px;font-size:18px}.wide{grid-column:span 2}</style>
</head>
<body>
  <main class="calc">
    <h1>Calculator</h1>
    <div id="display" class="display">0</div>
    <div class="keys" id="keys"></div>
  </main>
  <script>
    const display = document.getElementById('display');
    const keys = document.getElementById('keys');
    let current = '0', stored = null, operator = null, resetNext = false;
    const layout = ['7','8','9','/','4','5','6','*','1','2','3','-','0','.','=','+','C'];
    function update(){ display.textContent = current; }
    function appendDigit(value){ if (resetNext) { current = value === '.' ? '0.' : value; resetNext = false; return update(); } if (value === '.' && current.includes('.')) return; current = current === '0' && value !== '.' ? value : current + value; update(); }
    function calculate(){ const a = Number(stored), b = Number(current); if (operator === '+') current = String(a + b); if (operator === '-') current = String(a - b); if (operator === '*') current = String(a * b); if (operator === '/') current = b === 0 ? 'Error' : String(a / b); stored = null; operator = null; resetNext = true; update(); }
    function chooseOperator(value){ if (stored !== null && !resetNext) calculate(); stored = current; operator = value; resetNext = true; }
    function press(value){ if ('0123456789.'.includes(value)) appendDigit(value); else if ('+-*/'.includes(value)) chooseOperator(value); else if (value === '=') calculate(); else { current = '0'; stored = null; operator = null; resetNext = false; update(); } }
    layout.forEach(value => { const btn = document.createElement('button'); btn.textContent = value; if (value === 'C') btn.className = 'wide'; btn.onclick = () => press(value); keys.appendChild(btn); });
    update();
  </script>
</body>
</html>
"""
    return TaskPlanResponse(goal=prompt, mode="app-generator", steps=["Create a Calculator HTML app.", "Verify the HTML structure.", "Start preview server."], suggested_files=[{"path": "calculator_app.html", "content": html}], verify_commands=["Open http://127.0.0.1:9000/calculator_app.html"], notes=["This deterministic template is used for calculator app prompts."])

def _generated_task_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="json-generator-fallback",
        steps=["Classify the request.", "Ask the local model for a strict JSON file plan.", "Validate paths and file types.", "Write files and run lightweight verification."],
        suggested_files=[],
        verify_commands=[],
        notes=["No fixed template matched; autonomous run will try the JSON file generator fallback."],
    )


def _bugfix_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="bugfix",
        steps=["Run project_index to map files and symbols.", "Use search_files for the error name or stack trace.", "Use read_file_range around failing lines.", "Patch with replace_in_file.", "Run verification."],
        suggested_files=[],
        verify_commands=["python -m pytest -q", "npm test", "python <script>.py"],
        notes=["Bugfix tasks require existing project context and still use the normal approval-gated tools."],
    )


def _large_file_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="large-file-handler",
        steps=["Use project_index to find files.", "Use search_files to locate targets.", "Use read_file_range in bounded windows.", "Patch only exact ranges.", "Run focused verification."],
        suggested_files=[],
        verify_commands=["python -m pytest -q", "python -m py_compile <file>.py"],
        notes=["Do not paste or load a 100000-line file into the model at once."],
    )


def _general_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="coding-task",
        steps=["Use project_index to understand the workspace.", "Identify relevant files/ranges.", "Make changes with write_file or replace_in_file.", "Run verification."],
        suggested_files=[],
        verify_commands=["python -m pytest -q"],
        notes=["This is a planning scaffold for chat/tool mode."],
    )
























