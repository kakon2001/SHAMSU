from __future__ import annotations

import json
import re
from typing import Any

import ollama
from fastapi import APIRouter
from pydantic import BaseModel

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


class TaskRunRequest(BaseModel):
    prompt: str
    preview: bool = True
    overwrite: bool = True


class TaskRunStep(BaseModel):
    name: str
    status: str
    detail: str


class TaskRunResponse(BaseModel):
    goal: str
    mode: str
    ok: bool
    created_files: list[str]
    preview_url: str | None
    steps: list[TaskRunStep]
    notes: list[str]


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
    )


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
        re.search(r"\b(game|app|application|website|web page|html|system|tool|program|project|calculator|todo|quiz|os|operating system)\b", text)
    )


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











