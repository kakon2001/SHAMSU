from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class TaskPlanRequest(BaseModel):
    prompt: str


class TaskPlanResponse(BaseModel):
    goal: str
    mode: str
    steps: list[str]
    suggested_files: list[dict[str, str]]
    verify_commands: list[str]
    notes: list[str]


@router.post("/plan", response_model=TaskPlanResponse)
async def plan_task(body: TaskPlanRequest) -> TaskPlanResponse:
    prompt = body.prompt.strip()
    lower = prompt.lower()
    if any(word in lower for word in ["game", "bouncing", "ball", "canvas"]):
        return _game_plan(prompt)
    if any(word in lower for word in ["bug", "fix", "error", "traceback", "failing"]):
        return _bugfix_plan(prompt)
    if any(word in lower for word in ["large file", "100000", "100,000", "huge file"]):
        return _large_file_plan(prompt)
    return _general_plan(prompt)


def _game_plan(prompt: str) -> TaskPlanResponse:
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
        steps=[
            "Create an HTML canvas game file in the workspace.",
            "Open the file in a browser or serve the workspace with a simple local server.",
            "Verify the ball moves and bounces off all walls.",
            "Iterate by adding controls, score, obstacles, or styling.",
        ],
        suggested_files=[{"path": "bouncing_ball.html", "content": html}],
        verify_commands=["python -m http.server 9000", "Open http://127.0.0.1:9000/bouncing_ball.html"],
        notes=["This deterministic plan gives code and verification steps even if the model does not call write_file."],
    )


def _bugfix_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="bugfix",
        steps=[
            "Run project_index to map files and symbols.",
            "Use search_files for the error name, failing function, or stack trace line.",
            "Use read_file_range around the failing line instead of reading the whole file.",
            "Use replace_in_file for a small patch and review the diff.",
            "Run the relevant test or script and repeat if it fails.",
        ],
        suggested_files=[],
        verify_commands=["python -m pytest -q", "npm test", "python <script>.py"],
        notes=["Choose the verification command that matches the project."],
    )


def _large_file_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="large-file-handler",
        steps=[
            "Use project_index to find candidate files and symbol names.",
            "Use search_files to locate exact error text or function names.",
            "Use read_file_range in 200-500 line windows around matches.",
            "Patch only the target range with replace_in_file after reviewing context.",
            "Run a focused verification command.",
        ],
        suggested_files=[],
        verify_commands=["python -m pytest -q", "python -m py_compile <file>.py"],
        notes=["Do not paste or load a 100000-line file into the model at once."],
    )


def _general_plan(prompt: str) -> TaskPlanResponse:
    return TaskPlanResponse(
        goal=prompt,
        mode="coding-task",
        steps=[
            "Use project_index to understand the current workspace.",
            "Identify the smallest files/ranges needed for the task.",
            "Make changes with write_file for new files or replace_in_file for edits.",
            "Run verification and report pass/fail clearly.",
        ],
        suggested_files=[],
        verify_commands=["python -m pytest -q"],
        notes=["This is a reliable workflow scaffold when the local model is unsure."],
    )
