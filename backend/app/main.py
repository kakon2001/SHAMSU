from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import agent, files

app = FastAPI(title="Local Coding Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    # The dev server may be opened as localhost or 127.0.0.1 — allow both on any port.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router)
app.include_router(files.router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "workspace": str(settings.workdir_path)}
