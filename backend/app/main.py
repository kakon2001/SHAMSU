from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .agent.session_manager import manager
from .config import settings
from .routes import agent, context, files


def configure_activity_logging() -> None:
    logger = logging.getLogger("agent.activity")
    logger.setLevel(logging.INFO)
    if any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        return
    handler = logging.FileHandler(settings.activity_log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


configure_activity_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    await manager.load_from_db()
    yield
    await db.close_db()


app = FastAPI(title="Local Coding Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    # The dev server may be opened as localhost or 127.0.0.1 — allow both on any port.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router)
app.include_router(context.router)
app.include_router(files.router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "workspace": str(settings.workdir_path),
        "model": settings.model_name,
        "history_store": db.storage_mode(),
        "activity_log": str(settings.activity_log_file),
    }
