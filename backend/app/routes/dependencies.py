from __future__ import annotations

import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import dependencies

router = APIRouter(prefix="/api/dependencies", tags=["dependencies"])


class DependencyPlanRequest(BaseModel):
    prompt: str
    target: str = "auto"


class DependencyInstallRequest(BaseModel):
    manager: str
    packages: list[str] = Field(default_factory=list)
    project_path: str = "."
    approve: bool = False
    update_manifest: bool = True
    run_verification: bool = False


@router.get("/scan")
def scan_dependencies() -> dict[str, Any]:
    return dependencies.scan_projects()


@router.post("/plan")
def plan_dependencies(body: DependencyPlanRequest) -> dict[str, Any]:
    return dependencies.plan_dependencies(body.prompt, body.target)


@router.post("/install")
def install_dependencies(body: DependencyInstallRequest) -> dict[str, Any]:
    try:
        return dependencies.install_dependencies(
            manager=body.manager,
            packages=body.packages,
            project_path=body.project_path,
            approve=body.approve,
            update_manifest=body.update_manifest,
            run_verification=body.run_verification,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except subprocess.TimeoutExpired as exc:  # type: ignore[name-defined]
        raise HTTPException(status_code=408, detail=f"Dependency install timed out: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dependency install failed: {exc}") from exc
