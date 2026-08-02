from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .. import reports

router = APIRouter(prefix="/api/reports", tags=["reports"])


class ReportQueryRequest(BaseModel):
    sql: str
    title: str = ""
    limit: int = Field(default=100, ge=1, le=500)
    save: bool = True


@router.get("/schema")
async def report_schema() -> dict[str, object]:
    return reports.schema_overview()


@router.post("/query")
async def report_query(body: ReportQueryRequest) -> dict[str, object]:
    try:
        return reports.run_report_query(body.sql, limit=body.limit, title=body.title, save=body.save)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report query failed: {exc}") from exc


@router.get("/history")
async def report_history(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    return {"reports": reports.report_history(limit=limit)}
