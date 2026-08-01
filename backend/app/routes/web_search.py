from __future__ import annotations

from fastapi import APIRouter, Query

from ..web_search import search_web

router = APIRouter(prefix="/api/web", tags=["web"])


@router.get("/search")
def web_search(query: str = Query(..., min_length=1), limit: int = Query(5, ge=1, le=10)) -> dict:
    return search_web(query, limit=limit)
