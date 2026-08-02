from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from .. import context_index, vector_index, long_context

router = APIRouter(prefix="/api/context", tags=["context"])


@router.get("/summary")
async def context_summary() -> dict[str, object]:
    return context_index.summarize_workspace()


@router.get("/search")
async def context_search(query: str = Query(...), limit: int = Query(5, ge=1, le=20)) -> dict[str, object]:
    matches = context_index.search_context(query, limit=limit)
    return {
        "query": query,
        "matches": [
            {
                "path": match.path,
                "start_line": match.start_line,
                "end_line": match.end_line,
                "score": match.score,
                "text": match.text,
            }
            for match in matches
        ],
    }


@router.get("/auto")
async def automatic_context(query: str = Query(...), limit: int = Query(6, ge=1, le=20)) -> dict[str, object]:
    return {
        "query": query,
        "context": context_index.automatic_context(query, limit=limit),
    }


@router.get("/dashboard")
async def context_dashboard() -> dict[str, object]:
    return context_index.context_dashboard()


@router.get("/overview")
async def context_overview(query: str = Query("")) -> dict[str, object]:
    return {
        "query": query,
        "overview": context_index.automatic_summary_context(query),
    }


@router.get("/project-map")
async def context_project_map(limit: int = Query(80, ge=1, le=200)) -> dict[str, object]:
    return context_index.project_map(limit=limit)

class VectorRebuildRequest(BaseModel):
    limit_files: int = Field(default=500, ge=1, le=2000)


@router.get("/vector/stats")
async def vector_stats() -> dict[str, object]:
    return vector_index.stats()


@router.post("/vector/rebuild")
async def vector_rebuild(request: VectorRebuildRequest) -> dict[str, object]:
    return vector_index.rebuild_index(limit_files=request.limit_files)


@router.get("/vector/search")
async def vector_search(query: str = Query(...), limit: int = Query(8, ge=1, le=20)) -> dict[str, object]:
    return {"query": query, "matches": vector_index.semantic_search(query, limit=limit)}



@router.get("/bundle")
async def context_bundle(query: str = Query(...), budget: int = Query(7000, ge=2500, le=16000)) -> dict[str, object]:
    return long_context.build_bundle(query, budget=budget, fast=True)