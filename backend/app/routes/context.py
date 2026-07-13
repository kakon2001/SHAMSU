from fastapi import APIRouter, Query

from .. import context_index

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
