from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from .. import model_registry

router = APIRouter(prefix="/api/models", tags=["models"])


class SetModelRequest(BaseModel):
    model_id: str


@router.get("")
async def list_models() -> dict[str, object]:
    return {
        "current": model_registry.get_current_model(),
        "models": model_registry.list_models(),
    }


@router.post("/current")
async def set_current_model(body: SetModelRequest) -> dict[str, object]:
    try:
        current = model_registry.set_current_model(body.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "current": current,
        "models": model_registry.list_models(),
    }
