from __future__ import annotations

from dataclasses import dataclass

from .config import settings


@dataclass(frozen=True)
class LocalModel:
    id: str
    label: str
    size: str
    description: str


AVAILABLE_MODELS = [
    LocalModel("qwen3:8b", "Qwen3 8B", "8B", "Best quality, slower on smaller laptops."),
    LocalModel("qwen3:4b", "Qwen3 4B", "4B", "Requirement-sized balanced local model."),
    LocalModel("qwen3-8k:1.7b", "Qwen3 1.7B", "1.7B", "Fast demo model with smaller responses."),
]

_current_model = settings.model_name


def list_models() -> list[dict[str, str | bool]]:
    return [
        {
            "id": model.id,
            "label": model.label,
            "size": model.size,
            "description": model.description,
            "active": model.id == _current_model,
        }
        for model in AVAILABLE_MODELS
    ]


def get_current_model() -> str:
    return _current_model


def set_current_model(model_id: str) -> str:
    global _current_model
    known = {model.id for model in AVAILABLE_MODELS}
    if model_id not in known:
        raise ValueError(f"Unknown model '{model_id}'. Choose one of: {', '.join(sorted(known))}")
    _current_model = model_id
    settings.model_name = model_id
    return _current_model
