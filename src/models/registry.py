from __future__ import annotations

from typing import Any

from .base import EmbeddingModel
from .llama import LlamaEmbeddingModel


MODEL_REGISTRY: dict[str, type[EmbeddingModel]] = {
    "llama": LlamaEmbeddingModel,
}


def get_model_class(keyword: str) -> type[EmbeddingModel]:
    key = keyword.lower()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{keyword}'. Available models: {valid}") from exc


def load_model(keyword: str, **kwargs: Any) -> EmbeddingModel:
    return get_model_class(keyword)(**kwargs)
