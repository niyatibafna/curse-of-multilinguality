from __future__ import annotations

from typing import Any

from .base import EmbeddingModel
from .llama import LlamaEmbeddingModel
from .mbert import MBertEmbeddingModel
from .mistral import MistralEmbeddingModel
from .openai_text_embedding import OpenAITextEmbeddingModel
from .sentence_transformer import SentenceTransformerEmbeddingModel


MODEL_REGISTRY: dict[str, tuple[type[EmbeddingModel], dict[str, Any]]] = {
    "llama": (
        LlamaEmbeddingModel,
        {"model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct", "dtype": "auto"},
    ),
    "mbert": (MBertEmbeddingModel, {"model_name_or_path": "bert-base-multilingual-uncased"}),
    "mistral": (
        MistralEmbeddingModel,
        {"model_name_or_path": "mistralai/Ministral-8B-Instruct-2410", "dtype": "auto"},
    ),
    "openai-large": (OpenAITextEmbeddingModel, {"model_name_or_path": "text-embedding-3-large"}),
    "minilm": (SentenceTransformerEmbeddingModel, {"model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2"}),
    "bge-small": (SentenceTransformerEmbeddingModel, {"model_name_or_path": "BAAI/bge-small-en-v1.5"}),
    "bge-base": (SentenceTransformerEmbeddingModel, {"model_name_or_path": "BAAI/bge-base-en-v1.5"}),
    "bge-m3": (SentenceTransformerEmbeddingModel, {"model_name_or_path": "BAAI/bge-m3"}),
    "e5-base": (SentenceTransformerEmbeddingModel, {"model_name_or_path": "intfloat/e5-base-v2"}),
    "e5-large": (SentenceTransformerEmbeddingModel, {"model_name_or_path": "intfloat/e5-large-v2"}),
    "multilingual-e5-large": (
        SentenceTransformerEmbeddingModel,
        {"model_name_or_path": "intfloat/multilingual-e5-large", "prefix": "query: "},
    ),
    "jina-v4": (
        SentenceTransformerEmbeddingModel,
        {
            "model_name_or_path": "jinaai/jina-embeddings-v4",
            "trust_remote_code": True,
            "default_encode_kwargs": {"task": "text-matching"},
        },
    ),
    "nomic": (
        SentenceTransformerEmbeddingModel,
        {"model_name_or_path": "nomic-ai/nomic-embed-text-v1.5", "trust_remote_code": True},
    ),
}


def get_model_spec(keyword: str) -> tuple[type[EmbeddingModel], dict[str, Any]]:
    key = keyword.lower()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{keyword}'. Available models: {valid}") from exc


def get_model_class(keyword: str) -> type[EmbeddingModel]:
    model_cls, _ = get_model_spec(keyword)
    return model_cls


def load_model(keyword: str, **kwargs: Any) -> EmbeddingModel:
    model_cls, model_kwargs = get_model_spec(keyword)
    return model_cls(**{**model_kwargs, **kwargs})
