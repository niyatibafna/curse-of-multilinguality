from __future__ import annotations

from typing import Any

from .base import EmbeddingModel


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        normalize_embeddings: bool = False,
        prefix: str = "",
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install `sentence-transformers` to use SentenceTransformerEmbeddingModel."
            ) from exc

        model_kwargs.pop("layer", None)
        self.normalize_embeddings = normalize_embeddings
        self.prefix = prefix
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True

        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            **model_kwargs,
        )

    def encode(self, inputs: list[str], batch_size: int = 32, **encode_kwargs: Any):
        if isinstance(inputs, str):
            inputs = [inputs]
        encode_kwargs.pop("pooling", None)

        if self.prefix:
            inputs = [self.prefix + text for text in inputs]

        return self.model.encode(
            inputs,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            **encode_kwargs,
        )
