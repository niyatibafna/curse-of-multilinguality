from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import EmbeddingModel


class OpenAITextEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name_or_path: str = "text-embedding-3-large",
        api_key_env: str = "OPENAI_KEY",
        dimensions: int | None = None,
        **_: Any,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install `openai` to use OpenAITextEmbeddingModel.") from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Set {api_key_env} to use OpenAITextEmbeddingModel.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name_or_path
        self.dimensions = dimensions

    def encode(self, inputs: list[str], batch_size: int = 128, **request_kwargs: Any) -> np.ndarray:
        if isinstance(inputs, str):
            inputs = [inputs]
        request_kwargs.pop("pooling", None)

        embeddings = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            kwargs = {
                "model": self.model_name,
                "input": batch,
                **request_kwargs,
            }
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**kwargs)
            embeddings.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))

        return np.asarray(embeddings, dtype=np.float32)
