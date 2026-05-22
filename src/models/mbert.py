from __future__ import annotations

from typing import Any, Literal

from .base import EmbeddingModel


class MBertEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name_or_path: str = "bert-base-multilingual-cased",
        layer: int = -1,
        device: str | None = None,
        **model_kwargs: Any,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install `torch` and `transformers` to use MBertEmbeddingModel.") from exc

        self.torch = torch
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs).to(self.device)
        self.model.eval()

    def encode(
        self,
        inputs: list[str],
        batch_size: int = 32,
        pooling: Literal["cls", "last_token", "mean", "tokens"] = "cls",
        **tokenizer_kwargs: Any,
    ):
        if isinstance(inputs, str):
            inputs = [inputs]

        outputs = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                **tokenizer_kwargs,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with self.torch.no_grad():
                model_output = self.model(**encoded, output_hidden_states=True)

            hidden = model_output.hidden_states[self.layer]
            outputs.append(self._pool(hidden, encoded["attention_mask"], pooling).cpu())

        return self.torch.cat(outputs, dim=0)

    def _pool(self, hidden, attention_mask, pooling: str):
        if pooling == "tokens":
            return hidden
        if pooling == "cls":
            return hidden[:, 0]
        if pooling == "last_token":
            indices = attention_mask.sum(dim=1) - 1
            batch_indices = self.torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch_indices, indices]
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        raise ValueError(f"Unknown pooling strategy: {pooling}")
