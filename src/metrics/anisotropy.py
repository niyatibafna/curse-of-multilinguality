from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .metrics import COMMetric


class Anisotropy(COMMetric):
    """Mean off-diagonal pairwise cosine similarity of embeddings."""

    def compute(self) -> float | tuple[float, dict[str, Any]]:
        score, num_embeddings, embedding_dim = self._score_embedding_blocks(self.X.values())

        if self.return_details:
            return score, {
                "num_embeddings": num_embeddings,
                "embedding_dim": embedding_dim,
                "num_languages": self.num_languages,
                "num_concepts": self.num_concepts,
                "languages": list(self.X.keys()),
                "normalize": self.normalize,
            }

        return score

    def _score_embeddings(self, embeddings: np.ndarray) -> float:
        score, _, _ = self._score_embedding_blocks([embeddings])
        return score

    def _score_embedding_blocks(
        self,
        blocks: Iterable[np.ndarray],
    ) -> tuple[float, int, int]:
        total = 0
        embedding_dim: int | None = None
        vector_sum: np.ndarray | None = None
        squared_norm_sum = 0.0

        for block in blocks:
            embeddings = np.asarray(block, dtype=float)
            if embeddings.ndim != 2:
                raise ValueError("Expected embeddings to have shape (num_concepts, embedding_dim).")

            if embedding_dim is None:
                embedding_dim = embeddings.shape[1]
                vector_sum = np.zeros(embedding_dim, dtype=float)
            elif embeddings.shape[1] != embedding_dim:
                raise ValueError("All embeddings must have the same embedding dimension.")

            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)

            total += embeddings.shape[0]
            vector_sum += np.sum(embeddings, axis=0)
            squared_norm_sum += float(np.sum(embeddings * embeddings))

        if embedding_dim is None or vector_sum is None:
            raise ValueError("Anisotropy requires embeddings.")
        if total < 2:
            raise ValueError("Anisotropy requires at least two concepts.")

        gram_sum = float(vector_sum @ vector_sum)
        num_pairs = total * (total - 1)

        return (gram_sum - squared_norm_sum) / num_pairs, total, embedding_dim
