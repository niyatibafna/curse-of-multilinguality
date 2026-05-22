from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import COMMetric


class Anisotropy(COMMetric):
    """Mean off-diagonal pairwise cosine similarity of embeddings."""

    def compute(self) -> float | tuple[float, dict[str, Any]]:
        embeddings = np.vstack([np.asarray(x) for x in self.X.values()])
        score = self._score_embeddings(embeddings)

        if self.return_details:
            pass

        return score

    def _score_embeddings(self, embeddings: np.ndarray) -> float:
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim != 2:
            raise ValueError("Expected embeddings to have shape (num_concepts, embedding_dim).")
        if embeddings.shape[0] < 2:
            raise ValueError("Anisotropy requires at least two concepts.")

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)

        gram_sum = float(np.sum(embeddings @ embeddings.T))
        diagonal_sum = float(np.sum(embeddings * embeddings))
        num_pairs = embeddings.shape[0] * (embeddings.shape[0] - 1)

        return (gram_sum - diagonal_sum) / num_pairs
