from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import COMMetric


class Noncollapse(COMMetric):
    """Pairwise cosine-similarity summary over all evaluated embeddings."""

    def compute(self) -> dict[str, Any]:
        embeddings = self._normalize_rows(self._stack_embeddings())

        stats = self._pairwise_cosine_stats(embeddings)
        return {
            "score": stats["mean"],
            "mean": stats["mean"],
            "min": stats["min"],
            "max": stats["max"],
            "std_dev": stats["std_dev"],
            "std": stats["std_dev"],
            "num_embeddings": int(embeddings.shape[0]),
            "num_pairs": stats["num_pairs"],
            "num_languages": self.num_languages,
            "num_concepts": self.num_concepts,
            "embedding_dim": self.embedding_dim,
            "languages": list(self.X.keys()),
            "similarity": "cosine",
            "batch_size": self._batch_size(),
        }

    def _stack_embeddings(self) -> np.ndarray:
        expected_shape = (self.num_concepts, self.embedding_dim)
        arrays = []
        for language, embeddings in self.X.items():
            arr = np.asarray(embeddings, dtype=float)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Expected X[{language!r}] to have shape {expected_shape}, got {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"X[{language!r}] contains NaN or infinite values.")
            arrays.append(arr)
        if not arrays:
            raise ValueError("Noncollapse requires embeddings.")
        stacked = np.vstack(arrays)
        if stacked.shape[0] < 2:
            raise ValueError("Noncollapse requires at least two embeddings.")
        return stacked

    def _pairwise_cosine_stats(self, embeddings: np.ndarray) -> dict[str, float | int]:
        batch_size = self._batch_size()
        total = embeddings.shape[0]
        count = 0
        sim_sum = 0.0
        sim_sq_sum = 0.0
        sim_min = np.inf
        sim_max = -np.inf

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            block = embeddings[start:end]

            within = block @ block.T
            tri = within[np.triu_indices(end - start, k=1)]
            if tri.size:
                count, sim_sum, sim_sq_sum, sim_min, sim_max = self._update_stats(
                    tri, count, sim_sum, sim_sq_sum, sim_min, sim_max
                )

            for other_start in range(end, total, batch_size):
                other_end = min(other_start + batch_size, total)
                cross = block @ embeddings[other_start:other_end].T
                count, sim_sum, sim_sq_sum, sim_min, sim_max = self._update_stats(
                    cross, count, sim_sum, sim_sq_sum, sim_min, sim_max
                )

        mean = sim_sum / count
        variance = max((sim_sq_sum / count) - (mean * mean), 0.0)
        return {
            "mean": float(mean),
            "min": float(sim_min),
            "max": float(sim_max),
            "std_dev": float(np.sqrt(variance)),
            "num_pairs": int(count),
        }

    def _update_stats(
        self,
        similarities: np.ndarray,
        count: int,
        sim_sum: float,
        sim_sq_sum: float,
        sim_min: float,
        sim_max: float,
    ) -> tuple[int, float, float, float, float]:
        values = np.asarray(similarities, dtype=float)
        return (
            count + int(values.size),
            sim_sum + float(np.sum(values)),
            sim_sq_sum + float(np.sum(values * values)),
            min(sim_min, float(np.min(values))),
            max(sim_max, float(np.max(values))),
        )

    def _batch_size(self) -> int:
        batch_size = int(self.kwargs.get("noncollapse_batch_size", 2048))
        if batch_size <= 0:
            raise ValueError("noncollapse_batch_size must be positive.")
        return batch_size

    def _normalize_rows(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)
