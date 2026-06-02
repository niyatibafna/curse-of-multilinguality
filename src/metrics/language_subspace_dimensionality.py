from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import COMMetric
from .utils import (
    add_effective_dim_baseline,
    pairwise_displacement_effective_rank,
)


class LanguageSpaceDimGrowthByLanguage(COMMetric):
    """Effective dimension of same-concept cross-language displacements."""

    def compute(self) -> dict[str, Any]:
        X = self._stack_by_language()
        if self.normalize:
            X = self._normalize_embeddings(X)

        ordered_indices = self._language_order_indices()
        language_order = [list(self.X.keys())[index] for index in ordered_indices]
        pool = X.reshape(-1, self.embedding_dim)
        rng = self._random_baseline_rng()
        report = []
        for num_languages in self._language_counts():
            effective_dim = self._effective_dim_for_prefix(
                X,
                ordered_indices[:num_languages],
            )
            row = {
                "num_languages": num_languages,
                "effective_dim": effective_dim,
            }
            baseline = self._random_baseline(
                pool=pool,
                group_sizes=[num_languages] * self.num_concepts,
                rng=rng,
                normalize_by_dim=self._normalize_effective_dim(),
            )
            report.append(add_effective_dim_baseline(row, effective_dim, baseline))

        result: dict[str, Any] = {
            "language_subspace_scaling": report,
            "language_order": language_order,
        }
        if self.return_details:
            result["details"] = {
                "num_languages": self.num_languages,
                "num_concepts": self.num_concepts,
                "embedding_dim": self.embedding_dim,
                "language_order_seed": int(self.kwargs.get("language_order_seed", 0)),
                "effective_rank_method": self.kwargs.get("effective_rank_method", "stable"),
                "normalize_effective_dim": self._normalize_effective_dim(),
                "random_baseline_trials": self._random_baseline_trials(),
                "random_baseline_seed": self._random_baseline_seed(),
                "normalize": self.normalize,
            }
        return result

    def _stack_by_language(self) -> np.ndarray:
        arrays = []
        for language, embeddings in self.X.items():
            arr = np.asarray(embeddings, dtype=float)
            expected_shape = (self.num_concepts, self.embedding_dim)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Expected X[{language!r}] to have shape {expected_shape}, got {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"X[{language!r}] contains NaN or infinite values.")
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    def _language_counts(self) -> list[int]:
        if self.num_languages < 2:
            raise ValueError("Language subspace scaling requires at least two languages.")

        counts = [2]
        if self.num_languages >= 5:
            counts.append(5)
        counts.extend(range(10, self.num_languages + 1, 5))
        if counts[-1] != self.num_languages:
            counts.append(self.num_languages)
        return counts

    def _effective_dim_for_prefix(self, X: np.ndarray, selected: np.ndarray) -> float:
        return pairwise_displacement_effective_rank(
            (X[selected, concept_index, :] for concept_index in range(self.num_concepts)),
            embedding_dim=self.embedding_dim,
            normalize_by_dim=self._normalize_effective_dim(),
            **self._effective_rank_kwargs(),
        )

    def _language_order_indices(self) -> np.ndarray:
        seed = int(self.kwargs.get("language_order_seed", 0))
        rng = np.random.default_rng(seed)
        return rng.permutation(self.num_languages)

    def _normalize_embeddings(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=-1, keepdims=True)
        return X / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)

    def _normalize_effective_dim(self) -> bool:
        return bool(self.kwargs.get("normalize_effective_dim", True))


class LanguageSpaceGrowthByConcepts(COMMetric):
    """Language-displacement dimensionality as concepts are added."""

    def compute(self) -> dict[str, Any]:
        X = self._stack_by_language()
        if self.normalize:
            X = self._normalize_embeddings(X)

        concept_step = int(self.kwargs.get("concept_step", 25))
        if concept_step <= 0:
            raise ValueError("concept_step must be positive.")

        pool = X.reshape(-1, self.embedding_dim)
        rng = self._random_baseline_rng()
        report = []
        for num_concepts in self._concept_counts(concept_step):
            effective_dim = self._effective_dim_for_concepts(X, num_concepts)
            row = {
                "num_concepts": num_concepts,
                "effective_dim": effective_dim,
            }
            baseline = self._random_baseline(
                pool=pool,
                group_sizes=[self.num_languages] * num_concepts,
                rng=rng,
                normalize_by_dim=self._normalize_effective_dim(),
            )
            report.append(add_effective_dim_baseline(row, effective_dim, baseline))

        result: dict[str, Any] = {
            "language_space_growth_by_concepts": report,
        }
        if self.return_details:
            result["details"] = {
                "num_languages": self.num_languages,
                "num_concepts": self.num_concepts,
                "embedding_dim": self.embedding_dim,
                "concept_step": concept_step,
                "effective_rank_method": self.kwargs.get("effective_rank_method", "stable"),
                "normalize_effective_dim": self._normalize_effective_dim(),
                "random_baseline_trials": self._random_baseline_trials(),
                "random_baseline_seed": self._random_baseline_seed(),
                "normalize": self.normalize,
            }
        return result

    def _stack_by_language(self) -> np.ndarray:
        arrays = []
        for language, embeddings in self.X.items():
            arr = np.asarray(embeddings, dtype=float)
            expected_shape = (self.num_concepts, self.embedding_dim)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Expected X[{language!r}] to have shape {expected_shape}, got {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"X[{language!r}] contains NaN or infinite values.")
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    def _concept_counts(self, concept_step: int) -> list[int]:
        counts = list(range(concept_step, self.num_concepts + 1, concept_step))
        if not counts or counts[-1] != self.num_concepts:
            counts.append(self.num_concepts)
        return counts

    def _effective_dim_for_concepts(self, X: np.ndarray, num_concepts: int) -> float:
        return pairwise_displacement_effective_rank(
            (X[:, concept_index, :] for concept_index in range(num_concepts)),
            embedding_dim=self.embedding_dim,
            normalize_by_dim=self._normalize_effective_dim(),
            **self._effective_rank_kwargs(),
        )

    def _normalize_embeddings(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=-1, keepdims=True)
        return X / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)

    def _normalize_effective_dim(self) -> bool:
        return bool(self.kwargs.get("normalize_effective_dim", True))
