from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import COMMetric
from .utils import (
    add_effective_dim_baseline,
    pairwise_displacement_effective_rank,
)


class IndividualLanguageConceptDimensionality(COMMetric):
    """Same-language concept-displacement dimensionality for each language."""

    def compute(self) -> dict[str, Any]:
        dims = {
            language: self._effective_dim(embeddings)
            for language, embeddings in self.X.items()
        }
        sorted_dims = [
            {"language": language, "effective_dim": dim}
            for language, dim in sorted(dims.items(), key=lambda item: item[1], reverse=True)
        ]

        result: dict[str, Any] = {
            "effective_dim_by_language": dims,
            "sorted_effective_dims": sorted_dims,
        }
        if self.return_details:
            result["details"] = self._details()
        return result

    def _effective_dim(self, embeddings: np.ndarray) -> float:
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.shape != (self.num_concepts, self.embedding_dim):
            raise ValueError(
                f"Expected language embeddings to have shape "
                f"{(self.num_concepts, self.embedding_dim)}, got {embeddings.shape}."
            )
        if self.normalize:
            embeddings = _normalize_rows(embeddings)

        return pairwise_displacement_effective_rank(
            [embeddings],
            embedding_dim=self.embedding_dim,
            normalize_by_dim=self._normalize_effective_dim(),
            **self._effective_rank_kwargs(),
        )

    def _details(self) -> dict[str, Any]:
        return {
            "num_languages": self.num_languages,
            "num_concepts": self.num_concepts,
            "embedding_dim": self.embedding_dim,
            "languages": list(self.X.keys()),
            "effective_rank_method": self.kwargs.get("effective_rank_method", "stable"),
            "normalize_effective_dim": self._normalize_effective_dim(),
            "normalize": self.normalize,
        }

    def _normalize_effective_dim(self) -> bool:
        return bool(self.kwargs.get("normalize_effective_dim", True))


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)


class ConceptSpaceDimGrowthByLanguage(COMMetric):
    """Concept-displacement dimensionality as languages are added."""

    def compute(self) -> dict[str, Any]:
        ordered_languages = self._language_order()
        pool = self._embedding_pool()
        rng = self._random_baseline_rng()

        report = []
        for num_languages in self._language_counts():
            effective_dim = self._effective_dim_for_languages(
                ordered_languages[:num_languages]
            )
            row = {
                "num_languages": num_languages,
                "effective_dim": effective_dim,
            }
            baseline = self._random_baseline(
                pool=pool,
                group_sizes=[self.num_concepts] * num_languages,
                rng=rng,
                normalize_by_dim=self._normalize_effective_dim(),
            )
            report.append(add_effective_dim_baseline(row, effective_dim, baseline))

        result: dict[str, Any] = {
            "concept_space_dim_growth_by_language": report,
            "language_order": ordered_languages,
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

    def _language_counts(self) -> list[int]:
        counts = [1]
        if self.num_languages >= 2:
            counts.append(2)
        if self.num_languages >= 5:
            counts.append(5)
        counts.extend(range(10, self.num_languages + 1, 5))
        if counts[-1] != self.num_languages:
            counts.append(self.num_languages)
        return counts

    def _language_order(self) -> list[str]:
        languages = list(self.X.keys())
        seed = int(self.kwargs.get("language_order_seed", 0))
        rng = np.random.default_rng(seed)
        return [languages[index] for index in rng.permutation(len(languages))]

    def _effective_dim_for_languages(self, languages: list[str]) -> float:
        groups = []
        for language in languages:
            embeddings = np.asarray(self.X[language], dtype=float)
            expected_shape = (self.num_concepts, self.embedding_dim)
            if embeddings.shape != expected_shape:
                raise ValueError(
                    f"Expected X[{language!r}] to have shape {expected_shape}, "
                    f"got {embeddings.shape}."
                )
            if not np.all(np.isfinite(embeddings)):
                raise ValueError(f"X[{language!r}] contains NaN or infinite values.")
            if self.normalize:
                embeddings = _normalize_rows(embeddings)
            groups.append(embeddings)

        return pairwise_displacement_effective_rank(
            groups,
            embedding_dim=self.embedding_dim,
            normalize_by_dim=self._normalize_effective_dim(),
            **self._effective_rank_kwargs(),
        )

    def _normalize_effective_dim(self) -> bool:
        return bool(self.kwargs.get("normalize_effective_dim", True))

    def _embedding_pool(self) -> np.ndarray:
        pool = []
        for language_embeddings in self.X.values():
            embeddings = np.asarray(language_embeddings, dtype=float)
            if self.normalize:
                embeddings = _normalize_rows(embeddings)
            pool.append(embeddings)
        return np.vstack(pool)


class ConceptSpaceDimGrowthByConcept(COMMetric):
    """Concept-displacement dimensionality as concepts are added."""

    def compute(self) -> dict[str, Any]:
        concept_step = int(self.kwargs.get("concept_step", 25))
        if concept_step <= 0:
            raise ValueError("concept_step must be positive.")

        pool = self._embedding_pool()
        rng = self._random_baseline_rng()
        report = []
        for num_concepts in self._concept_counts(concept_step):
            effective_dim = self._effective_dim_for_concepts(num_concepts)
            row = {
                "num_concepts": num_concepts,
                "effective_dim": effective_dim,
            }
            baseline = self._random_baseline(
                pool=pool,
                group_sizes=[num_concepts] * self.num_languages,
                rng=rng,
                normalize_by_dim=self._normalize_effective_dim(),
            )
            report.append(add_effective_dim_baseline(row, effective_dim, baseline))

        result: dict[str, Any] = {
            "concept_space_dim_growth_by_concept": report,
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

    def _concept_counts(self, concept_step: int) -> list[int]:
        counts = list(range(concept_step, self.num_concepts + 1, concept_step))
        if not counts or counts[-1] != self.num_concepts:
            counts.append(self.num_concepts)
        return counts

    def _effective_dim_for_concepts(self, num_concepts: int) -> float:
        groups = []
        for language, language_embeddings in self.X.items():
            embeddings = np.asarray(language_embeddings[:num_concepts], dtype=float)
            expected_shape = (num_concepts, self.embedding_dim)
            if embeddings.shape != expected_shape:
                raise ValueError(
                    f"Expected X[{language!r}][:{num_concepts}] to have shape "
                    f"{expected_shape}, got {embeddings.shape}."
                )
            if not np.all(np.isfinite(embeddings)):
                raise ValueError(f"X[{language!r}] contains NaN or infinite values.")
            if self.normalize:
                embeddings = _normalize_rows(embeddings)
            groups.append(embeddings)

        return pairwise_displacement_effective_rank(
            groups,
            embedding_dim=self.embedding_dim,
            normalize_by_dim=self._normalize_effective_dim(),
            **self._effective_rank_kwargs(),
        )

    def _normalize_effective_dim(self) -> bool:
        return bool(self.kwargs.get("normalize_effective_dim", True))

    def _embedding_pool(self) -> np.ndarray:
        pool = []
        for language_embeddings in self.X.values():
            embeddings = np.asarray(language_embeddings, dtype=float)
            if self.normalize:
                embeddings = _normalize_rows(embeddings)
            pool.append(embeddings)
        return np.vstack(pool)
