from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import COMMetric
from .utils import (
    pairwise_displacement_effective_rank,
    pairwise_displacement_subspace_basis,
    stack_language_embeddings,
)


class Comness(COMMetric):
    """
    Multilingual overhead / cross-lingual alignment diagnostic.

    This metric asks: of the effective dimensions spanned by observed
    variation, what fraction is taken up by language differences rather than
    concept differences?

    Computes:

        COM(X) = d_lang / (d_lang + d_concept)

    where:
        d_lang    = effrank({x[c, l] - x[c, m]     : l != m})
        d_concept = effrank({x[c, l] - x[c', l]    : c != c'})

    Small scores mean language variation occupies few effective dimensions
    relative to concept variation. Large scores mean language variation is
    geometrically complex relative to the semantic concept space.

    The direct implementation would build both displacement matrices M and run
    SVD on them. That is memory-heavy when there are many languages/concepts.
    Instead, this code uses the fact that the singular values of the 
    centered matrix M_c are the square roots of the eigenvalues 
    of the Gram matrix of the center matrix (M_c.T @ M_c).
    It accumulates the centered Gram matrix:

        M_c.T @ M_c = M.T @ M - n * mean(M).T @ mean(M)

    and recovers singular values from:

        eigvals(M_c.T @ M_c) = singular_values(M_c)^2

    Expected input:
        self.X is a dict mapping language -> array of shape
        (num_concepts, embedding_dim).
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.effective_rank_method = kwargs.get("effective_rank_method", "stable")
        if self.effective_rank_method == "threshold":
            self.singular_value_threshold = kwargs.get("singular_value_threshold", 1e-12)

    def compute(self) -> float | tuple[float, dict[str, Any]]:
        X = stack_language_embeddings(
            self.X,
            self.num_concepts,
            self.num_languages,
            self.embedding_dim,
        )

        if self.normalize:
            X = self._normalize_embeddings(X)

        num_language_displacements = self.num_concepts * self._num_pairs(self.num_languages)
        num_concept_displacements = self.num_languages * self._num_pairs(self.num_concepts)

        d_lang = self._language_effective_rank(X)
        d_concept = self._concept_effective_rank(X)

        denom = d_lang + d_concept
        score = 0.0 if denom <= np.finfo(float).eps else d_lang / denom

        if self.return_details:
            normalized = self._normalized_comness_details(X, d_lang, d_concept)
            details: dict[str, Any] = {
                "d_lang": d_lang,
                "d_concept": d_concept,
                "num_language_displacements": num_language_displacements,
                "num_concept_displacements": num_concept_displacements,
                "num_languages": self.num_languages,
                "num_concepts": self.num_concepts,
                "embedding_dim": self.embedding_dim,
                "languages": list(self.X.keys()),
                "effective_rank_method": self.kwargs.get("effective_rank_method", "stable"),
                "normalize": self.normalize,
            }
            details.update(normalized)
            return score, details

        return score

    def _normalize_embeddings(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=-1, keepdims=True)
        return X / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)

    def _language_effective_rank(self, X: np.ndarray) -> float:
        """
        Same-concept cross-lingual displacements:

            x[c, l] - x[c, m], l != m

        Shape:
            (num_concepts * num_language_pairs, embedding_dim)

        By default, this uses unordered language pairs l < m because effective
        rank is unchanged by adding the negated copy of every vector.
        """
        if self.num_languages < 2:
            raise ValueError("Language displacements require at least two languages.")

        return pairwise_displacement_effective_rank(
            (X[:, c, :] for c in range(self.num_concepts)),
            embedding_dim=self.embedding_dim,
            **self._effective_rank_kwargs(),
        )

    def _concept_effective_rank(self, X: np.ndarray) -> float:
        """
        Same-language concept displacements:

            x[c, l] - x[c', l], c != c'

        Shape:
            (num_languages * num_concept_pairs, embedding_dim)

        By default, this uses unordered concept pairs c < c' because effective
        rank is unchanged by adding the negated copy of every vector.
        """
        if self.num_concepts < 2:
            raise ValueError("Concept displacements require at least two concepts.")

        return pairwise_displacement_effective_rank(
            (X[l] for l in range(self.num_languages)),
            embedding_dim=self.embedding_dim,
            **self._effective_rank_kwargs(),
        )

    def _normalized_comness_details(
        self,
        X: np.ndarray,
        d_lang: float,
        d_concept: float,
    ) -> dict[str, Any]:
        rng = self._random_baseline_rng()
        pool = X.reshape(-1, self.embedding_dim)
        lang_baseline = self._random_baseline(
            pool=pool,
            group_sizes=[self.num_languages] * self.num_concepts,
            rng=rng,
            normalize_by_dim=False,
        )
        concept_baseline = self._random_baseline(
            pool=pool,
            group_sizes=[self.num_concepts] * self.num_languages,
            rng=rng,
            normalize_by_dim=False,
        )
        if not lang_baseline or not concept_baseline:
            return {}

        d_lang_ratio = self._ratio(
            d_lang,
            lang_baseline["random_effective_dim_mean"],
        )
        d_concept_ratio = self._ratio(
            d_concept,
            concept_baseline["random_effective_dim_mean"],
        )
        normalized_comness = None
        if d_lang_ratio is not None and d_concept_ratio is not None:
            denom = d_lang_ratio + d_concept_ratio
            if denom > np.finfo(float).eps:
                normalized_comness = d_lang_ratio / denom

        return {
            "d_lang_random_effective_dim_mean": lang_baseline["random_effective_dim_mean"],
            "d_lang_random_effective_dim_std": lang_baseline["random_effective_dim_std"],
            "d_concept_random_effective_dim_mean": concept_baseline["random_effective_dim_mean"],
            "d_concept_random_effective_dim_std": concept_baseline["random_effective_dim_std"],
            "d_lang_ratio": d_lang_ratio,
            "d_concept_ratio": d_concept_ratio,
            "normalized_comness": normalized_comness,
            "random_baseline_trials": lang_baseline["random_baseline_trials"],
            "random_baseline_seed": self._random_baseline_seed(),
        }

    def _ratio(self, observed: float, random_mean: float) -> float | None:
        if random_mean <= np.finfo(float).eps:
            return None
        return observed / random_mean

    def _num_pairs(self, n: int) -> int:
        return n * (n - 1) // 2


class ConceptLanguagePrincipalAngleOverlap(COMMetric):
    """Principal-angle overlap between concept and language displacement subspaces."""

    def compute(self) -> dict[str, Any]:
        X = stack_language_embeddings(
            self.X,
            self.num_concepts,
            self.num_languages,
            self.embedding_dim,
        )
        if self.normalize:
            X = self._normalize_embeddings(X)

        energy_threshold = float(self.kwargs.get("subspace_energy_threshold", 0.9))
        if not 0 < energy_threshold <= 1:
            raise ValueError("subspace_energy_threshold must be in (0, 1].")

        concept_basis, concept_dim, concept_energy = pairwise_displacement_subspace_basis(
            (X[l] for l in range(self.num_languages)),
            embedding_dim=self.embedding_dim,
            energy_threshold=energy_threshold,
        )
        language_basis, language_dim, language_energy = pairwise_displacement_subspace_basis(
            (X[:, c, :] for c in range(self.num_concepts)),
            embedding_dim=self.embedding_dim,
            energy_threshold=energy_threshold,
        )

        if concept_dim == 0 or language_dim == 0:
            cosines = np.array([], dtype=float)
        else:
            cosines = np.linalg.svd(concept_basis.T @ language_basis, compute_uv=False)
            cosines = np.clip(cosines, 0.0, 1.0)

        squared_cosines = cosines ** 2
        mean_squared_cosine = (
            float(np.mean(squared_cosines)) if squared_cosines.size else 0.0
        )
        max_cosine = float(np.max(cosines)) if cosines.size else 0.0
        principal_angles_degrees = np.degrees(np.arccos(cosines))
        random_expected = self._random_expected_mean_squared_cosine(
            concept_dim,
            language_dim,
        )
        adjusted = self._adjusted_overlap(mean_squared_cosine, random_expected)

        result: dict[str, Any] = {
            "mean_squared_cosine": mean_squared_cosine,
            "random_expected_mean_squared_cosine": random_expected,
            "adjusted_mean_squared_cosine": adjusted,
            "max_cosine": max_cosine,
            "principal_angle_cosines": cosines.tolist(),
            "principal_angles_degrees": principal_angles_degrees.tolist(),
            "concept_subspace_dim": concept_dim,
            "language_subspace_dim": language_dim,
            "concept_energy_explained": concept_energy,
            "language_energy_explained": language_energy,
            "subspace_energy_threshold": energy_threshold,
        }
        if self.return_details:
            result["details"] = {
                "num_languages": self.num_languages,
                "num_concepts": self.num_concepts,
                "embedding_dim": self.embedding_dim,
                "languages": list(self.X.keys()),
                "normalize": self.normalize,
            }
        return result

    def _normalize_embeddings(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=-1, keepdims=True)
        return X / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)

    def _random_expected_mean_squared_cosine(
        self,
        concept_dim: int,
        language_dim: int,
    ) -> float:
        if concept_dim == 0 or language_dim == 0:
            return 0.0
        return min(max(concept_dim, language_dim) / self.embedding_dim, 1.0)

    def _adjusted_overlap(self, observed: float, expected: float) -> float | None:
        denom = 1.0 - expected
        if denom <= np.finfo(float).eps:
            return None
        return (observed - expected) / denom
