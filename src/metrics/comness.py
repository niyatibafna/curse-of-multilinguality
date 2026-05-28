from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import COMMetric
from .utils import pairwise_displacement_effective_rank


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
        print(f"Effective rank method: {self.effective_rank_method}")
        if self.effective_rank_method == "threshold":
            self.singular_value_threshold = kwargs.get("singular_value_threshold", 1e-12)
            print(f"Singular value threshold: {self.singular_value_threshold}")

    def compute(self) -> float | tuple[float, dict[str, Any]]:
        X = self._stack_by_language()

        if self.normalize:
            X = self._normalize_embeddings(X)

        num_language_displacements = self.num_concepts * self._num_pairs(self.num_languages)
        num_concept_displacements = self.num_languages * self._num_pairs(self.num_concepts)

        d_lang = self._language_effective_rank(X)
        d_concept = self._concept_effective_rank(X)

        denom = d_lang + d_concept
        score = 0.0 if denom <= np.finfo(float).eps else d_lang / denom

        if self.return_details:
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
            return score, details

        return score

    def _stack_by_language(self) -> np.ndarray:
        """
        Returns an array with shape:
            (num_languages, num_concepts, embedding_dim)
        """
        if len(self.X) != self.num_languages:
            raise ValueError(
                f"Expected {self.num_languages} languages, got {len(self.X)}."
            )

        arrays: list[np.ndarray] = []

        for lang, embeddings in self.X.items():
            arr = np.asarray(embeddings, dtype=float)

            expected_shape = (self.num_concepts, self.embedding_dim)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Expected X[{lang!r}] to have shape {expected_shape}, "
                    f"got {arr.shape}."
                )

            if not np.all(np.isfinite(arr)):
                raise ValueError(f"X[{lang!r}] contains NaN or infinite values.")

            arrays.append(arr)

        return np.stack(arrays, axis=0)

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

    def _num_pairs(self, n: int) -> int:
        return n * (n - 1) // 2
