from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import COMMetric


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

        # For each concept, accumulate all language-pair displacement moments.
        accumulator = _DisplacementMoments(self.embedding_dim)
        weights = self._pair_sum_weights(self.num_languages)

        for c in range(self.num_concepts):
            group = X[:, c, :]
            accumulator.update_pairwise(group, weights)

        return self._effective_rank_from_moments(accumulator)

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

        # For each language, accumulate all concept-pair displacement moments.
        accumulator = _DisplacementMoments(self.embedding_dim)
        weights = self._pair_sum_weights(self.num_concepts)

        for l in range(self.num_languages):
            accumulator.update_pairwise(X[l], weights)

        return self._effective_rank_from_moments(accumulator)

    def _num_pairs(self, n: int) -> int:
        return n * (n - 1) // 2

    def _pair_sum_weights(self, n: int) -> np.ndarray:
        # Coefficients for sum_{i < j} (x_i - x_j).
        return np.arange(n - 1, -n, -2, dtype=float)

    def _effective_rank_from_moments(self, moments: "_DisplacementMoments") -> float:
        """
        Effective rank of a centered displacement matrix from summary moments.

        Default method is entropy effective rank:

            effrank(M) = exp(H(p))
            p_i = s_i / sum_j s_j

        where s_i are the singular values of centered M.

        Optional kwargs:
            effective_rank_method:
                "entropy"  -> exp(entropy of normalized singular values)
                "stable"   -> (sum s_i)^2 / sum s_i^2
                "threshold" -> number of singular values above threshold

            singular_value_threshold:
                Threshold for method="threshold".
                Default: 1e-12.
        """
        if moments.count == 0:
            return 0.0

        # Centering is applied in Gram space without materializing M_c.
        mean = moments.total / moments.count
        covariance = moments.gram - moments.count * np.outer(mean, mean)
        covariance = (covariance + covariance.T) / 2

        # Singular values of M_c are sqrt(eigenvalues(M_c.T @ M_c)).
        eigenvalues = np.linalg.eigvalsh(covariance)
        singular_values = np.sqrt(np.clip(eigenvalues, a_min=0.0, a_max=None))
        singular_values = singular_values[
            singular_values > np.finfo(float).eps
        ]

        if singular_values.size == 0:
            return 0.0

        method = self.kwargs.get("effective_rank_method", "stable")

        if method == "entropy":
            probs = singular_values / np.sum(singular_values)
            entropy = -float(np.sum(probs * np.log(probs)))
            return float(np.exp(entropy))

        if method == "stable":
            numerator = float(np.sum(singular_values) ** 2)
            denominator = float(np.sum(singular_values ** 2))
            return 0.0 if denominator <= np.finfo(float).eps else numerator / denominator

        if method == "threshold":
            threshold = float(self.kwargs.get("singular_value_threshold", 1e-12))
            return float(np.sum(singular_values > threshold))

        raise ValueError(
            "Unknown effective_rank_method. Expected one of: "
            "'entropy', 'stable', or 'threshold'."
        )


class _DisplacementMoments:
    """
    Stores enough statistics to compute effrank of pairwise differences.

    For a group G = [x_0, ..., x_{n-1}], this accumulates the moments of all
    rows (x_i - x_j), i < j, without building those rows.
    """
    def __init__(self, embedding_dim: int) -> None:
        self.count = 0
        self.total = np.zeros(embedding_dim, dtype=float)
        self.gram = np.zeros((embedding_dim, embedding_dim), dtype=float)

    def update_pairwise(self, group: np.ndarray, weights: np.ndarray) -> None:
        n = group.shape[0]
        if n < 2:
            return

        group_sum = np.sum(group, axis=0)
        self.count += n * (n - 1) // 2
        self.total += weights @ group

        # sum_{i < j} (x_i - x_j)(x_i - x_j).T
        #   = n * sum_i x_i x_i.T - (sum_i x_i)(sum_i x_i).T
        self.gram += n * (group.T @ group) - np.outer(group_sum, group_sum)
