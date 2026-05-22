from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import COMMetric


class Comness(COMMetric):
    """
    Multilingual overhead / cross-lingual alignment diagnostic.

    Computes:

        COM(X) = d_lang / (d_lang + d_concept)

    where:
        d_lang    = effrank({x[c, l] - x[c, m]     : l != m})
        d_concept = effrank({x[c, l] - x[c', l]    : c != c'})

    Small scores mean language variation occupies few effective dimensions
    relative to concept variation. Large scores mean language variation is
    geometrically complex relative to the semantic concept space.

    Expected input:
        self.X is a dict mapping language -> array of shape
        (num_concepts, embedding_dim).
    """

    def compute(self) -> float | tuple[float, dict[str, Any]]:
        X = self._stack_by_language()

        if self.normalize:
            X = self._normalize_embeddings(X)

        lang_displacements = self._language_displacements(X)
        concept_displacements = self._concept_displacements(X)

        d_lang = self._effective_rank(lang_displacements)
        d_concept = self._effective_rank(concept_displacements)

        denom = d_lang + d_concept
        score = 0.0 if denom <= np.finfo(float).eps else d_lang / denom

        if self.return_details:
            details: dict[str, Any] = {
                "d_lang": d_lang,
                "d_concept": d_concept,
                "num_language_displacements": lang_displacements.shape[0],
                "num_concept_displacements": concept_displacements.shape[0],
                "num_languages": self.num_languages,
                "num_concepts": self.num_concepts,
                "embedding_dim": self.embedding_dim,
                "languages": list(self.X.keys()),
                "effective_rank_method": self.kwargs.get("effective_rank_method", "entropy"),
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

    def _language_displacements(self, X: np.ndarray) -> np.ndarray:
        """
        Same-concept cross-lingual displacements:

            x[c, l] - x[c, m], l != m

        Shape:
            (num_concepts * num_language_pairs, embedding_dim)

        By default, this uses unordered language pairs l < m because effective
        rank is unchanged by adding the negated copy of every vector. Set
        ordered_pairs=True in kwargs to include both l -> m and m -> l.
        """
        ordered_pairs = bool(self.kwargs.get("ordered_pairs", False))

        displacements: list[np.ndarray] = []

        for l in range(self.num_languages):
            if ordered_pairs:
                language_range = range(self.num_languages)
            else:
                language_range = range(l + 1, self.num_languages)

            for m in language_range:
                if l == m:
                    continue
                displacements.append(X[l] - X[m])

        if not displacements:
            raise ValueError("Language displacements require at least two languages.")

        return np.vstack(displacements)

    def _concept_displacements(self, X: np.ndarray) -> np.ndarray:
        """
        Same-language concept displacements:

            x[c, l] - x[c', l], c != c'

        Shape:
            (num_languages * num_concept_pairs, embedding_dim)

        By default, this uses unordered concept pairs c < c' because effective
        rank is unchanged by adding the negated copy of every vector. Set
        ordered_pairs=True in kwargs to include both c -> c' and c' -> c.
        """
        if self.num_concepts < 2:
            raise ValueError("Concept displacements require at least two concepts.")

        ordered_pairs = bool(self.kwargs.get("ordered_pairs", False))
        displacements: list[np.ndarray] = []

        for l in range(self.num_languages):
            for c in range(self.num_concepts):
                if ordered_pairs:
                    concept_range = range(self.num_concepts)
                else:
                    concept_range = range(c + 1, self.num_concepts)

                for cp in concept_range:
                    if c == cp:
                        continue
                    displacements.append(X[l, c] - X[l, cp])

        if not displacements:
            raise ValueError("Concept displacements require at least two concepts.")

        return np.vstack(displacements)

    def _effective_rank(self, M: np.ndarray) -> float:
        """
        Effective rank of a displacement matrix.

        Default method is entropy effective rank:

            effrank(M) = exp(H(p))
            p_i = s_i / sum_j s_j

        where s_i are the singular values of centered M.

        Optional kwargs:
            effective_rank_method:
                "entropy"  -> exp(entropy of normalized singular values)
                "stable"   -> (sum s_i)^2 / sum s_i^2
                "threshold" -> number of singular values above threshold

            center_displacements:
                Whether to mean-center displacement vectors before SVD.
                Default: True.

            singular_value_threshold:
                Threshold for method="threshold".
                Default: 1e-12.
        """
        M = np.asarray(M, dtype=float)

        if M.ndim != 2:
            raise ValueError("Expected displacement matrix to be 2-dimensional.")

        if M.shape[0] == 0:
            return 0.0

        center = bool(self.kwargs.get("center_displacements", True))
        if center:
            M = M - np.mean(M, axis=0, keepdims=True)

        singular_values = np.linalg.svd(M, full_matrices=False, compute_uv=False)
        singular_values = singular_values[
            singular_values > np.finfo(float).eps
        ]

        if singular_values.size == 0:
            return 0.0

        method = self.kwargs.get("effective_rank_method", "entropy")

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