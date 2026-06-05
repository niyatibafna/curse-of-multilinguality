from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import COMMetric


class AlignmentCondition(COMMetric):
    """
    Fraction of ordered cross-language equivalents closer than any different concept.

    For each concept c and ordered language pair (l1, l2), this checks whether
    x[c, l2] is strictly more similar to x[c, l1] than a set of different-concept
    negatives. The default strong view compares against every x[c', l] with
    c' != c; the weak view compares only against x[c', l2].
    """

    def compute(self) -> dict[str, Any]:
        if self.num_languages < 2:
            raise ValueError("AlignmentCondition requires at least two languages.")
        if self.num_concepts < 2:
            raise ValueError("AlignmentCondition requires at least two concepts.")

        similarity = self.kwargs.get("similarity", "cosine")
        negative_view = self.kwargs.get("negative_view", "strong_view")
        if negative_view not in {"strong_view", "weak_view"}:
            raise ValueError("negative_view must be 'strong_view' or 'weak_view'.")
        batch_size = int(self.kwargs.get("alignment_batch_size", 64))
        if batch_size <= 0:
            raise ValueError("alignment_batch_size must be positive.")

        languages = list(self.X.keys())
        arrays = self._validated_arrays(languages)
        pair_success = np.zeros((self.num_languages, self.num_languages), dtype=np.int64)

        for source_index, source_language in enumerate(languages):
            source = self._prepare_embeddings(arrays[source_index], similarity)
            if negative_view == "strong_view":
                negative_max = np.full(self.num_concepts, -np.inf, dtype=float)
                target_negative_max = None
            else:
                negative_max = None
                target_negative_max = np.full(
                    (self.num_concepts, self.num_languages),
                    -np.inf,
                    dtype=float,
                )
            positives = np.empty((self.num_concepts, self.num_languages), dtype=float)

            for candidate_index, candidate_array in enumerate(arrays):
                candidate = (
                    source
                    if candidate_index == source_index
                    else self._prepare_embeddings(candidate_array, similarity)
                )
                for start in range(0, self.num_concepts, batch_size):
                    end = min(start + batch_size, self.num_concepts)
                    concept_indices = np.arange(start, end)
                    source_batch = source[start:end]

                    sims = self._similarity_matrix(source_batch, candidate, similarity)
                    positives[start:end, candidate_index] = sims[
                        np.arange(end - start),
                        concept_indices,
                    ]
                    sims[np.arange(end - start), concept_indices] = -np.inf
                    candidate_negative_max = np.max(sims, axis=1)
                    if negative_view == "strong_view":
                        assert negative_max is not None
                        negative_max[start:end] = np.maximum(
                            negative_max[start:end],
                            candidate_negative_max,
                        )
                    else:
                        assert target_negative_max is not None
                        target_negative_max[start:end, candidate_index] = candidate_negative_max

            if negative_view == "strong_view":
                assert negative_max is not None
                successes = positives > negative_max[:, None]
            else:
                assert target_negative_max is not None
                successes = positives > target_negative_max
            successes[:, source_index] = False
            pair_success[source_index] += np.sum(successes, axis=0)

        pair_rows = []
        total_success = 0
        total_pairs = self.num_concepts * self.num_languages * (self.num_languages - 1)
        for source_index, source_language in enumerate(languages):
            for target_index, target_language in enumerate(languages):
                if source_index == target_index:
                    continue
                num_success = int(pair_success[source_index, target_index])
                total_success += num_success
                pair_rows.append({
                    "source_language": source_language,
                    "target_language": target_language,
                    "num_success": num_success,
                    "num_pairs": self.num_concepts,
                    "score": num_success / self.num_concepts,
                })

        return {
            "score": total_success / total_pairs,
            "num_success": int(total_success),
            "num_pairs": int(total_pairs),
            "similarity": similarity,
            "negative_view": negative_view,
            "strict": True,
            "batch_size": batch_size,
            "num_languages": self.num_languages,
            "num_concepts": self.num_concepts,
            "languages": languages,
            "language_pairs": pair_rows,
        }

    def _validated_arrays(self, languages: list[str]) -> list[np.ndarray]:
        expected_shape = (self.num_concepts, self.embedding_dim)
        arrays = []
        for language in languages:
            arr = np.asarray(self.X[language])
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Expected X[{language!r}] to have shape {expected_shape}, got {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"X[{language!r}] contains NaN or infinite values.")
            arrays.append(arr)
        return arrays

    def _prepare_embeddings(self, embeddings: np.ndarray, similarity: str) -> np.ndarray:
        if similarity == "cosine":
            return self._normalize_rows(embeddings)
        if similarity in {"dot", "negative_squared_l2"}:
            return embeddings
        raise ValueError(
            "Unknown similarity. Expected one of: 'cosine', 'dot', or 'negative_squared_l2'."
        )

    def _similarity_matrix(
        self,
        source: np.ndarray,
        candidates: np.ndarray,
        similarity: str,
    ) -> np.ndarray:
        if similarity in {"cosine", "dot"}:
            return source @ candidates.T
        if similarity == "negative_squared_l2":
            source_norm = np.sum(source * source, axis=1, keepdims=True)
            candidate_norm = np.sum(candidates * candidates, axis=1, keepdims=True).T
            return -(source_norm + candidate_norm - 2.0 * (source @ candidates.T))
        raise ValueError(
            "Unknown similarity. Expected one of: 'cosine', 'dot', or 'negative_squared_l2'."
        )

    def _normalize_rows(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)


class MonolingualStructureCondition(COMMetric):
    """
    Correlation between within-language concept-pair distances across languages.
    """

    def compute(self) -> dict[str, Any]:
        if self.num_languages < 2:
            raise ValueError("MonolingualStructureCondition requires at least two languages.")
        if self.num_concepts < 3:
            raise ValueError("MonolingualStructureCondition requires at least three concepts.")

        languages = list(self.X.keys())
        arrays = self._validated_arrays(languages)
        distance_vectors = [
            self._cosine_distance_pairs(array)
            for array in arrays
        ]

        pair_rows = []
        correlations = []
        num_concept_pairs = len(distance_vectors[0])
        for i, language_1 in enumerate(languages):
            for j in range(i + 1, self.num_languages):
                language_2 = languages[j]
                correlation = self._pearson_correlation(
                    distance_vectors[i],
                    distance_vectors[j],
                )
                if correlation is not None:
                    correlations.append(correlation)
                pair_rows.append({
                    "language_1": language_1,
                    "language_2": language_2,
                    "correlation": correlation,
                    "num_concept_pairs": num_concept_pairs,
                })

        score = float(np.mean(correlations)) if correlations else None
        return {
            "score": score,
            "distance": "cosine",
            "correlation": "pearson",
            "num_languages": self.num_languages,
            "num_concepts": self.num_concepts,
            "num_concept_pairs": num_concept_pairs,
            "num_valid_language_pairs": len(correlations),
            "languages": languages,
            "language_pairs": pair_rows,
        }

    def _validated_arrays(self, languages: list[str]) -> list[np.ndarray]:
        expected_shape = (self.num_concepts, self.embedding_dim)
        arrays = []
        for language in languages:
            arr = np.asarray(self.X[language])
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Expected X[{language!r}] to have shape {expected_shape}, got {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"X[{language!r}] contains NaN or infinite values.")
            arrays.append(arr)
        return arrays

    def _cosine_distance_pairs(self, embeddings: np.ndarray) -> np.ndarray:
        normalized = self._normalize_rows(embeddings)
        similarity = normalized @ normalized.T
        rows, cols = np.triu_indices(self.num_concepts, k=1)
        return 1.0 - similarity[rows, cols]

    def _normalize_rows(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)

    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float | None:
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
        if denom == 0.0:
            return None
        return float(np.dot(x_centered, y_centered) / denom)
