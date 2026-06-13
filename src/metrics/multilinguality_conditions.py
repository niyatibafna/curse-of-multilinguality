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
        distance_stats = [
            self._distance_vector_stats(vector)
            for vector in distance_vectors
        ]

        pair_rows = []
        measure_values: dict[str, list[float]] = {
            "pearson": [],
            "spearman": [],
            "mae": [],
            "rmse": [],
            "normalized_rmse": [],
            "centered_rmse": [],
            "standardized_rmse": [],
            "mean_distance_ratio": [],
            "std_distance_ratio": [],
        }
        num_concept_pairs = len(distance_vectors[0])
        for i, language_1 in enumerate(languages):
            for j in range(i + 1, self.num_languages):
                language_2 = languages[j]
                measures = self._distance_vector_measures(distance_stats[i], distance_stats[j])
                for name, value in measures.items():
                    if value is not None and name in measure_values:
                        measure_values[name].append(value)
                pair_rows.append({
                    "language_1": language_1,
                    "language_2": language_2,
                    "num_concept_pairs": num_concept_pairs,
                    **measures,
                })

        mean_measures = {
            f"mean_{name}": float(np.mean(values)) if values else None
            for name, values in measure_values.items()
        }
        return {
            "score": mean_measures["mean_pearson"],
            "distance": "cosine",
            "correlation": "pearson",
            "num_languages": self.num_languages,
            "num_concepts": self.num_concepts,
            "num_concept_pairs": num_concept_pairs,
            "num_valid_language_pairs": len(measure_values["pearson"]),
            **mean_measures,
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

    def _distance_vector_stats(self, distances: np.ndarray) -> dict[str, Any]:
        mean = float(np.mean(distances))
        std = float(np.std(distances))
        centered = distances - mean
        return {
            "distances": distances,
            "mean": mean,
            "std": std,
            "centered": centered,
            "standardized": self._standardize(distances, mean, std),
            "ranks": self._rankdata(distances),
        }

    def _distance_vector_measures(
        self,
        x_stats: dict[str, Any],
        y_stats: dict[str, Any],
    ) -> dict[str, float | None]:
        x = x_stats["distances"]
        y = y_stats["distances"]
        x_mean = x_stats["mean"]
        y_mean = y_stats["mean"]
        x_std = x_stats["std"]
        y_std = y_stats["std"]
        diff = x - y

        return {
            "correlation": self._centered_cosine(x_stats["centered"], y_stats["centered"]),
            "pearson": self._centered_cosine(x_stats["centered"], y_stats["centered"]),
            "spearman": self._pearson_correlation(x_stats["ranks"], y_stats["ranks"]),
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "normalized_rmse": self._safe_ratio(
                float(np.sqrt(np.mean(diff ** 2))),
                0.5 * (x_mean + y_mean),
            ),
            "centered_rmse": float(
                np.sqrt(np.mean((x_stats["centered"] - y_stats["centered"]) ** 2))
            ),
            "standardized_rmse": self._rmse_or_none(
                x_stats["standardized"],
                y_stats["standardized"],
            ),
            "mean_distance_1": x_mean,
            "mean_distance_2": y_mean,
            "std_distance_1": x_std,
            "std_distance_2": y_std,
            "mean_distance_ratio": self._safe_ratio(y_mean, x_mean),
            "std_distance_ratio": self._safe_ratio(y_std, x_std),
        }

    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float | None:
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        return self._centered_cosine(x_centered, y_centered)

    def _centered_cosine(self, x_centered: np.ndarray, y_centered: np.ndarray) -> float | None:
        denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
        if denom == 0.0:
            return None
        return float(np.dot(x_centered, y_centered) / denom)

    def _spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> float | None:
        return self._pearson_correlation(self._rankdata(x), self._rankdata(y))

    def _rankdata(self, values: np.ndarray) -> np.ndarray:
        values = np.round(values, decimals=12)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return ranks

    def _standardize(self, values: np.ndarray, mean: float, std: float) -> np.ndarray | None:
        if std <= np.finfo(float).eps:
            return None
        return (values - mean) / std

    def _rmse_or_none(self, x: np.ndarray | None, y: np.ndarray | None) -> float | None:
        if x is None or y is None:
            return None
        return float(np.sqrt(np.mean((x - y) ** 2)))

    def _safe_ratio(self, numerator: float, denominator: float) -> float | None:
        if abs(denominator) <= np.finfo(float).eps:
            return None
        return numerator / denominator


class RmseAgainstMonolingual(MonolingualStructureCondition):
    """
    RMSE between each language's concept-pair distances and an external monolingual model.
    """

    def compute(self) -> dict[str, Any]:
        if self.num_languages < 1:
            raise ValueError("RmseAgainstMonolingual requires at least one language.")
        if self.num_concepts < 3:
            raise ValueError("RmseAgainstMonolingual requires at least three concepts.")

        reference_embeddings = self.kwargs.get("reference_embeddings")
        if reference_embeddings is None:
            raise ValueError("RmseAgainstMonolingual requires reference_embeddings.")
        reference = np.asarray(reference_embeddings)
        if reference.shape[0] != self.num_concepts:
            raise ValueError(
                "Expected reference_embeddings to have "
                f"{self.num_concepts} concepts, got {reference.shape[0]}."
            )
        if reference.ndim != 2:
            raise ValueError("Expected reference_embeddings to be a 2D array.")
        if not np.all(np.isfinite(reference)):
            raise ValueError("reference_embeddings contains NaN or infinite values.")

        languages = list(self.X.keys())
        arrays = self._validated_arrays(languages)
        reference_stats = self._distance_vector_stats(self._cosine_distance_pairs(reference))

        language_rows = []
        measure_values: dict[str, list[float]] = {
            "pearson": [],
            "spearman": [],
            "mae": [],
            "rmse": [],
            "normalized_rmse": [],
            "centered_rmse": [],
            "standardized_rmse": [],
            "mean_distance_ratio": [],
            "std_distance_ratio": [],
        }
        num_concept_pairs = len(reference_stats["distances"])
        for language, array in zip(languages, arrays):
            measures = self._distance_vector_measures(
                self._distance_vector_stats(self._cosine_distance_pairs(array)),
                reference_stats,
            )
            for name, value in measures.items():
                if value is not None and name in measure_values:
                    measure_values[name].append(value)
            language_rows.append({
                "language": language,
                "reference_language": self.kwargs.get("reference_language"),
                "num_concept_pairs": num_concept_pairs,
                **measures,
            })

        mean_measures = {
            f"mean_{name}": float(np.mean(values)) if values else None
            for name, values in measure_values.items()
        }
        return {
            "score": mean_measures["mean_rmse"],
            "distance": "cosine",
            "reference_language": self.kwargs.get("reference_language"),
            "reference_model": self.kwargs.get("reference_model"),
            "num_languages": self.num_languages,
            "num_concepts": self.num_concepts,
            "num_concept_pairs": num_concept_pairs,
            "num_valid_languages": len(measure_values["rmse"]),
            **mean_measures,
            "languages": languages,
            "language_comparisons": language_rows,
        }


class NearestNeighborOverlapAgainstMonolingual(MonolingualStructureCondition):
    """
    Mean k-nearest-neighbor concept overlap against an external monolingual model.
    """

    def compute(self) -> dict[str, Any]:
        if self.num_languages < 1:
            raise ValueError(
                "NearestNeighborOverlapAgainstMonolingual requires at least one language."
            )
        k = int(self.kwargs.get("nearest_neighbor_k", 10))
        if k <= 0:
            raise ValueError("nearest_neighbor_k must be positive.")
        if self.num_concepts <= k:
            raise ValueError(
                "NearestNeighborOverlapAgainstMonolingual requires more concepts than k."
            )

        reference_embeddings = self.kwargs.get("reference_embeddings")
        if reference_embeddings is None:
            raise ValueError(
                "NearestNeighborOverlapAgainstMonolingual requires reference_embeddings."
            )
        reference = np.asarray(reference_embeddings)
        if reference.shape[0] != self.num_concepts:
            raise ValueError(
                "Expected reference_embeddings to have "
                f"{self.num_concepts} concepts, got {reference.shape[0]}."
            )
        if reference.ndim != 2:
            raise ValueError("Expected reference_embeddings to be a 2D array.")
        if not np.all(np.isfinite(reference)):
            raise ValueError("reference_embeddings contains NaN or infinite values.")

        languages = list(self.X.keys())
        arrays = self._validated_arrays(languages)
        reference_neighbors = self._nearest_neighbor_indices(reference, k)

        language_rows = []
        language_overlaps = []
        for language, array in zip(languages, arrays):
            target_neighbors = self._nearest_neighbor_indices(array, k)
            concept_overlaps = self._concept_overlaps(reference_neighbors, target_neighbors, k)
            mean_overlap = float(np.mean(concept_overlaps))
            language_overlaps.append(mean_overlap)
            row = {
                "language": language,
                "reference_language": self.kwargs.get("reference_language"),
                "k": k,
                "num_concepts": self.num_concepts,
                "mean_overlap": mean_overlap,
                "std_overlap": float(np.std(concept_overlaps)),
            }
            if self.return_details:
                row["concept_overlaps"] = concept_overlaps
            language_rows.append(row)

        mean_overlap = float(np.mean(language_overlaps)) if language_overlaps else None
        return {
            "score": mean_overlap,
            "mean_overlap": mean_overlap,
            "k": k,
            "similarity": "cosine",
            "reference_language": self.kwargs.get("reference_language"),
            "reference_model": self.kwargs.get("reference_model"),
            "num_languages": self.num_languages,
            "num_concepts": self.num_concepts,
            "languages": languages,
            "language_comparisons": language_rows,
        }

    def _nearest_neighbor_indices(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        normalized = self._normalize_rows(embeddings)
        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, -np.inf)
        return np.argsort(-similarity, axis=1, kind="mergesort")[:, :k]

    def _concept_overlaps(
        self,
        reference_neighbors: np.ndarray,
        target_neighbors: np.ndarray,
        k: int,
    ) -> list[float]:
        overlaps = []
        for reference_row, target_row in zip(reference_neighbors, target_neighbors):
            overlaps.append(len(set(reference_row).intersection(target_row)) / k)
        return overlaps
