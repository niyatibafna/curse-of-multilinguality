from __future__ import annotations

from typing import Iterable, Literal

import numpy as np


EffectiveRankMethod = Literal["entropy", "stable", "threshold"]


def effective_rank(
    matrix: np.ndarray,
    method: EffectiveRankMethod | str = "stable",
    center: bool = True,
    singular_value_threshold: float = 1e-12,
    normalize_by_dim: bool = False,
) -> float:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected matrix to have shape (num_items, embedding_dim).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Matrix contains NaN or infinite values.")
    if matrix.shape[0] == 0:
        return 0.0

    if center:
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True)

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return effective_rank_from_singular_values(
        singular_values,
        method=method,
        singular_value_threshold=singular_value_threshold,
        normalize_by_dim=normalize_by_dim,
        embedding_dim=matrix.shape[1],
    )


def effective_rank_from_singular_values(
    singular_values: np.ndarray,
    method: EffectiveRankMethod | str = "stable",
    singular_value_threshold: float = 1e-12,
    normalize_by_dim: bool = False,
    embedding_dim: int | None = None,
) -> float:
    singular_values = np.asarray(singular_values, dtype=float)
    if singular_values.ndim != 1:
        raise ValueError("Expected singular_values to be one-dimensional.")
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("Singular values contain NaN or infinite values.")

    singular_values = singular_values[singular_values > np.finfo(float).eps]
    if singular_values.size == 0:
        return 0.0

    if method == "entropy":
        probs = singular_values / np.sum(singular_values)
        entropy = -float(np.sum(probs * np.log(probs)))
        return _normalize_effective_rank(float(np.exp(entropy)), normalize_by_dim, embedding_dim)

    if method == "stable":
        numerator = float(np.sum(singular_values) ** 2)
        denominator = float(np.sum(singular_values ** 2))
        rank = 0.0 if denominator <= np.finfo(float).eps else numerator / denominator
        return _normalize_effective_rank(rank, normalize_by_dim, embedding_dim)

    if method == "threshold":
        rank = float(np.sum(singular_values > singular_value_threshold))
        return _normalize_effective_rank(rank, normalize_by_dim, embedding_dim)

    raise ValueError(
        "Unknown effective rank method. Expected one of: "
        "'entropy', 'stable', or 'threshold'."
    )


def pairwise_displacement_effective_rank(
    groups: Iterable[np.ndarray],
    embedding_dim: int,
    method: EffectiveRankMethod | str = "stable",
    singular_value_threshold: float = 1e-12,
    normalize_by_dim: bool = False,
) -> float:
    moments = PairwiseDisplacementMoments(embedding_dim)
    for group in groups:
        moments.update(group)
    return effective_rank_from_moments(
        moments,
        method=method,
        singular_value_threshold=singular_value_threshold,
        normalize_by_dim=normalize_by_dim,
    )


def effective_rank_from_moments(
    moments: "PairwiseDisplacementMoments",
    method: EffectiveRankMethod | str = "stable",
    singular_value_threshold: float = 1e-12,
    normalize_by_dim: bool = False,
) -> float:
    if moments.count == 0:
        return 0.0

    mean = moments.total / moments.count
    covariance = moments.gram - moments.count * np.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2

    eigenvalues = np.linalg.eigvalsh(covariance)
    singular_values = np.sqrt(np.clip(eigenvalues, a_min=0.0, a_max=None))
    return effective_rank_from_singular_values(
        singular_values,
        method=method,
        singular_value_threshold=singular_value_threshold,
        normalize_by_dim=normalize_by_dim,
        embedding_dim=moments.embedding_dim,
    )


class PairwiseDisplacementMoments:
    """
    Accumulates moments of pairwise difference rows without materializing them.

    For each group G = [x_0, ..., x_{n-1}], this represents all rows
    (x_i - x_j), i < j.
    """
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.count = 0
        self.total = np.zeros(embedding_dim, dtype=float)
        self.gram = np.zeros((embedding_dim, embedding_dim), dtype=float)

    def update(self, group: np.ndarray) -> None:
        group = np.asarray(group, dtype=float)
        if group.ndim != 2:
            raise ValueError("Expected group to have shape (num_items, embedding_dim).")
        if group.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected group embedding_dim={self.embedding_dim}, got {group.shape[1]}."
            )
        if not np.all(np.isfinite(group)):
            raise ValueError("Group contains NaN or infinite values.")

        n = group.shape[0]
        if n < 2:
            return

        group_sum = np.sum(group, axis=0)
        self.count += n * (n - 1) // 2
        self.total += pair_sum_weights(n) @ group

        # sum_{i < j} (x_i - x_j)(x_i - x_j).T
        #   = n * sum_i x_i x_i.T - (sum_i x_i)(sum_i x_i).T
        self.gram += n * (group.T @ group) - np.outer(group_sum, group_sum)


def pair_sum_weights(n: int) -> np.ndarray:
    return np.arange(n - 1, -n, -2, dtype=float)


def _normalize_effective_rank(
    rank: float,
    normalize_by_dim: bool,
    embedding_dim: int | None,
) -> float:
    if not normalize_by_dim:
        return rank
    if embedding_dim is None:
        raise ValueError("embedding_dim is required when normalize_by_dim=True.")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive when normalize_by_dim=True.")
    return rank / embedding_dim
