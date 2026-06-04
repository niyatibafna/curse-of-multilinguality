from __future__ import annotations

from typing import Any, Iterable, Literal

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


def pairwise_displacement_subspace_basis(
    groups: Iterable[np.ndarray],
    embedding_dim: int,
    energy_threshold: float = 0.9,
) -> tuple[np.ndarray, int, float]:
    if not 0 < energy_threshold <= 1:
        raise ValueError("energy_threshold must be in (0, 1].")

    moments = PairwiseDisplacementMoments(embedding_dim)
    for group in groups:
        moments.update(group)

    if moments.count == 0:
        return np.zeros((embedding_dim, 0)), 0, 0.0

    covariance = centered_gram_from_moments(moments)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
    eigenvectors = eigenvectors[:, order]

    positive = eigenvalues > np.finfo(float).eps
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    if eigenvalues.size == 0:
        return np.zeros((embedding_dim, 0)), 0, 0.0

    cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    dim = min(
        int(np.searchsorted(cumulative_energy, energy_threshold) + 1),
        eigenvalues.size,
    )
    return eigenvectors[:, :dim], dim, float(cumulative_energy[dim - 1])


def stack_language_embeddings(
    X: dict[str, np.ndarray],
    num_concepts: int,
    num_languages: int,
    embedding_dim: int,
) -> np.ndarray:
    if len(X) != num_languages:
        raise ValueError(f"Expected {num_languages} languages, got {len(X)}.")

    arrays = []
    expected_shape = (num_concepts, embedding_dim)
    for language, embeddings in X.items():
        arr = np.asarray(embeddings, dtype=float)
        if arr.shape != expected_shape:
            raise ValueError(
                f"Expected X[{language!r}] to have shape {expected_shape}, got {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"X[{language!r}] contains NaN or infinite values.")
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def random_baseline_effective_rank(
    pool: np.ndarray,
    group_sizes: Iterable[int],
    embedding_dim: int,
    trials: int,
    rng: np.random.Generator,
    method: EffectiveRankMethod | str = "stable",
    singular_value_threshold: float = 1e-12,
    normalize_by_dim: bool = False,
) -> dict[str, Any]:
    group_sizes = list(group_sizes)
    if trials <= 0:
        return {}

    ranks = [
        pairwise_displacement_effective_rank(
            random_groups_like(pool, group_sizes, rng),
            embedding_dim=embedding_dim,
            method=method,
            singular_value_threshold=singular_value_threshold,
            normalize_by_dim=normalize_by_dim,
        )
        for _ in range(trials)
    ]

    return {
        "random_effective_dim_mean": float(np.mean(ranks)),
        "random_effective_dim_std": float(np.std(ranks)),
        "random_baseline_trials": trials,
    }


def random_groups_like(
    pool: np.ndarray,
    group_sizes: Iterable[int],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    pool = np.asarray(pool, dtype=float)
    if pool.ndim != 2:
        raise ValueError("Expected pool to have shape (num_points, embedding_dim).")
    if not np.all(np.isfinite(pool)):
        raise ValueError("Pool contains NaN or infinite values.")

    group_sizes = list(group_sizes)
    if any(size < 0 for size in group_sizes):
        raise ValueError("Group sizes must be non-negative.")

    total = sum(group_sizes)
    replace = total > pool.shape[0]
    indices = rng.choice(pool.shape[0], size=total, replace=replace)

    groups = []
    start = 0
    for size in group_sizes:
        groups.append(pool[indices[start:start + size]])
        start += size
    return groups


def add_effective_dim_baseline(
    row: dict[str, Any],
    observed_effective_dim: float,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    if not baseline:
        return row

    random_mean = baseline["random_effective_dim_mean"]
    ratio = None
    if random_mean > np.finfo(float).eps:
        ratio = observed_effective_dim / random_mean

    return {
        **row,
        **baseline,
        "effective_dim_ratio": ratio,
    }


def effective_rank_from_moments(
    moments: "PairwiseDisplacementMoments",
    method: EffectiveRankMethod | str = "stable",
    singular_value_threshold: float = 1e-12,
    normalize_by_dim: bool = False,
) -> float:
    if moments.count == 0:
        return 0.0

    eigenvalues = np.linalg.eigvalsh(centered_gram_from_moments(moments))
    singular_values = np.sqrt(np.clip(eigenvalues, a_min=0.0, a_max=None))
    return effective_rank_from_singular_values(
        singular_values,
        method=method,
        singular_value_threshold=singular_value_threshold,
        normalize_by_dim=normalize_by_dim,
        embedding_dim=moments.embedding_dim,
    )


def centered_gram_from_moments(moments: "PairwiseDisplacementMoments") -> np.ndarray:
    if moments.count == 0:
        return np.zeros((moments.embedding_dim, moments.embedding_dim), dtype=float)

    mean = moments.total / moments.count
    covariance = moments.gram - moments.count * np.outer(mean, mean)
    return (covariance + covariance.T) / 2


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
