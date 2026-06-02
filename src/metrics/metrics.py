from __future__ import annotations

from typing import Any

import numpy as np

from .utils import random_baseline_effective_rank


class COMMetric:
    def __init__(
        self,
        X: dict[str, np.ndarray],
        num_concepts: int,
        num_languages: int,
        embedding_dim: int,
        return_details: bool = False,
        normalize: bool = True,
        **kwargs: Any,
    ):
        """
        Base class for multilingual embedding diagnostics.

        Parameters
        ----------
        X : dict[str, np.ndarray]
            Multilingual embedding tensor with shape:
                {
                    "lang": np.ndarray,
                    ...
                }
            where X[lang] is the embeddings matrix for language lang. 
            X[lang] has shape (num_concepts, embedding_dim).

        num_concepts : int
            Number of concepts.

        num_languages : int
            Number of languages.

        embedding_dim : int
            Dimension of the embeddings.

        normalize : bool, default=True
            Whether to normalize the embeddings.

        return_details : bool, default=False
            Whether to return diagnostic information in addition to the main score.

        **kwargs : Any
            Extra options. The base class stores them in `self.kwargs` but does not interpret them.
        """
        self.X = X
        self.return_details = return_details
        self.normalize = normalize
        self.num_concepts = num_concepts
        self.num_languages = num_languages
        self.embedding_dim = embedding_dim
        self.kwargs = kwargs

    def compute(self):
        pass

    def _effective_rank_kwargs(self) -> dict[str, Any]:
        method = self.kwargs.get("effective_rank_method", "stable")
        kwargs: dict[str, Any] = {"method": method}
        if method == "threshold":
            kwargs["singular_value_threshold"] = float(
                self.kwargs.get("singular_value_threshold", 1e-12)
            )
        return kwargs

    def _random_baseline(
        self,
        pool: np.ndarray,
        group_sizes: list[int],
        rng: np.random.Generator,
        normalize_by_dim: bool,
    ) -> dict[str, Any]:
        return random_baseline_effective_rank(
            pool,
            group_sizes,
            embedding_dim=self.embedding_dim,
            trials=self._random_baseline_trials(),
            rng=rng,
            normalize_by_dim=normalize_by_dim,
            **self._effective_rank_kwargs(),
        )

    def _random_baseline_trials(self) -> int:
        trials = int(self.kwargs.get("random_baseline_trials", 1))
        if trials < 0:
            raise ValueError("random_baseline_trials must be non-negative.")
        return trials

    def _random_baseline_seed(self) -> int:
        return int(self.kwargs.get("random_baseline_seed", 0))

    def _random_baseline_rng(self) -> np.random.Generator:
        return np.random.default_rng(self._random_baseline_seed())
