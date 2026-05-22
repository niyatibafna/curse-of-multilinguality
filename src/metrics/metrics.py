from __future__ import annotations

from typing import Any

import numpy as np


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