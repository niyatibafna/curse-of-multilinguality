from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


FormattedParallelText = dict[str, Any]


class ParallelDataset(ABC):
    """Base interface for datasets with sentence-aligned multilingual text."""

    def __init__(self, split: str = "dev"):
        self.split = split

    @abstractmethod
    def download(self) -> Any:
        """Download/load the raw dataset."""
        raise NotImplementedError

    @abstractmethod
    def multiparallel_format(self) -> list[FormattedParallelText]:
        """Return texts as {id, data: {lang: text}, metadata: {...}}."""
        raise NotImplementedError
