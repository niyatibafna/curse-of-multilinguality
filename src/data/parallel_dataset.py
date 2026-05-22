from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


FormattedParallelText = dict[str, Any]


class ParallelDataset(ABC):
    """Base interface for datasets with sentence-aligned multilingual text."""

    def __init__(self, split: str = "dev", cache_dir: str | Path | None = None):
        self.split = split
        self.cache_dir = str(cache_dir) if cache_dir is not None else None

    @abstractmethod
    def download(self) -> Any:
        """Download/load the raw dataset."""
        raise NotImplementedError

    @abstractmethod
    def multiparallel_format(self) -> list[FormattedParallelText]:
        """Return texts as {id, data: {lang: text}, metadata: {...}}."""
        raise NotImplementedError
