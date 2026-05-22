from __future__ import annotations

from typing import Any

from .floresplus import FloresPlus
from .parallel_dataset import ParallelDataset, FormattedParallelText


DATASET_REGISTRY: dict[str, type[ParallelDataset]] = {
    "floresplus": FloresPlus,
    "flores+": FloresPlus,
}


def get_dataset_class(keyword: str) -> type[ParallelDataset]:
    key = keyword.lower()
    try:
        return DATASET_REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{keyword}'. Available datasets: {valid}") from exc


def load_dataset(keyword: str, **kwargs: Any) -> ParallelDataset:
    return get_dataset_class(keyword)(**kwargs)
