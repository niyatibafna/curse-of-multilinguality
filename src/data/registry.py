from __future__ import annotations

from typing import Any

from .bouquet import Bouquet
from .floresplus import FloresPlus
from .parallel_dataset import ParallelDataset, FormattedParallelText
from .wmt24pp import WMT24PP


DATASET_REGISTRY: dict[str, type[ParallelDataset]] = {
    "bouquet": Bouquet,
    "facebook/bouquet": Bouquet,
    "floresplus": FloresPlus,
    "flores+": FloresPlus,
    "wmt24++": WMT24PP,
    "wmt24pp": WMT24PP,
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
