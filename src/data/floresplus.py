from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .parallel_dataset import FormattedParallelText, ParallelDataset


class FloresPlus(ParallelDataset):
    """FLORES+ downloader and formatter.

    The Hugging Face dataset is gated; accept its terms and authenticate with
    `huggingface_hub.login()` before downloading.
    """

    hf_dataset_name = "openlanguagedata/flores_plus"

    def __init__(
        self,
        split: str = "dev",
        languages: Iterable[str] | None = None,
        cache_dir: str | Path | None = None,
        **load_kwargs: Any,
    ):
        super().__init__(split=split, cache_dir=cache_dir)
        self.languages = list(languages) if languages is not None else None
        self.load_kwargs = load_kwargs

    def download(self) -> Any:
        try:
            from datasets import concatenate_datasets, load_dataset
        except ImportError as exc:
            raise ImportError("Install `datasets` to download FLORES+: pip install datasets") from exc

        common_kwargs = {
            "path": self.hf_dataset_name,
            "split": self.split,
            "cache_dir": self.cache_dir,
            **self.load_kwargs,
        }

        if self.languages is None:
            return load_dataset(**common_kwargs)

        datasets = [
            load_dataset(name=language, **common_kwargs)
            for language in self.languages
        ]
        return concatenate_datasets(datasets)

    def multiparallel_format(self) -> list[FormattedParallelText]:
        raw_dataset = self.download()
        grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"data": {}, "metadata": {}})

        for row in raw_dataset:
            row_id = str(row["id"])
            language = self._language_key(row, grouped[row_id]["data"])
            grouped[row_id]["data"][language] = row["text"]
            grouped[row_id]["metadata"].setdefault("split", row.get("split", self.split))
            grouped[row_id]["metadata"].setdefault("source", self.hf_dataset_name)
            self._add_row_metadata(grouped[row_id]["metadata"], row)

        texts = [
            {"id": row_id, "data": values["data"], "metadata": values["metadata"]}
            for row_id, values in sorted(grouped.items(), key=lambda item: self._sort_key(item[0]))
        ]
        FLORES_PLUS_SHUFFLE_SEED = 42
        random.Random(FLORES_PLUS_SHUFFLE_SEED).shuffle(texts)
        return texts

    def _language_key(self, row: dict[str, Any], existing_data: dict[str, str]) -> str:
        base_parts = [row["iso_639_3"], row["iso_15924"]]
        if row.get("variant"):
            base_parts.append(str(row["variant"]))

        language = "_".join(base_parts)
        if language not in existing_data:
            return language

        parts = [*base_parts, str(row.get("glottocode", "unknown"))]
        return "_".join(parts)

    def _add_row_metadata(self, metadata: dict[str, Any], row: dict[str, Any]) -> None:
        for key in ("url", "domain", "topic", "has_image", "has_hyperlink"):
            if key in row:
                metadata.setdefault(key, row[key])

    def _sort_key(self, row_id: str) -> tuple[int, int | str]:
        try:
            return (0, int(row_id))
        except ValueError:
            return (1, row_id)


def load_floresplus(
    split: str = "dev",
    languages: Iterable[str] | None = None,
    cache_dir: str | Path | None = None,
    **load_kwargs: Any,
) -> list[FormattedParallelText]:
    return FloresPlus(
        split=split,
        languages=languages,
        cache_dir=cache_dir,
        **load_kwargs,
    ).multiparallel_format()
