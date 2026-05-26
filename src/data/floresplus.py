from __future__ import annotations

import random
import re
from collections import defaultdict
from itertools import chain
from typing import Any, Iterable

from .parallel_dataset import FormattedParallelText, ParallelDataset


class FloresPlus(ParallelDataset):
    """FLORES+ downloader and formatter.

    The Hugging Face dataset is gated; accept its terms and authenticate with
    `huggingface_hub.login()` before downloading.
    """

    hf_dataset_name = "openlanguagedata/flores_plus"
    language_config_pattern = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}(?:_[A-Za-z0-9]+)?$")

    def __init__(
        self,
        split: str = "dev",
        languages: Iterable[str] | None = None,
        **load_kwargs: Any,
    ):
        super().__init__(split=split)
        self.languages = list(languages) if languages is not None else None
        self.load_kwargs = load_kwargs

    def download(self) -> Any:
        try:
            from datasets import get_dataset_config_names, load_dataset
        except ImportError as exc:
            raise ImportError("Install `datasets` to download FLORES+: pip install datasets") from exc

        common_kwargs = {
            "path": self.hf_dataset_name,
            "split": self.split,
            **self.load_kwargs,
        }

        languages = self._languages(get_dataset_config_names)

        datasets, skipped = self._load_language_datasets(languages, load_dataset, common_kwargs)
        if not datasets:
            raise ValueError(f"No FLORES+ languages provide split '{self.split}'.")
        return self._iter_language_datasets(datasets)

    def _languages(self, get_dataset_config_names: Any) -> list[str]:
        if self.languages is not None:
            return self.languages

        return [
            config
            for config in get_dataset_config_names(self.hf_dataset_name)
            if self._is_language_config(config)
        ]

    @classmethod
    def _is_language_config(cls, config: str) -> bool:
        return bool(cls.language_config_pattern.fullmatch(config))

    @staticmethod
    def _is_unknown_split_error(exc: ValueError) -> bool:
        message = str(exc)
        return "Unknown split" in message and "Should be one of" in message

    def _load_language_datasets(
        self,
        languages: Iterable[str],
        load_dataset: Any,
        common_kwargs: dict[str, Any],
    ) -> tuple[list[Any], list[str]]:
        datasets = []
        skipped = []
        for language in languages:
            try:
                datasets.append(load_dataset(name=language, **common_kwargs))
            except ValueError as exc:
                if not self._is_unknown_split_error(exc):
                    raise
                if self.languages is not None:
                    raise ValueError(
                        f"FLORES+ language '{language}' does not provide split '{self.split}'."
                    ) from exc
                skipped.append(language)

        if skipped:
            print(
                f"Skipping {len(skipped)} FLORES+ language(s) without split '{self.split}': "
                f"{', '.join(skipped)}",
                flush=True,
            )
        return datasets, skipped

    def _iter_language_datasets(self, datasets: Iterable[Iterable[Any]]) -> Iterable[Any]:
        return chain.from_iterable(datasets)

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
    **load_kwargs: Any,
) -> list[FormattedParallelText]:
    return FloresPlus(
        split=split,
        languages=languages,
        **load_kwargs,
    ).multiparallel_format()
