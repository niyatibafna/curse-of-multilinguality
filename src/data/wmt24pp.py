from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from .parallel_dataset import FormattedParallelText, ParallelDataset


class WMT24PP(ParallelDataset):
    """WMT24++ downloader and formatter."""

    hf_dataset_name = "google/wmt24pp"
    language_pair_config_pattern = re.compile(r"^[a-z]{2,3}(?:_[A-Z]{2})?-[a-z]{2,3}(?:_[A-Z]{2})?$")

    def __init__(
        self,
        split: str = "dev",
        languages: Iterable[str] | None = None,
        **load_kwargs: Any,
    ):
        super().__init__(split=split)
        self.hf_split = "train" if split == "dev" else split
        self.languages = list(languages) if languages is not None else None
        self.load_kwargs = load_kwargs
        self._loaded_configs: list[str] = []

    def download(self) -> Any:
        try:
            from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
        except ImportError as exc:
            raise ImportError("Install `datasets` to download WMT24++: pip install datasets") from exc

        configs = self._configs(get_dataset_config_names)
        self._loaded_configs = configs
        datasets = [
            load_dataset(
                self.hf_dataset_name,
                config,
                split=self.hf_split,
                **self.load_kwargs,
            )
            for config in configs
        ]
        return concatenate_datasets(datasets)

    def multiparallel_format(self) -> list[FormattedParallelText]:
        raw_dataset = self.download()
        grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"data": {}, "metadata": {}})

        for row in raw_dataset:
            if row["domain"] == "canary":
                continue
            row_id = str(row["segment_id"])
            source_lang, target_lang = row["lp"].split("-", maxsplit=1)
            grouped[row_id]["data"][source_lang] = row["source"]
            grouped[row_id]["data"][target_lang] = row["target"]
            grouped[row_id]["metadata"]["domain"] = row["domain"]
            self._add_row_metadata(grouped[row_id]["metadata"], row)

        expected_languages = self._expected_languages()
        return [
            {
                "id": row_id,
                "data": {
                    language: values["data"][language]
                    for language in expected_languages
                },
                "metadata": values["metadata"],
            }
            for row_id, values in sorted(grouped.items())
            if all(language in values["data"] for language in expected_languages)
        ]

    def _configs(self, get_dataset_config_names: Any) -> list[str]:
        configs = [
            config
            for config in get_dataset_config_names(self.hf_dataset_name)
            if self._is_language_pair_config(config)
        ]
        if self.languages is None:
            return configs

        requested = set(self.languages)
        return [
            config
            for config in configs
            if self._matches_requested_language(config, requested)
        ]

    @classmethod
    def _is_language_pair_config(cls, config: str) -> bool:
        return bool(cls.language_pair_config_pattern.fullmatch(config))

    def _matches_requested_language(self, config: str, requested: set[str]) -> bool:
        source_lang, target_lang = config.split("-", maxsplit=1)
        return (
            config in requested
            or source_lang in requested
            or target_lang in requested
            or source_lang.split("_", maxsplit=1)[0] in requested
            or target_lang.split("_", maxsplit=1)[0] in requested
        )

    def _expected_languages(self) -> list[str]:
        languages = {
            language
            for config in self._loaded_configs
            for language in config.split("-", maxsplit=1)
        }
        return sorted(languages)

    def _add_row_metadata(self, metadata: dict[str, Any], row: dict[str, Any]) -> None:
        for key in ("domain", "document_id", "is_bad_source", "original_target"):
            if key in row:
                metadata.setdefault(key, row[key])


def load_wmt24pp(
    split: str = "dev",
    languages: Iterable[str] | None = None,
    **load_kwargs: Any,
) -> list[FormattedParallelText]:
    return WMT24PP(
        split=split,
        languages=languages,
        **load_kwargs,
    ).multiparallel_format()
