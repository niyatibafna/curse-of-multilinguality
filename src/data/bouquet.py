from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from .parallel_dataset import FormattedParallelText, ParallelDataset


class Bouquet(ParallelDataset):
    """facebook/bouquet downloader and formatter."""

    hf_dataset_name = "facebook/bouquet"
    language_config_pattern = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")

    def __init__(
        self,
        split: str = "dev",
        languages: Iterable[str] | None = None,
        config: str = "sentence_level",
        **load_kwargs: Any,
    ):
        super().__init__(split=split)
        self.languages = list(languages) if languages is not None else None
        self.config = config
        self.load_kwargs = load_kwargs

    def download(self) -> Any:
        try:
            from datasets import get_dataset_config_names, load_dataset
        except ImportError as exc:
            raise ImportError("Install `datasets` to download BOUQuET: pip install datasets") from exc

        languages = self._languages(get_dataset_config_names)
        datasets = []
        for index, language in enumerate(languages, start=1):
            print(f"Loading BOUQuET language {index}/{len(languages)}: {language}", flush=True)
            datasets.append(load_dataset(
                self.hf_dataset_name,
                language,
                split=self.split,
                **self.load_kwargs,
            ))
        return datasets

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

    def multiparallel_format(self) -> list[FormattedParallelText]:
        raw_datasets = self.download()
        grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"data": {}, "metadata": {}})

        for raw_dataset in raw_datasets:
            for row in raw_dataset:
                row_id = str(row["uniq_id"])
                grouped[row_id]["data"][row["src_lang"]] = row["src_text"]
                grouped[row_id]["data"][row["tgt_lang"]] = row["tgt_text"]
                grouped[row_id]["metadata"].setdefault("split", row.get("split", self.split))
                grouped[row_id]["metadata"].setdefault("source", self.hf_dataset_name)
                self._add_row_metadata(grouped[row_id]["metadata"], row)

        return [
            {"id": row_id, "data": values["data"], "metadata": values["metadata"]}
            for row_id, values in sorted(grouped.items())
        ]

    def _add_row_metadata(self, metadata: dict[str, Any], row: dict[str, Any]) -> None:
        for key in ("level", "domain", "register", "tags", "par_id", "par_comment"):
            if key in row:
                metadata.setdefault(key, row[key])


def load_bouquet(
    split: str = "dev",
    languages: Iterable[str] | None = None,
    config: str = "sentence_level",
    **load_kwargs: Any,
) -> list[FormattedParallelText]:
    return Bouquet(
        split=split,
        languages=languages,
        config=config,
        **load_kwargs,
    ).multiparallel_format()
