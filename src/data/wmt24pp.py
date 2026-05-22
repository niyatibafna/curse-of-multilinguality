from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .parallel_dataset import FormattedParallelText, ParallelDataset


class WMT24PP(ParallelDataset):
    """WMT24++ downloader and formatter."""

    hf_dataset_name = "google/wmt24pp"

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

    def download(self) -> Any:
        try:
            from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
        except ImportError as exc:
            raise ImportError("Install `datasets` to download WMT24++: pip install datasets") from exc

        configs = self._configs(get_dataset_config_names)
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
            row_id = str(row["segment_id"])
            source_lang, target_lang = row["lp"].split("-", maxsplit=1)
            grouped[row_id]["data"][source_lang] = row["source"]
            grouped[row_id]["data"][target_lang] = row["target"]
            grouped[row_id]["metadata"].setdefault("split", self.split)
            grouped[row_id]["metadata"].setdefault("source", self.hf_dataset_name)
            self._add_row_metadata(grouped[row_id]["metadata"], row)

        return [
            {"id": row_id, "data": values["data"], "metadata": values["metadata"]}
            for row_id, values in sorted(grouped.items())
        ]

    def _configs(self, get_dataset_config_names: Any) -> list[str]:
        if self.languages is None:
            return get_dataset_config_names(self.hf_dataset_name)
        return [
            language if language.startswith("en-") else f"en-{language}"
            for language in self.languages
            if language != "en"
        ]

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
