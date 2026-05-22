from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .parallel_dataset import FormattedParallelText, ParallelDataset


class Bouquet(ParallelDataset):
    """facebook/bouquet downloader and formatter."""

    hf_dataset_name = "facebook/bouquet"

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
            from datasets import concatenate_datasets, load_dataset
        except ImportError as exc:
            raise ImportError("Install `datasets` to download BOUQuET: pip install datasets") from exc

        if self.languages is None:
            return load_dataset(
                self.hf_dataset_name,
                self.config,
                split=self.split,
                **self.load_kwargs,
            )

        datasets = [
            load_dataset(
                self.hf_dataset_name,
                language,
                split=self.split,
                **self.load_kwargs,
            )
            for language in self.languages
            if language != "eng_Latn"
        ]
        if not datasets:
            return load_dataset(
                self.hf_dataset_name,
                "eng_Latn",
                split=self.split,
                **self.load_kwargs,
            )
        return concatenate_datasets(datasets)

    def multiparallel_format(self) -> list[FormattedParallelText]:
        raw_dataset = self.download()
        grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"data": {}, "metadata": {}})

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
