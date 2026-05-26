from __future__ import annotations

import unittest

from src.data.bouquet import Bouquet
from src.data.floresplus import FloresPlus
from src.data.registry import DATASET_REGISTRY
from src.data.wmt24pp import WMT24PP


def ids(rows):
    return [row["id"] for row in rows]


def bouquet_download(self):
    return [[
        {
            "uniq_id": "2",
            "src_lang": "eng_Latn",
            "src_text": "two",
            "tgt_lang": "fra_Latn",
            "tgt_text": "deux",
        },
        {
            "uniq_id": "1",
            "src_lang": "eng_Latn",
            "src_text": "one",
            "tgt_lang": "fra_Latn",
            "tgt_text": "un",
        },
    ]]


def floresplus_download(self):
    return [
        {
            "id": "2",
            "iso_639_3": "eng",
            "iso_15924": "Latn",
            "variant": None,
            "text": "two",
        },
        {
            "id": "1",
            "iso_639_3": "eng",
            "iso_15924": "Latn",
            "variant": None,
            "text": "one",
        },
        {
            "id": "3",
            "iso_639_3": "eng",
            "iso_15924": "Latn",
            "variant": None,
            "text": "three",
        },
    ]


def wmt24pp_download(self):
    self._loaded_configs = ["en-fr"]
    return [
        {
            "segment_id": "2",
            "lp": "en-fr",
            "source": "two",
            "target": "deux",
            "domain": "news",
        },
        {
            "segment_id": "1",
            "lp": "en-fr",
            "source": "one",
            "target": "un",
            "domain": "news",
        },
    ]


class DatasetDeterminismTest(unittest.TestCase):
    cases = [
        ("bouquet", Bouquet, bouquet_download),
        ("floresplus", FloresPlus, floresplus_download),
        ("wmt24pp", WMT24PP, wmt24pp_download),
    ]

    def test_dataset_prefixes_are_deterministic(self):
        for dataset_name, cls, download in self.cases:
            with self.subTest(dataset=dataset_name):
                self.assertIs(DATASET_REGISTRY[dataset_name], cls)
                original_download = cls.download
                cls.download = download
                try:
                    first = cls().multiparallel_format()
                    second = cls().multiparallel_format()
                finally:
                    cls.download = original_download

                self.assertEqual(first, second)
                for k in range(1, len(first) + 1):
                    self.assertEqual(ids(first[:k]), ids(second[:k]))


class FloresPlusDownloadTest(unittest.TestCase):
    def test_download_returns_iterable_without_concatenating_features(self):
        dataset = FloresPlus()
        rows = dataset._iter_language_datasets([[{"id": "1"}], [{"id": "2"}]])
        self.assertEqual(list(rows), [{"id": "1"}, {"id": "2"}])

    def test_download_skips_auto_languages_missing_requested_split(self):
        def fake_load_dataset(name, **kwargs):
            if name == "cat_Latn_vale1252":
                raise ValueError('Unknown split "dev". Should be one of [\'devtest\'].')
            return [name]

        dataset = FloresPlus()
        datasets, skipped = dataset._load_language_datasets(
            ["eng_Latn", "cat_Latn_vale1252"],
            fake_load_dataset,
            {"path": dataset.hf_dataset_name, "split": "dev"},
        )

        self.assertEqual(datasets, [["eng_Latn"]])
        self.assertEqual(skipped, ["cat_Latn_vale1252"])

    def test_download_raises_for_requested_language_missing_requested_split(self):
        def fake_load_dataset(name, **kwargs):
            raise ValueError('Unknown split "dev". Should be one of [\'devtest\'].')

        dataset = FloresPlus(languages=["cat_Latn_vale1252"])
        with self.assertRaisesRegex(ValueError, "cat_Latn_vale1252"):
            dataset._load_language_datasets(
                ["cat_Latn_vale1252"],
                fake_load_dataset,
                {"path": dataset.hf_dataset_name, "split": "dev"},
            )


if __name__ == "__main__":
    unittest.main()
