from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils.language_sorting.sort_by_wiki_count import (
    read_counts,
    read_dataset_langs,
    sort_langs_by_wiki_count,
)


class SortByWikiCountTest(unittest.TestCase):
    def test_sorts_and_keeps_missing_values_at_zero(self):
        rows = sort_langs_by_wiki_count(
            ["eng_Latn", "missing_Latn", "deu_Latn", "hin_Deva"],
            {"eng_Latn": "en", "deu_Latn": "de", "hin_Deva": "hi"},
            {"en": 10, "de": 7},
            quiet=True,
        )

        self.assertEqual(
            rows,
            [
                ("eng_Latn", 10),
                ("deu_Latn", 7),
                ("missing_Latn", 0),
                ("hin_Deva", 0),
            ],
        )

    def test_reads_raw_wikistats_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wikistats.csv"
            path.write_text("prefix,good\nen,10\nde,7\n")

            self.assertEqual(read_counts(path), {"en": 10, "de": 7})

    def test_reads_bundled_dataset_languages(self):
        langs = read_dataset_langs(
            Path("src/utils/language_sorting/dataset_languages.csv"),
            "wmt24pp",
        )

        self.assertIn("en", langs)
        self.assertIn("de_DE", langs)


if __name__ == "__main__":
    unittest.main()
