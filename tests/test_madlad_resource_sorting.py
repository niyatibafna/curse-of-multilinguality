from __future__ import annotations

import unittest

from src.utils.language_sorting.sort_by_madlad_resource import (
    sort_madlad_codes,
    sort_langs_by_madlad_resource,
)
from src.training.make_language_plan import (
    code_features,
    codes_match,
    madlad_resource_column,
    madlad_resource_scores,
    manual_code_match,
)
from src.utils.language_sorting.update_madlad_counts import parse_count, parse_madlad_counts


class MadladResourceSortingTest(unittest.TestCase):
    def test_sorts_all_madlad_codes(self):
        rows = sort_madlad_codes({"en": 10, "de": 7, "hi": 9})

        self.assertEqual(rows, [("en", 10), ("hi", 9), ("de", 7)])

    def test_parse_count_suffixes(self):
        self.assertEqual(parse_count("4.1M"), 4_100_000)
        self.assertEqual(parse_count("2.6 T"), 2_600_000_000_000)
        self.assertEqual(parse_count("191"), 191)

    def test_parse_final_dataset_table(self):
        markdown = """
## Final Dataset information

BCP-47          | docs (noisy)   | docs (clean)   | sents (noisy)   | sents (clean)   | toks (noisy)   | toks (clean)   | chars (noisy)   | chars (clean)   | clean    | noisy    |
----------------|:---------------|:---------------|:----------------|:----------------|:---------------|:---------------|:----------------|:----------------|:---------|:---------|
total*          | 7.2B           | 3.7B           | 133.1B          | 97.5B           | 4.6T           | 2.6T           | 30.6T           | 16.0T           | 11.4 T   | 6.3 T
en*             | 3.0B           | 1.5B           | 71.1B           | 45.4B           | 2.0T           | 1.3T           | 12.3T           | 7.6T            | 2.6 T    | 4.3 T    |
de              | 478.6M         | 225.1M         | 11.5B           | 6B              | 299.5B         | 139.6B         | 2.2T            | 1T              | 370.6 G  | 815.5 G  |
"""
        rows = parse_madlad_counts(markdown)

        self.assertEqual(rows[0]["madlad_code"], "en")
        self.assertEqual(rows[0]["clean_docs"], 1_500_000_000)
        self.assertEqual(rows[1]["clean_bytes"], 370_600_000_000)

    def test_sorts_and_keeps_missing_values_at_zero(self):
        rows = sort_langs_by_madlad_resource(
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

    def test_language_plan_resource_scores_use_byte_counts(self):
        scores = madlad_resource_scores("unused", "clean")

        self.assertEqual(scores["en"], 2_600_000_000_000)
        self.assertEqual(scores["ru"], 832_900_000_000)

    def test_language_plan_resource_split_maps_to_count_column(self):
        self.assertEqual(madlad_resource_column("clean"), "clean_bytes")
        self.assertEqual(madlad_resource_column("noisy"), "noisy_bytes")
        self.assertEqual(madlad_resource_column("clean_tokens"), "clean_tokens")

    def test_language_plan_uses_explicit_script_only(self):
        features = code_features("zh_Hant")

        self.assertEqual(features["scripts"], {"Hant"})

    def test_language_plan_matches_manual_language_aliases(self):
        aliases = [
            ("ar", "arb_Arab"),
            ("zh", "cmn_Hans"),
            ("he", "iw"),
            ("id", "in"),
            ("yi", "ji"),
            ("ro", "mo"),
            ("jv", "jw"),
            ("ko", "kor_Hang"),
        ]

        for left, right in aliases:
            with self.subTest(left=left, right=right):
                self.assertTrue(codes_match(code_features(left), code_features(right)))

    def test_language_plan_aliases_still_respect_explicit_scripts(self):
        self.assertFalse(codes_match(code_features("zh_Hans"), code_features("cmn_Hant")))

    def test_language_plan_manual_code_matches(self):
        self.assertTrue(manual_code_match("fa", "pes_Arab"))
        self.assertTrue(manual_code_match("no", "nno_Latn"))
        self.assertTrue(manual_code_match("no", "nob_Latn"))
        self.assertFalse(manual_code_match("no", "nob_Latn_radical"))
        self.assertFalse(manual_code_match("ar", "apc_Arab"))


if __name__ == "__main__":
    unittest.main()
