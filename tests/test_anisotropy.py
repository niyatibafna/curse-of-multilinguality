from __future__ import annotations

import unittest

import numpy as np

from src.metrics.anisotropy import Anisotropy


class AnisotropyTest(unittest.TestCase):
    def setUp(self):
        self.embeddings = {
            "en": np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]]),
            "fr": np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 2.0]]),
        }

    def test_matches_explicit_gram_score(self):
        metric = Anisotropy(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=True,
        )
        stacked = np.vstack(list(self.embeddings.values()))
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        stacked = stacked / np.clip(norms, a_min=np.finfo(float).eps, a_max=None)
        expected = (
            np.sum(stacked @ stacked.T) - np.sum(stacked * stacked)
        ) / (stacked.shape[0] * (stacked.shape[0] - 1))

        self.assertAlmostEqual(metric.compute(), expected)

    def test_details_report_all_embeddings(self):
        score, details = Anisotropy(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            return_details=True,
        ).compute()

        self.assertIsInstance(score, float)
        self.assertEqual(details["num_embeddings"], 6)
        self.assertEqual(details["embedding_dim"], 2)


if __name__ == "__main__":
    unittest.main()
