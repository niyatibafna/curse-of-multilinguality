from __future__ import annotations

import unittest

import numpy as np

from src.scripts.run_metrics import embedding_cache_metadata, resolve_pooling, slice_embeddings


class EmbeddingCacheTest(unittest.TestCase):
    def setUp(self):
        self.texts = [
            {"id": "3", "data": {"en": "three", "fr": "trois"}},
            {"id": "1", "data": {"en": "one", "fr": "un"}},
            {"id": "2", "data": {"en": "two", "fr": "deux"}},
        ]

    def test_embedding_metadata_uses_full_texts(self):
        metadata = embedding_cache_metadata(
            model_name="model",
            model_type="model_type",
            dataset_name="dataset",
            split="dev",
            dataset_languages=None,
            eval_languages=["en", "fr"],
            texts=self.texts,
            layer=-1,
            batch_size=8,
            pooling="last_token",
        )

        self.assertEqual(metadata["num_texts"], 3)
        self.assertNotIn("requested_max_texts", metadata)

    def test_slice_embeddings_takes_first_rows(self):
        embeddings = {
            "en": np.arange(12).reshape(4, 3),
            "fr": np.arange(100, 112).reshape(4, 3),
        }

        sliced = slice_embeddings(embeddings, 2)

        np.testing.assert_array_equal(sliced["en"], embeddings["en"][:2])
        np.testing.assert_array_equal(sliced["fr"], embeddings["fr"][:2])

    def test_slice_embeddings_none_returns_original(self):
        embeddings = {"en": np.zeros((2, 3))}

        self.assertIs(slice_embeddings(embeddings, None), embeddings)

    def test_resolve_pooling_uses_model_default(self):
        self.assertEqual(resolve_pooling("mbert", None), "cls")
        self.assertEqual(resolve_pooling("llama", None), "last_token")
        self.assertEqual(resolve_pooling("mbert", "mean"), "mean")


if __name__ == "__main__":
    unittest.main()
