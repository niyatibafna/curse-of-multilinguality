from __future__ import annotations

import unittest

import numpy as np

from src.metrics.concept_space_dimensionality import (
    ConceptSpaceDimGrowthByLanguage,
    IndividualLanguageConceptDimensionality,
)
from src.metrics.language_subspace_dimensionality import (
    LanguageSpaceDimGrowthByLanguage,
    LanguageSpaceGrowthByConcepts,
)
from src.metrics.utils import (
    effective_rank,
    effective_rank_from_singular_values,
    pairwise_displacement_effective_rank,
)


class EffectiveRankTest(unittest.TestCase):
    def test_stable_rank_matches_equal_singular_values(self):
        singular_values = np.array([2.0, 2.0, 2.0])
        self.assertAlmostEqual(
            effective_rank_from_singular_values(singular_values, method="stable"),
            3.0,
        )

    def test_entropy_rank_matches_equal_singular_values(self):
        singular_values = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(
            effective_rank_from_singular_values(singular_values, method="entropy"),
            4.0,
        )

    def test_threshold_rank_counts_large_singular_values(self):
        singular_values = np.array([3.0, 0.1, 1e-13])
        self.assertEqual(
            effective_rank_from_singular_values(
                singular_values,
                method="threshold",
                singular_value_threshold=1e-12,
            ),
            2.0,
        )

    def test_effective_rank_centers_constant_matrix_to_zero(self):
        matrix = np.ones((5, 3))
        self.assertEqual(effective_rank(matrix, center=True), 0.0)

    def test_effective_rank_of_identity_without_centering(self):
        self.assertAlmostEqual(effective_rank(np.eye(4), center=False), 4.0)

    def test_effective_rank_can_normalize_by_embedding_dim(self):
        self.assertAlmostEqual(
            effective_rank(np.eye(4), center=False, normalize_by_dim=True),
            1.0,
        )

    def test_pairwise_displacement_rank_matches_explicit_matrix(self):
        groups = [
            np.array([
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 3.0],
                [2.0, 1.0, 1.0],
            ]),
            np.array([
                [1.0, 2.0, 0.0],
                [3.0, 0.0, 1.0],
                [2.0, 2.0, 2.0],
                [0.0, 1.0, 1.0],
            ]),
        ]
        explicit = np.vstack([
            group[i] - group[j]
            for group in groups
            for i in range(group.shape[0])
            for j in range(i + 1, group.shape[0])
        ])

        for method in ("stable", "entropy", "threshold"):
            with self.subTest(method=method):
                self.assertAlmostEqual(
                    pairwise_displacement_effective_rank(
                        groups,
                        embedding_dim=3,
                        method=method,
                        singular_value_threshold=1e-12,
                    ),
                    effective_rank(
                        explicit,
                        method=method,
                        center=True,
                        singular_value_threshold=1e-12,
                    ),
                )

    def test_pairwise_displacement_rank_can_normalize_by_embedding_dim(self):
        groups = [np.eye(3)]
        absolute = pairwise_displacement_effective_rank(groups, embedding_dim=3)
        normalized = pairwise_displacement_effective_rank(
            groups,
            embedding_dim=3,
            normalize_by_dim=True,
        )
        self.assertAlmostEqual(normalized, absolute / 3)


class ConceptDimensionalityMetricsTest(unittest.TestCase):
    def setUp(self):
        self.embeddings = {
            "b": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            "a": np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        }

    def test_individual_language_metric_returns_sorted_plot_data(self):
        metric = IndividualLanguageConceptDimensionality(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=False,
        )
        result = metric.compute()

        self.assertEqual(set(result["effective_dim_by_language"]), {"a", "b"})
        sorted_dims = result["sorted_effective_dims"]
        self.assertEqual(len(sorted_dims), 2)
        self.assertGreaterEqual(
            sorted_dims[0]["effective_dim"],
            sorted_dims[1]["effective_dim"],
        )

    def test_concept_dimensionality_normalizes_effective_dim_by_default(self):
        normalized = IndividualLanguageConceptDimensionality(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=False,
        ).compute()
        absolute = IndividualLanguageConceptDimensionality(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=False,
            normalize_effective_dim=False,
        ).compute()

        language = "b"
        self.assertAlmostEqual(
            normalized["effective_dim_by_language"][language],
            absolute["effective_dim_by_language"][language] / 2,
        )

    def test_individual_concept_dimensionality_uses_concept_displacements(self):
        result = IndividualLanguageConceptDimensionality(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=False,
            normalize_effective_dim=False,
        ).compute()
        language = "b"
        explicit = np.vstack([
            self.embeddings[language][i] - self.embeddings[language][j]
            for i in range(3)
            for j in range(i + 1, 3)
        ])

        self.assertAlmostEqual(
            result["effective_dim_by_language"][language],
            effective_rank(explicit, center=True),
        )

    def test_concept_space_dim_growth_by_language_uses_same_language_displacements(self):
        result = ConceptSpaceDimGrowthByLanguage(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=False,
            language_order_seed=0,
            normalize_effective_dim=False,
        ).compute()
        languages = result["language_order"][:2]
        explicit = np.vstack([
            self.embeddings[language][i] - self.embeddings[language][j]
            for language in languages
            for i in range(3)
            for j in range(i + 1, 3)
        ])

        self.assertAlmostEqual(
            result["concept_space_dim_growth_by_language"][1]["effective_dim"],
            effective_rank(explicit, center=True),
        )


class LanguageSubspaceDimensionalityMetricsTest(unittest.TestCase):
    def setUp(self):
        self.embeddings = {
            f"lang_{index}": np.array([
                [index + 1.0, 0.0, 1.0],
                [0.0, index + 2.0, 2.0],
                [1.0, 1.0, index + 3.0],
            ])
            for index in range(6)
        }

    def test_language_scaling_starts_at_two_and_records_order(self):
        metric = LanguageSpaceDimGrowthByLanguage(
            self.embeddings,
            num_concepts=3,
            num_languages=6,
            embedding_dim=3,
            normalize=False,
            language_order_seed=7,
            normalize_effective_dim=False,
        )
        result = metric.compute()

        self.assertEqual(
            [row["num_languages"] for row in result["language_subspace_scaling"]],
            [2, 5, 6],
        )
        self.assertEqual(len(result["language_order"]), 6)
        self.assertEqual(set(result["language_order"]), set(self.embeddings))

    def test_language_scaling_matches_explicit_displacements(self):
        metric = LanguageSpaceDimGrowthByLanguage(
            self.embeddings,
            num_concepts=3,
            num_languages=6,
            embedding_dim=3,
            normalize=False,
            language_order_seed=7,
            normalize_effective_dim=False,
        )
        result = metric.compute()
        selected_languages = result["language_order"][:2]
        explicit = np.vstack([
            self.embeddings[selected_languages[0]][concept_index]
            - self.embeddings[selected_languages[1]][concept_index]
            for concept_index in range(3)
        ])

        self.assertAlmostEqual(
            result["language_subspace_scaling"][0]["effective_dim"],
            effective_rank(explicit, center=True),
        )

    def test_language_scaling_normalizes_effective_dim_by_default(self):
        normalized = LanguageSpaceDimGrowthByLanguage(
            self.embeddings,
            num_concepts=3,
            num_languages=6,
            embedding_dim=3,
            normalize=False,
            language_order_seed=7,
        ).compute()
        absolute = LanguageSpaceDimGrowthByLanguage(
            self.embeddings,
            num_concepts=3,
            num_languages=6,
            embedding_dim=3,
            normalize=False,
            language_order_seed=7,
            normalize_effective_dim=False,
        ).compute()

        self.assertAlmostEqual(
            normalized["language_subspace_scaling"][0]["effective_dim"],
            absolute["language_subspace_scaling"][0]["effective_dim"] / 3,
        )

    def test_language_space_growth_by_concepts_uses_concept_steps(self):
        metric = LanguageSpaceGrowthByConcepts(
            self.embeddings,
            num_concepts=3,
            num_languages=6,
            embedding_dim=3,
            normalize=False,
            concept_step=2,
            normalize_effective_dim=False,
        )
        result = metric.compute()

        self.assertEqual(
            [row["num_concepts"] for row in result["language_space_growth_by_concepts"]],
            [2, 3],
        )

    def test_language_space_growth_by_concepts_matches_explicit_displacements(self):
        result = LanguageSpaceGrowthByConcepts(
            self.embeddings,
            num_concepts=3,
            num_languages=6,
            embedding_dim=3,
            normalize=False,
            concept_step=3,
            normalize_effective_dim=False,
        ).compute()
        languages = list(self.embeddings)
        explicit = np.vstack([
            self.embeddings[languages[i]][concept_index]
            - self.embeddings[languages[j]][concept_index]
            for concept_index in range(3)
            for i in range(len(languages))
            for j in range(i + 1, len(languages))
        ])

        self.assertAlmostEqual(
            result["language_space_growth_by_concepts"][0]["effective_dim"],
            effective_rank(explicit, center=True),
        )

    def test_language_space_growth_by_concepts_normalizes_by_default(self):
        normalized = LanguageSpaceGrowthByConcepts(
            self.embeddings,
            num_concepts=3,
            num_languages=6,
            embedding_dim=3,
            normalize=False,
            concept_step=3,
        ).compute()
        absolute = LanguageSpaceGrowthByConcepts(
            self.embeddings,
            num_concepts=3,
            num_languages=6,
            embedding_dim=3,
            normalize=False,
            concept_step=3,
            normalize_effective_dim=False,
        ).compute()

        self.assertAlmostEqual(
            normalized["language_space_growth_by_concepts"][0]["effective_dim"],
            absolute["language_space_growth_by_concepts"][0]["effective_dim"] / 3,
        )


if __name__ == "__main__":
    unittest.main()
