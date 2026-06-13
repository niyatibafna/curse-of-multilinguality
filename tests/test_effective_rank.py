from __future__ import annotations

import unittest

import numpy as np

from src.metrics.interaction_between_concept_and_language import (
    Comness,
    ConceptLanguagePrincipalAngleOverlap,
    EffLangspaceDimProp,
)
from src.metrics.concept_space_dimensionality import (
    ConceptSpaceDimGrowthByConcept,
    ConceptSpaceDimGrowthByLanguage,
    IndividualLanguageConceptDimensionality,
)
from src.metrics.language_subspace_dimensionality import (
    LanguageSpaceDimGrowthByLanguage,
    LanguageSpaceGrowthByConcepts,
)
from src.metrics.utils import (
    PairwiseDisplacementMoments,
    centered_gram_from_moments,
    effective_rank,
    effective_rank_from_singular_values,
    pairwise_displacement_effective_rank,
    pairwise_displacement_subspace_basis,
    random_baseline_effective_rank,
    random_groups_like,
    stack_language_embeddings,
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

    def test_random_groups_like_preserves_group_sizes(self):
        pool = np.arange(30, dtype=float).reshape(10, 3)
        rng = np.random.default_rng(0)

        groups = random_groups_like(pool, [2, 3, 5], rng)

        self.assertEqual([group.shape for group in groups], [(2, 3), (3, 3), (5, 3)])

    def test_centered_gram_from_moments_matches_explicit_displacements(self):
        groups = [
            np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]),
            np.array([[0.0, 2.0], [3.0, 1.0]]),
        ]
        explicit = np.vstack([
            group[i] - group[j]
            for group in groups
            for i in range(group.shape[0])
            for j in range(i + 1, group.shape[0])
        ])
        centered = explicit - np.mean(explicit, axis=0, keepdims=True)

        moments = PairwiseDisplacementMoments(embedding_dim=2)
        for group in groups:
            moments.update(group)

        np.testing.assert_allclose(centered_gram_from_moments(moments), centered.T @ centered)

    def test_pairwise_displacement_subspace_basis_uses_energy_threshold(self):
        groups = [np.diag([3.0, 1.0, 0.0])]

        basis, dim, energy = pairwise_displacement_subspace_basis(
            groups,
            embedding_dim=3,
            energy_threshold=0.8,
        )

        self.assertEqual(basis.shape, (3, dim))
        self.assertGreaterEqual(energy, 0.8)
        np.testing.assert_allclose(basis.T @ basis, np.eye(dim), atol=1e-12)

    def test_stack_language_embeddings_validates_and_stacks(self):
        embeddings = {
            "en": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "fr": np.array([[2.0, 0.0], [0.0, 2.0]]),
        }

        stacked = stack_language_embeddings(
            embeddings,
            num_concepts=2,
            num_languages=2,
            embedding_dim=2,
        )

        self.assertEqual(stacked.shape, (2, 2, 2))
        np.testing.assert_array_equal(stacked[0], embeddings["en"])


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

    def test_concept_space_dim_growth_by_language_reports_random_baseline(self):
        result = ConceptSpaceDimGrowthByLanguage(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=False,
            random_baseline_trials=2,
            random_baseline_seed=123,
        ).compute()

        row = result["concept_space_dim_growth_by_language"][0]
        self.assertIn("random_effective_dim_mean", row)
        self.assertIn("random_effective_dim_std", row)
        self.assertEqual(row["random_baseline_trials"], 2)
        self.assertIn("effective_dim_ratio", row)

    def test_concept_space_dim_growth_by_concept_uses_concept_steps(self):
        result = ConceptSpaceDimGrowthByConcept(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=False,
            concept_step=2,
            normalize_effective_dim=False,
        ).compute()

        self.assertEqual(
            [row["num_concepts"] for row in result["concept_space_dim_growth_by_concept"]],
            [2, 3],
        )

    def test_concept_space_dim_growth_by_concept_matches_explicit_displacements(self):
        result = ConceptSpaceDimGrowthByConcept(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=False,
            concept_step=3,
            normalize_effective_dim=False,
        ).compute()
        explicit = np.vstack([
            embeddings[i] - embeddings[j]
            for embeddings in self.embeddings.values()
            for i in range(3)
            for j in range(i + 1, 3)
        ])

        self.assertAlmostEqual(
            result["concept_space_dim_growth_by_concept"][0]["effective_dim"],
            effective_rank(explicit, center=True),
        )

    def test_concept_space_dim_growth_by_concept_normalizes_by_default(self):
        normalized = ConceptSpaceDimGrowthByConcept(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=False,
            concept_step=3,
        ).compute()
        absolute = ConceptSpaceDimGrowthByConcept(
            self.embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=2,
            normalize=False,
            concept_step=3,
            normalize_effective_dim=False,
        ).compute()

        self.assertAlmostEqual(
            normalized["concept_space_dim_growth_by_concept"][0]["effective_dim"],
            absolute["concept_space_dim_growth_by_concept"][0]["effective_dim"] / 2,
        )


class ComnessTest(unittest.TestCase):
    def setUp(self):
        self.embeddings = {
            "a": np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]),
            "b": np.array([
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.5],
                [0.5, 0.0, 1.0],
                [1.0, 0.5, 0.5],
            ]),
            "c": np.array([
                [0.5, 1.0, 0.0],
                [0.0, 0.5, 1.0],
                [1.0, 0.0, 0.5],
                [0.5, 1.0, 0.5],
            ]),
        }

    def test_reports_random_baseline_normalized_comness(self):
        score, details = Comness(
            self.embeddings,
            num_concepts=4,
            num_languages=3,
            embedding_dim=3,
            normalize=False,
            return_details=True,
            random_baseline_trials=2,
            random_baseline_seed=123,
        ).compute()

        self.assertGreaterEqual(score, 0.0)
        self.assertIn("normalized_comness", details)
        self.assertIn("d_lang_ratio", details)
        self.assertIn("d_concept_ratio", details)
        self.assertEqual(details["random_baseline_trials"], 2)

    def test_comness_random_baselines_match_language_and_concept_group_shapes(self):
        _, details = Comness(
            self.embeddings,
            num_concepts=4,
            num_languages=3,
            embedding_dim=3,
            normalize=False,
            return_details=True,
            random_baseline_trials=2,
            random_baseline_seed=123,
        ).compute()
        pool = np.vstack(list(self.embeddings.values()))
        rng = np.random.default_rng(123)
        lang_baseline = random_baseline_effective_rank(
            pool,
            [3] * 4,
            embedding_dim=3,
            trials=2,
            rng=rng,
        )
        concept_baseline = random_baseline_effective_rank(
            pool,
            [4] * 3,
            embedding_dim=3,
            trials=2,
            rng=rng,
        )

        self.assertAlmostEqual(
            details["d_lang_random_effective_dim_mean"],
            lang_baseline["random_effective_dim_mean"],
        )
        self.assertAlmostEqual(
            details["d_concept_random_effective_dim_mean"],
            concept_baseline["random_effective_dim_mean"],
        )

    def test_random_baseline_trials_zero_keeps_legacy_details_only(self):
        _, details = Comness(
            self.embeddings,
            num_concepts=4,
            num_languages=3,
            embedding_dim=3,
            normalize=False,
            return_details=True,
            random_baseline_trials=0,
        ).compute()

        self.assertIn("d_lang", details)
        self.assertNotIn("normalized_comness", details)

    def test_eff_langspace_dim_prop_uses_pairwise_total_displacements(self):
        result = EffLangspaceDimProp(
            self.embeddings,
            num_concepts=4,
            num_languages=3,
            embedding_dim=3,
            normalize=False,
        ).compute()
        X = stack_language_embeddings(self.embeddings, 4, 3, 3)
        d_lang = pairwise_displacement_effective_rank(
            (X[:, c, :] for c in range(4)),
            embedding_dim=3,
        )
        d_total = pairwise_displacement_effective_rank(
            [X.reshape(-1, 3)],
            embedding_dim=3,
        )

        self.assertAlmostEqual(result["d_lang"], d_lang)
        self.assertAlmostEqual(result["d_total"], d_total)
        self.assertAlmostEqual(result["score"], d_lang / d_total)
        self.assertEqual(result["num_language_displacements"], 12)
        self.assertEqual(result["num_total_displacements"], 66)

    def test_eff_langspace_dim_prop_requires_two_languages(self):
        with self.assertRaises(ValueError):
            EffLangspaceDimProp(
                {"a": self.embeddings["a"]},
                num_concepts=4,
                num_languages=1,
                embedding_dim=3,
                normalize=False,
            ).compute()


class ConceptLanguagePrincipalAngleOverlapTest(unittest.TestCase):
    def setUp(self):
        self.embeddings = {
            "a": np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]),
            "b": np.array([
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ]),
            "c": np.array([
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
            ]),
        }

    def test_reports_orthogonal_concept_and_language_subspaces(self):
        result = ConceptLanguagePrincipalAngleOverlap(
            self.embeddings,
            num_concepts=3,
            num_languages=3,
            embedding_dim=3,
            normalize=False,
        ).compute()

        self.assertAlmostEqual(result["mean_squared_cosine"], 0.0)
        self.assertAlmostEqual(result["max_cosine"], 0.0)
        self.assertEqual(result["concept_subspace_dim"], 1)
        self.assertEqual(result["language_subspace_dim"], 1)
        self.assertAlmostEqual(result["random_expected_mean_squared_cosine"], 1 / 3)
        self.assertAlmostEqual(result["adjusted_mean_squared_cosine"], -0.5)

    def test_matches_explicit_principal_angles(self):
        embeddings = {
            "a": np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            "b": np.array([
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
            ]),
        }
        result = ConceptLanguagePrincipalAngleOverlap(
            embeddings,
            num_concepts=3,
            num_languages=2,
            embedding_dim=3,
            normalize=False,
            subspace_energy_threshold=1.0,
        ).compute()

        concept = np.vstack([
            matrix[i] - matrix[j]
            for matrix in embeddings.values()
            for i in range(3)
            for j in range(i + 1, 3)
        ])
        languages = list(embeddings)
        language = np.vstack([
            embeddings[languages[i]][concept_index]
            - embeddings[languages[j]][concept_index]
            for concept_index in range(3)
            for i in range(len(languages))
            for j in range(i + 1, len(languages))
        ])
        expected = _principal_angle_cosines(concept, language)

        np.testing.assert_allclose(
            result["principal_angle_cosines"],
            expected,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            result["mean_squared_cosine"],
            float(np.mean(expected ** 2)),
        )
        random_expected = max(
            result["concept_subspace_dim"],
            result["language_subspace_dim"],
        ) / 3
        self.assertAlmostEqual(
            result["random_expected_mean_squared_cosine"],
            random_expected,
        )
        if random_expected < 1:
            self.assertAlmostEqual(
                result["adjusted_mean_squared_cosine"],
                (result["mean_squared_cosine"] - random_expected) / (1 - random_expected),
            )


def _principal_angle_cosines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first - np.mean(first, axis=0, keepdims=True)
    second = second - np.mean(second, axis=0, keepdims=True)
    _, first_singular_values, first_vt = np.linalg.svd(first, full_matrices=False)
    _, second_singular_values, second_vt = np.linalg.svd(second, full_matrices=False)
    first_basis = first_vt[first_singular_values > np.finfo(float).eps].T
    second_basis = second_vt[second_singular_values > np.finfo(float).eps].T
    return np.linalg.svd(first_basis.T @ second_basis, compute_uv=False)


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

    def test_language_scaling_reports_random_baseline(self):
        result = LanguageSpaceDimGrowthByLanguage(
            self.embeddings,
            num_concepts=3,
            num_languages=6,
            embedding_dim=3,
            normalize=False,
            random_baseline_trials=2,
            random_baseline_seed=123,
        ).compute()

        row = result["language_subspace_scaling"][0]
        self.assertIn("random_effective_dim_mean", row)
        self.assertIn("random_effective_dim_std", row)
        self.assertEqual(row["random_baseline_trials"], 2)
        self.assertIn("effective_dim_ratio", row)

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
