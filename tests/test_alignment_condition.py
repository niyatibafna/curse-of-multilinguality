from __future__ import annotations

import numpy as np

from src.metrics.multilinguality_conditions import (
    AlignmentCondition,
    MonolingualStructureCondition,
)


def test_alignment_condition_all_pairs_succeed() -> None:
    embeddings = {
        "en": np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        "fr": np.array([[0.9, 0.1], [0.1, 0.9], [-0.9, -0.1]]),
    }

    result = AlignmentCondition(
        embeddings,
        num_concepts=3,
        num_languages=2,
        embedding_dim=2,
    ).compute()

    assert result["score"] == 1.0
    assert result["num_success"] == 6
    assert result["num_pairs"] == 6
    assert [row["num_success"] for row in result["language_pairs"]] == [3, 3]


def test_alignment_condition_requires_strict_closeness() -> None:
    embeddings = {
        "en": np.array([[1.0, 0.0], [1.0, 0.0]]),
        "fr": np.array([[0.0, 1.0], [0.0, 1.0]]),
    }

    result = AlignmentCondition(
        embeddings,
        num_concepts=2,
        num_languages=2,
        embedding_dim=2,
    ).compute()

    assert result["score"] == 0.0
    assert result["num_success"] == 0


def test_alignment_condition_language_pair_counts_recover_subset() -> None:
    embeddings = {
        "en": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "fr": np.array([[0.9, 0.1], [0.1, 0.9]]),
        "de": np.array([[0.8, 0.2], [0.2, 0.8]]),
    }

    result = AlignmentCondition(
        embeddings,
        num_concepts=2,
        num_languages=3,
        embedding_dim=2,
        alignment_batch_size=1,
    ).compute()

    en_targets = [
        row for row in result["language_pairs"]
        if row["source_language"] == "en"
    ]
    subset_score = (
        sum(row["num_success"] for row in en_targets)
        / sum(row["num_pairs"] for row in en_targets)
    )

    assert subset_score == 1.0
    assert result["num_pairs"] == 12


def test_alignment_condition_weak_view_uses_target_language_negatives_only() -> None:
    embeddings = {
        "en": np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        "fr": np.array([
            [0.8, 0.6],
            [0.0, 1.0],
        ]),
        "de": np.array([
            [0.0, 1.0],
            [0.99, 0.1],
        ]),
    }

    strong = AlignmentCondition(
        embeddings,
        num_concepts=2,
        num_languages=3,
        embedding_dim=2,
        negative_view="strong_view",
    ).compute()
    weak = AlignmentCondition(
        embeddings,
        num_concepts=2,
        num_languages=3,
        embedding_dim=2,
        negative_view="weak_view",
    ).compute()

    strong_en_fr = next(
        row for row in strong["language_pairs"]
        if row["source_language"] == "en" and row["target_language"] == "fr"
    )
    weak_en_fr = next(
        row for row in weak["language_pairs"]
        if row["source_language"] == "en" and row["target_language"] == "fr"
    )

    assert strong["negative_view"] == "strong_view"
    assert weak["negative_view"] == "weak_view"
    assert strong_en_fr["num_success"] == 0
    assert weak_en_fr["num_success"] == 2


def test_monolingual_structure_condition_matches_rotated_structure() -> None:
    embeddings = {
        "en": np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-1.0, 0.0],
        ]),
        "fr": np.array([
            [np.sqrt(3.0) / 2.0, 0.5],
            [0.0, 1.0],
            [-np.sqrt(3.0) / 2.0, 0.5],
            [-np.sqrt(3.0) / 2.0, -0.5],
        ]),
    }

    result = MonolingualStructureCondition(
        embeddings,
        num_concepts=4,
        num_languages=2,
        embedding_dim=2,
    ).compute()

    assert np.isclose(result["score"], 1.0)
    assert result["distance"] == "cosine"
    assert result["num_concept_pairs"] == 6
    assert np.isclose(result["language_pairs"][0]["correlation"], 1.0)


def test_monolingual_structure_condition_averages_language_pairs() -> None:
    embeddings = {
        "en": np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]),
        "fr": np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]),
        "de": np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-1.0, 0.0],
        ]),
    }

    result = MonolingualStructureCondition(
        embeddings,
        num_concepts=4,
        num_languages=3,
        embedding_dim=2,
    ).compute()

    correlations = [
        row["correlation"]
        for row in result["language_pairs"]
        if row["correlation"] is not None
    ]
    assert len(result["language_pairs"]) == 3
    assert np.isclose(result["score"], np.mean(correlations))


def test_monolingual_structure_condition_constant_distances_are_undefined() -> None:
    embeddings = {
        "en": np.eye(3),
        "fr": np.eye(3),
    }

    result = MonolingualStructureCondition(
        embeddings,
        num_concepts=3,
        num_languages=2,
        embedding_dim=3,
    ).compute()

    assert result["score"] is None
    assert result["num_valid_language_pairs"] == 0
    assert result["language_pairs"][0]["correlation"] is None
