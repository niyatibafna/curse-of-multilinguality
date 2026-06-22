from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.run_metrics import METRICS
from src.training.common import parse_sizes, read_json, training_dir, write_json


DATASETS = ["bouquet", "floresplus", "wmt24pp"]
STRATEGIES = ["fixed", "additive"]
DEFAULT_METRICS = list(METRICS)
DATASET_SPLITS = {
    "bouquet": "dev",
    "floresplus": "dev",
    "wmt24pp": "dev",
}
MIN_LANGUAGES = {
    "alignment_condition": 2,
    "alignment_condition_weak_view": 2,
    "comness": 2,
    "concept_language_principal_angle_overlap": 2,
    "concept_language_principal_angle_overlap_20": 2,
    "concept_language_principal_angle_overlap_50": 2,
    "concept_language_principal_angle_overlap_90": 2,
    "eff_langspace_dim_prop": 2,
    "language_space_dim_growth_by_language": 2,
    "language_space_growth_by_concepts": 2,
    "monolingual_structure_condition": 2,
    "nearest_neighbor_overlap_against_monolingual": 1,
    "nearest_neighbor_overlap_against_monolingual_5": 1,
    "nearest_neighbor_overlap_against_monolingual_10": 1,
    "nearest_neighbor_overlap_against_monolingual_20": 1,
    "nearest_neighbor_overlap_against_monolingual_50": 1,
    "rmse_against_monolingual": 1,
}


def main(
    language_plan_path: str | None = None,
    output_path: str | None = None,
    checkpoint_root: str | None = None,
    metrics: str | None = None,
    datasets: str = ",".join(DATASETS),
    strategies: str = ",".join(STRATEGIES),
    sizes: str | None = None,
) -> None:
    plan = read_json(language_plan_path or training_dir() / "language_plan.json")
    requested_metrics = split_csv(metrics) if metrics else DEFAULT_METRICS
    requested_datasets = split_csv(datasets)
    requested_strategies = split_csv(strategies)
    requested_sizes = parse_sizes(sizes)
    checkpoints = Path(checkpoint_root) if checkpoint_root else training_dir() / "checkpoints"

    entries = []
    skipped = []
    for strategy in requested_strategies:
        for size in requested_sizes:
            subset = f"n{size}"
            languages = plan["subsets"][subset]
            checkpoint_path = checkpoints / strategy / subset
            for dataset in requested_datasets:
                dataset_split = DATASET_SPLITS.get(dataset, "dev")
                eval_languages = eval_languages_for_dataset(plan, languages, dataset)
                for metric in requested_metrics:
                    min_languages = MIN_LANGUAGES.get(metric, 1)
                    row = {
                        "strategy": strategy,
                        "size": size,
                        "subset": subset,
                        "checkpoint_path": str(checkpoint_path),
                        "dataset": dataset,
                        "dataset_split": dataset_split,
                        "metric": metric,
                        "train_languages": languages,
                        "eval_languages": eval_languages,
                        "min_languages": min_languages,
                    }
                    if len(eval_languages) < min_languages:
                        skipped.append({**row, "reason": "not_enough_eval_languages"})
                        continue
                    entries.append(row)

    output = Path(output_path) if output_path else training_dir() / "eval_manifest.json"
    write_json(
        output,
        {
            "entries": entries,
            "skipped": skipped,
            "metrics": requested_metrics,
            "datasets": requested_datasets,
            "strategies": requested_strategies,
            "sizes": requested_sizes,
        },
    )
    print(output)
    print(f"entries={len(entries)} skipped={len(skipped)}")


def eval_languages_for_dataset(
    plan: dict[str, Any],
    train_languages: list[str],
    dataset: str,
) -> list[str]:
    seen = set()
    languages = []
    coverage = plan.get("eval_coverage", {})
    for train_language in train_languages:
        for eval_language in coverage.get(train_language, {}).get(dataset, []):
            if eval_language not in seen:
                seen.add(eval_language)
                languages.append(eval_language)
    return languages


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--language_plan_path")
    parser.add_argument("--output_path")
    parser.add_argument("--checkpoint_root")
    parser.add_argument("--metrics")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--sizes")
    main(**vars(parser.parse_args()))
