from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import parse_sizes, read_json, write_json


def main(
    base_manifest_path: str,
    output_path: str,
    sizes: str,
    eval_language_source_subset: str | None = None,
    eval_language_mode: str = "source_subset",
    datasets: str | None = None,
) -> None:
    base = read_json(base_manifest_path)
    requested_sizes = parse_sizes(sizes)
    source_subset = eval_language_source_subset or f"n{min(requested_sizes)}"
    if eval_language_mode not in {"source_subset", "train_subset"}:
        raise ValueError("--eval_language_mode must be source_subset or train_subset.")
    requested_datasets = split_csv(datasets) if datasets else None

    by_combo: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in base["entries"]:
        if requested_datasets is not None and row["dataset"] not in requested_datasets:
            continue
        key = (row["strategy"], row["subset"], row["dataset"])
        by_combo.setdefault(key, row)

    entries = []
    skipped = []
    for strategy, dataset, source_row in source_rows(by_combo, source_subset, eval_language_mode):
        for size in requested_sizes:
            target_subset = f"n{size}"
            target = by_combo.get((strategy, target_subset, dataset))
            eval_languages = (
                target["eval_languages"]
                if eval_language_mode == "train_subset" and target
                else source_row["eval_languages"]
            )
            row = {
                "strategy": strategy,
                "size": size,
                "subset": target_subset,
                "dataset": dataset,
                "dataset_split": source_row.get("dataset_split", "dev"),
                "metric": "mlm_loss",
                "eval_stream": source_row.get("eval_stream"),
                "eval_language_mode": eval_language_mode,
                "eval_language_source_subset": source_subset,
                "eval_languages": eval_languages,
                "source_eval_languages": source_row["eval_languages"],
                "train_languages": target.get("train_languages") if target else None,
                "checkpoint_path": target.get("checkpoint_path") if target else None,
            }
            if target is None:
                skipped.append({**row, "reason": "missing_target_subset"})
                continue
            entries.append(row)

    if not entries:
        raise ValueError(
            f"No MLM-loss entries built from {base_manifest_path} "
            f"for sizes={requested_sizes} source_subset={source_subset}."
        )

    write_json(
        output_path,
        {
            "entries": entries,
            "skipped": skipped,
            "metrics": ["mlm_loss"],
            "sizes": requested_sizes,
            "eval_language_mode": eval_language_mode,
            "eval_language_source_subset": source_subset,
            "datasets": sorted({row["dataset"] for row in entries}),
            "source_manifest": str(base_manifest_path),
        },
    )
    print(output_path)
    print(f"entries={len(entries)} skipped={len(skipped)}")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def source_rows(
    by_combo: dict[tuple[str, str, str], dict[str, Any]],
    source_subset: str,
    eval_language_mode: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    rows = []
    if eval_language_mode == "source_subset":
        for (strategy, subset, dataset), row in sorted(by_combo.items()):
            if subset == source_subset:
                rows.append((strategy, dataset, row))
        return rows

    seen = set()
    for strategy, _, dataset in sorted(by_combo):
        key = (strategy, dataset)
        if key in seen:
            continue
        seen.add(key)
        source = by_combo.get((strategy, source_subset, dataset))
        if source is None:
            source = by_combo[(strategy, sorted_subsets(by_combo, strategy, dataset)[0], dataset)]
        rows.append((strategy, dataset, source))
    return rows


def sorted_subsets(
    by_combo: dict[tuple[str, str, str], dict[str, Any]],
    strategy: str,
    dataset: str,
) -> list[str]:
    return sorted(
        [
            subset
            for row_strategy, subset, row_dataset in by_combo
            if row_strategy == strategy and row_dataset == dataset
        ],
        key=lambda subset: int(subset.removeprefix("n")),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sizes", required=True)
    parser.add_argument("--eval_language_source_subset")
    parser.add_argument("--eval_language_mode", default="source_subset")
    parser.add_argument("--datasets")
    main(**vars(parser.parse_args()))
