from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
METRIC = "monolingual_structure_condition"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MonolingualStructureCondition results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    rows, pair_rows = load_results(args.input_dir)
    if not rows:
        raise ValueError(f"No {METRIC} results found in {args.input_dir}")

    output_dir = metric_dir(args.output_dir, METRIC)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "monolingual_structure_condition.csv", summary_fieldnames())
    write_csv(
        pair_rows,
        output_dir / "monolingual_structure_condition_language_pairs.csv",
        pair_fieldnames(),
    )
    plot_heatmap(rows, output_dir, args.formats)
    plot_by_dataset(rows, output_dir, args.formats)
    plot_pair_distributions(pair_rows, output_dir, args.formats)

    print(f"Wrote MonolingualStructureCondition plots to {output_dir}")


def load_results(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    pair_rows = []
    for path in sorted(input_dir.glob(f"*/*/{METRIC}.json")):
        with path.open() as handle:
            payload = json.load(handle)

        result = payload.get("result")
        if not isinstance(result, dict) or "score" not in result:
            continue

        model = str(payload.get("model") or path.parents[1].name)
        dataset = str(payload.get("dataset") or path.parent.name)
        score = optional_float(result.get("score"))
        row = {
            "model": model,
            "dataset": dataset,
            "score": score,
            "distance": str(result.get("distance", "")),
            "correlation": str(result.get("correlation", "")),
            "num_valid_language_pairs": int(result.get("num_valid_language_pairs", 0)),
            "num_languages": int(result.get("num_languages", payload.get("languages", 0))),
            "num_concepts": int(result.get("num_concepts", payload.get("num_concepts", 0))),
            "num_concept_pairs": int(result.get("num_concept_pairs", 0)),
            "embedding_dim": int(payload.get("embedding_dim", 0)),
            **summary_measures(result),
        }
        rows.append(row)

        for pair in result.get("language_pairs", []):
            pair_rows.append({
                "model": model,
                "dataset": dataset,
                "language_1": str(pair["language_1"]),
                "language_2": str(pair["language_2"]),
                "correlation": optional_float(pair.get("correlation")),
                "num_concept_pairs": int(pair.get("num_concept_pairs", 0)),
                **pair_measures(pair),
            })
    return rows, pair_rows


def plot_heatmap(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    values = {(row["model"], row["dataset"]): row["score"] for row in rows}
    matrix = np.array([
        [nan_if_none(values.get((model, dataset))) for dataset in datasets]
        for model in models
    ])

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(datasets) + 3), max(5, 0.45 * len(models) + 2)))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(datasets)), [pretty(dataset) for dataset in datasets])
    ax.set_yticks(np.arange(len(models)), models)
    ax.set_title("Monolingual Structure Condition")
    for row_index, model in enumerate(models):
        for col_index, dataset in enumerate(datasets):
            value = values.get((model, dataset))
            if value is not None:
                ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Average Correlation")
    fig.tight_layout()
    save_figure(fig, output_dir / "all_datasets__score_heatmap", formats)


def plot_by_dataset(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for dataset, subset in sorted(group_by(rows, "dataset").items()):
        subset = sorted(
            [row for row in subset if row["score"] is not None],
            key=lambda row: row["score"],
            reverse=True,
        )
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(subset) + 3), 5))
        ax.bar([row["model"] for row in subset], [row["score"] for row in subset], color="#4C78A8")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"Monolingual Structure Condition on {pretty(dataset)}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Average Correlation")
        ax.set_ylim(-1, 1)
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        add_value_labels(ax)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__score_by_model", formats)


def plot_pair_distributions(
    pair_rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    valid_rows = [row for row in pair_rows if row["correlation"] is not None]
    if not valid_rows:
        return

    for dataset, subset in sorted(group_by(valid_rows, "dataset").items()):
        models = sorted({row["model"] for row in subset})
        values = [[row["correlation"] for row in subset if row["model"] == model] for model in models]
        fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(models) + 3), 5))
        ax.boxplot(values, tick_labels=models, showfliers=False)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"Language-Pair Structure Correlations on {pretty(dataset)}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Language-Pair Correlation")
        ax.set_ylim(-1, 1)
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__language_pair_correlation_distribution", formats)


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: tuple(str(row[field]) for field in fieldnames if field in row)))


def summary_fieldnames() -> list[str]:
    return [
        "model",
        "dataset",
        "score",
        "distance",
        "correlation",
        "num_valid_language_pairs",
        "num_languages",
        "num_concepts",
        "num_concept_pairs",
        "embedding_dim",
        "mean_pearson",
        "mean_spearman",
        "mean_mae",
        "mean_rmse",
        "mean_normalized_rmse",
        "mean_centered_rmse",
        "mean_standardized_rmse",
        "mean_mean_distance_ratio",
        "mean_std_distance_ratio",
    ]


def pair_fieldnames() -> list[str]:
    return [
        "model",
        "dataset",
        "language_1",
        "language_2",
        "correlation",
        "pearson",
        "spearman",
        "mae",
        "rmse",
        "normalized_rmse",
        "centered_rmse",
        "standardized_rmse",
        "mean_distance_1",
        "mean_distance_2",
        "std_distance_1",
        "std_distance_2",
        "mean_distance_ratio",
        "std_distance_ratio",
        "num_concept_pairs",
    ]


def summary_measures(result: dict[str, Any]) -> dict[str, float | None]:
    return {
        "mean_pearson": optional_float(result.get("mean_pearson", result.get("score"))),
        "mean_spearman": optional_float(result.get("mean_spearman")),
        "mean_mae": optional_float(result.get("mean_mae")),
        "mean_rmse": optional_float(result.get("mean_rmse")),
        "mean_normalized_rmse": optional_float(result.get("mean_normalized_rmse")),
        "mean_centered_rmse": optional_float(result.get("mean_centered_rmse")),
        "mean_standardized_rmse": optional_float(result.get("mean_standardized_rmse")),
        "mean_mean_distance_ratio": optional_float(result.get("mean_mean_distance_ratio")),
        "mean_std_distance_ratio": optional_float(result.get("mean_std_distance_ratio")),
    }


def pair_measures(pair: dict[str, Any]) -> dict[str, float | None]:
    return {
        "pearson": optional_float(pair.get("pearson", pair.get("correlation"))),
        "spearman": optional_float(pair.get("spearman")),
        "mae": optional_float(pair.get("mae")),
        "rmse": optional_float(pair.get("rmse")),
        "normalized_rmse": optional_float(pair.get("normalized_rmse")),
        "centered_rmse": optional_float(pair.get("centered_rmse")),
        "standardized_rmse": optional_float(pair.get("standardized_rmse")),
        "mean_distance_1": optional_float(pair.get("mean_distance_1")),
        "mean_distance_2": optional_float(pair.get("mean_distance_2")),
        "std_distance_1": optional_float(pair.get("std_distance_1")),
        "std_distance_2": optional_float(pair.get("std_distance_2")),
        "mean_distance_ratio": optional_float(pair.get("mean_distance_ratio")),
        "std_distance_ratio": optional_float(pair.get("std_distance_ratio")),
    }


def add_value_labels(ax: plt.Axes) -> None:
    for patch in ax.patches:
        value = patch.get_height()
        offset = 0.03 if value >= 0 else -0.07
        va = "bottom" if value >= 0 else "top"
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + offset,
            f"{value:.3f}",
            ha="center",
            va=va,
            fontsize=8,
        )


def save_figure(fig: plt.Figure, output_base: Path, formats: list[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def metric_dir(output_dir: Path, metric: str) -> Path:
    return output_dir / slugify(metric)


def group_by(rows: list[dict[str, Any]], *keys: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[key] for key in keys)
        if len(key) == 1:
            key = key[0]
        groups.setdefault(key, []).append(row)
    return groups


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def nan_if_none(value: float | None) -> float:
    if value is None:
        return np.nan
    return value


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    main()
