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
DEFAULT_METRIC = "alignment_condition"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AlignmentCondition results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    rows, pair_rows = load_results(args.input_dir, args.metric)
    if not rows:
        raise ValueError(f"No {args.metric} results found in {args.input_dir}")

    output_dir = metric_dir(args.output_dir, args.metric)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "alignment_condition.csv", summary_fieldnames())
    write_csv(pair_rows, output_dir / "alignment_condition_language_pairs.csv", pair_fieldnames())
    plot_heatmap(rows, output_dir, args.formats)
    plot_by_dataset(rows, output_dir, args.formats)
    plot_pair_score_distributions(pair_rows, output_dir, args.formats)
    plot_language_pair_heatmaps(pair_rows, output_dir / "language_pair_heatmaps", args.formats)

    print(f"Wrote AlignmentCondition plots to {output_dir}")


def load_results(input_dir: Path, metric: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    pair_rows = []
    for path in sorted(input_dir.glob(f"*/*/{metric}.json")):
        with path.open() as handle:
            payload = json.load(handle)

        result = payload.get("result")
        if not isinstance(result, dict) or "score" not in result:
            continue

        model = str(payload.get("model") or path.parents[1].name)
        dataset = str(payload.get("dataset") or path.parent.name)
        num_pairs = int(result["num_pairs"])
        num_success = int(result["num_success"])
        row = {
            "model": model,
            "dataset": dataset,
            "score": float(result["score"]),
            "num_success": num_success,
            "num_pairs": num_pairs,
            "failure_rate": 1.0 - float(result["score"]),
            "similarity": str(result.get("similarity", "")),
            "negative_view": str(result.get("negative_view", "strong_view")),
            "strict": bool(result.get("strict", True)),
            "num_languages": int(result.get("num_languages", payload.get("languages", 0))),
            "num_concepts": int(result.get("num_concepts", payload.get("num_concepts", 0))),
            "embedding_dim": int(payload.get("embedding_dim", 0)),
        }
        rows.append(row)

        for pair in result.get("language_pairs", []):
            pair_rows.append({
                "model": model,
                "dataset": dataset,
                "source_language": str(pair["source_language"]),
                "target_language": str(pair["target_language"]),
                "score": float(pair["score"]),
                "num_success": int(pair["num_success"]),
                "num_pairs": int(pair["num_pairs"]),
            })
    return rows, pair_rows


def plot_heatmap(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    values = {(row["model"], row["dataset"]): row["score"] for row in rows}
    matrix = np.array([
        [values.get((model, dataset), np.nan) for dataset in datasets]
        for model in models
    ])

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(datasets) + 3), max(5, 0.45 * len(models) + 2)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(datasets)), [pretty(dataset) for dataset in datasets])
    ax.set_yticks(np.arange(len(models)), models)
    ax.set_title("Alignment Condition")
    for row_index, model in enumerate(models):
        for col_index, dataset in enumerate(datasets):
            value = values.get((model, dataset))
            if value is not None:
                ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Score")
    fig.tight_layout()
    save_figure(fig, output_dir / "all_datasets__score_heatmap", formats)


def plot_by_dataset(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for dataset, subset in sorted(group_by(rows, "dataset").items()):
        subset = sorted(subset, key=lambda row: row["score"], reverse=True)
        fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(subset) + 3), 5))
        ax.bar([row["model"] for row in subset], [row["score"] for row in subset], color="#4C78A8")
        ax.set_title(f"Alignment Condition on {pretty(dataset)}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        add_value_labels(ax)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__score_by_model", formats)


def plot_pair_score_distributions(
    pair_rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    if not pair_rows:
        return

    for dataset, subset in sorted(group_by(pair_rows, "dataset").items()):
        models = sorted({row["model"] for row in subset})
        values = [[row["score"] for row in subset if row["model"] == model] for model in models]
        fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(models) + 3), 5))
        ax.boxplot(values, tick_labels=models, showfliers=False)
        ax.set_title(f"Language-Pair Alignment Scores on {pretty(dataset)}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Language-Pair Score")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__language_pair_score_distribution", formats)


def plot_language_pair_heatmaps(
    pair_rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    if not pair_rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, model), subset in sorted(group_by(pair_rows, "dataset", "model").items()):
        languages = sorted(
            {
                row["source_language"]
                for row in subset
            }
            | {
                row["target_language"]
                for row in subset
            }
        )
        index = {language: i for i, language in enumerate(languages)}
        matrix = np.full((len(languages), len(languages)), np.nan)
        for row in subset:
            source = index[row["source_language"]]
            target = index[row["target_language"]]
            matrix[source, target] = row["score"]

        size = max(7, min(18, 0.08 * len(languages) + 5))
        fig, ax = plt.subplots(figsize=(size, size))
        image = ax.imshow(matrix, aspect="equal", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{model} on {pretty(dataset)}")
        ax.set_xlabel("Target language")
        ax.set_ylabel("Source language")
        set_language_ticks(ax, languages)
        fig.colorbar(image, ax=ax, label="AlignmentCondition score", fraction=0.046, pad=0.04)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__{slugify(model)}__language_pair_heatmap", formats)


def set_language_ticks(ax: plt.Axes, languages: list[str]) -> None:
    if len(languages) <= 60:
        positions = np.arange(len(languages))
        ax.set_xticks(positions, languages)
        ax.set_yticks(positions, languages)
        ax.tick_params(axis="x", labelrotation=90, labelsize=5)
        ax.tick_params(axis="y", labelsize=5)
        return

    step = max(1, int(np.ceil(len(languages) / 30)))
    positions = np.arange(0, len(languages), step)
    ax.set_xticks(positions, [languages[i] for i in positions])
    ax.set_yticks(positions, [languages[i] for i in positions])
    ax.tick_params(axis="x", labelrotation=90, labelsize=4)
    ax.tick_params(axis="y", labelsize=4)


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
        "failure_rate",
        "num_success",
        "num_pairs",
        "similarity",
        "negative_view",
        "strict",
        "num_languages",
        "num_concepts",
        "embedding_dim",
    ]


def pair_fieldnames() -> list[str]:
    return [
        "model",
        "dataset",
        "source_language",
        "target_language",
        "score",
        "num_success",
        "num_pairs",
    ]


def add_value_labels(ax: plt.Axes) -> None:
    for patch in ax.patches:
        value = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
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


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    main()
