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
BASE_METRIC = "concept_language_principal_angle_overlap"
THRESHOLD_METRICS = {
    "concept_language_principal_angle_overlap_20": 0.2,
    "concept_language_principal_angle_overlap_50": 0.5,
    "concept_language_principal_angle_overlap_90": 0.9,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot principal-angle overlap threshold sweep.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    rows = load_results(args.input_dir)
    if not rows:
        raise ValueError(f"No {BASE_METRIC} sweep results found in {args.input_dir}")

    output_dir = args.output_dir / "principal_angle_overlap_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "principal_angle_overlap_sweep.csv")
    plot_threshold_curves(rows, output_dir, args.formats, "adjusted_mean_squared_cosine")
    plot_threshold_curves(rows, output_dir, args.formats, "mean_squared_cosine")
    plot_threshold_heatmaps(rows, output_dir, args.formats, "adjusted_mean_squared_cosine")
    plot_threshold_heatmaps(rows, output_dir, args.formats, "mean_squared_cosine")
    plot_dim_curves(rows, output_dir, args.formats)
    print(f"Wrote principal-angle overlap sweep plots to {output_dir}")


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for metric, threshold in THRESHOLD_METRICS.items():
        for path in sorted(input_dir.glob(f"*/*/{metric}.json")):
            with path.open() as handle:
                payload = json.load(handle)
            result = payload.get("result")
            if not isinstance(result, dict):
                continue
            concept_dim = int(result["concept_subspace_dim"])
            language_dim = int(result["language_subspace_dim"])
            embedding_dim = int(payload["embedding_dim"])
            random_expected = optional_float(result.get("random_expected_mean_squared_cosine"))
            if random_expected is None:
                random_expected = max(concept_dim, language_dim) / embedding_dim
            adjusted = optional_float(result.get("adjusted_mean_squared_cosine"))
            if adjusted is None and random_expected < 1:
                adjusted = (float(result["mean_squared_cosine"]) - random_expected) / (1 - random_expected)
            rows.append({
                "model": str(payload.get("model") or path.parents[1].name),
                "dataset": str(payload.get("dataset") or path.parent.name),
                "threshold": threshold,
                "metric": metric,
                "mean_squared_cosine": float(result["mean_squared_cosine"]),
                "random_expected_mean_squared_cosine": random_expected,
                "adjusted_mean_squared_cosine": adjusted,
                "max_cosine": float(result["max_cosine"]),
                "concept_subspace_dim": concept_dim,
                "language_subspace_dim": language_dim,
                "embedding_dim": embedding_dim,
                "languages": int(payload["languages"]),
                "num_concepts": int(payload["num_concepts"]),
            })
    return rows


def plot_threshold_curves(
    rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
    field: str,
) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig, axes = plt.subplots(1, len(datasets), figsize=(max(5.5 * len(datasets), 10), 4.8), sharey=True, squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for model, color in zip(models, colors):
            subset = sorted(
                [row for row in rows if row["dataset"] == dataset and row["model"] == model],
                key=lambda row: row["threshold"],
            )
            if not subset:
                continue
            ax.plot(
                [row["threshold"] for row in subset],
                [row[field] for row in subset],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=model,
                color=color,
            )
        ax.set_title(pretty(dataset))
        ax.set_xlabel("Energy threshold")
        ax.set_xticks([0.2, 0.5, 0.9])
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel(pretty(field))
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / f"{field}_by_threshold", formats)


def plot_threshold_heatmaps(
    rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
    field: str,
) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    thresholds = [0.2, 0.5, 0.9]

    for dataset in datasets:
        values = {
            (row["model"], row["threshold"]): row[field]
            for row in rows
            if row["dataset"] == dataset
        }
        matrix = np.array([
            [values.get((model, threshold), np.nan) for threshold in thresholds]
            for model in models
        ])
        fig, ax = plt.subplots(figsize=(6, max(5, 0.45 * len(models) + 1.5)))
        vmin = -1 if field == "adjusted_mean_squared_cosine" else 0
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=1)
        ax.set_xticks(np.arange(len(thresholds)), [str(threshold) for threshold in thresholds])
        ax.set_yticks(np.arange(len(models)), models)
        ax.set_title(f"{pretty(dataset)}")
        ax.set_xlabel("Energy threshold")
        for row_index, model in enumerate(models):
            for col_index, threshold in enumerate(thresholds):
                value = values.get((model, threshold))
                if value is not None:
                    ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, label=pretty(field))
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__{field}_heatmap", formats)


def plot_dim_curves(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(5.5 * len(datasets), 10), 4.5), sharey=True, squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        subset = sorted(
            [row for row in rows if row["dataset"] == dataset],
            key=lambda row: (row["threshold"], row["model"]),
        )
        thresholds = sorted({row["threshold"] for row in subset})
        concept_means = [
            np.mean([row["concept_subspace_dim"] / row["embedding_dim"] for row in subset if row["threshold"] == threshold])
            for threshold in thresholds
        ]
        language_means = [
            np.mean([row["language_subspace_dim"] / row["embedding_dim"] for row in subset if row["threshold"] == threshold])
            for threshold in thresholds
        ]
        ax.plot(thresholds, concept_means, marker="o", label="concept")
        ax.plot(thresholds, language_means, marker="o", label="language")
        ax.set_title(pretty(dataset))
        ax.set_xlabel("Energy threshold")
        ax.set_xticks([0.2, 0.5, 0.9])
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("Mean retained dim / embedding dim")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "mean_retained_dim_fraction_by_threshold", formats)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model",
        "dataset",
        "threshold",
        "metric",
        "mean_squared_cosine",
        "random_expected_mean_squared_cosine",
        "adjusted_mean_squared_cosine",
        "max_cosine",
        "concept_subspace_dim",
        "language_subspace_dim",
        "embedding_dim",
        "languages",
        "num_concepts",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], row["model"], row["threshold"])))


def save_figure(fig: plt.Figure, output_base: Path, formats: list[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    main()
