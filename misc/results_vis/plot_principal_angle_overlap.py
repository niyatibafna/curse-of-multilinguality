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
METRIC = "concept_language_principal_angle_overlap"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot concept-language principal-angle overlap.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    rows = load_results(args.input_dir)
    if not rows:
        raise ValueError(f"No {METRIC} results found in {args.input_dir}")

    output_dir = metric_dir(args.output_dir, METRIC)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "principal_angle_overlap.csv")

    plot_heatmap(rows, output_dir, args.formats, "mean_squared_cosine", vmin=0, vmax=1)
    plot_by_dataset(rows, output_dir, args.formats, "mean_squared_cosine", ylim=(0, 1))
    plot_heatmap(rows, output_dir, args.formats, "max_cosine", vmin=0, vmax=1)
    plot_dimension_heatmaps(rows, output_dir, args.formats)
    plot_dimension_scatter(rows, output_dir, args.formats)

    print(f"Wrote principal-angle overlap plots to {output_dir}")


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob(f"*/*/{METRIC}.json")):
        with path.open() as handle:
            payload = json.load(handle)

        result = payload.get("result")
        if not isinstance(result, dict):
            continue

        cosines = result.get("principal_angle_cosines") or []
        rows.append({
            "model": str(payload.get("model") or path.parents[1].name),
            "dataset": str(payload.get("dataset") or path.parent.name),
            "mean_squared_cosine": float(result["mean_squared_cosine"]),
            "max_cosine": float(result["max_cosine"]),
            "mean_principal_angle_degrees": mean_angle_degrees(cosines),
            "concept_subspace_dim": int(result["concept_subspace_dim"]),
            "language_subspace_dim": int(result["language_subspace_dim"]),
            "concept_energy_explained": float(result["concept_energy_explained"]),
            "language_energy_explained": float(result["language_energy_explained"]),
            "subspace_energy_threshold": float(result["subspace_energy_threshold"]),
            "embedding_dim": int(payload["embedding_dim"]),
            "languages": int(payload["languages"]),
            "num_concepts": int(payload["num_concepts"]),
        })
    return rows


def plot_heatmap(
    rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
    field: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    values = {(row["model"], row["dataset"]): row[field] for row in rows}
    matrix = np.array([
        [values.get((model, dataset), np.nan) for dataset in datasets]
        for model in models
    ])

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(datasets) + 3), max(5, 0.45 * len(models) + 2)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(datasets)), [pretty(dataset) for dataset in datasets])
    ax.set_yticks(np.arange(len(models)), models)
    ax.set_title(pretty(field))
    for row_index, model in enumerate(models):
        for col_index, dataset in enumerate(datasets):
            value = values.get((model, dataset))
            if value is not None:
                ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label=pretty(field))
    fig.tight_layout()
    save_figure(fig, output_dir / f"all_datasets__{field}_heatmap", formats)


def plot_by_dataset(
    rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
    field: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    for dataset, subset in sorted(group_by(rows, "dataset").items()):
        subset = sorted(subset, key=lambda row: row[field], reverse=True)
        fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(subset) + 3), 5))
        ax.bar([row["model"] for row in subset], [row[field] for row in subset], color="#4C78A8")
        ax.set_title(f"{pretty(field)} on {pretty(dataset)}")
        ax.set_xlabel("Model")
        ax.set_ylabel(pretty(field))
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        add_value_labels(ax)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__{field}_by_model", formats)


def plot_dimension_heatmaps(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for field in ("concept_subspace_dim", "language_subspace_dim"):
        plot_heatmap(rows, output_dir, formats, field)


def plot_dimension_scatter(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(5 * len(datasets), 8), 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        ax.scatter(
            [row["concept_subspace_dim"] / row["embedding_dim"] for row in subset],
            [row["language_subspace_dim"] / row["embedding_dim"] for row in subset],
            s=55,
            color="#4C78A8",
        )
        for row in subset:
            ax.annotate(
                row["model"],
                (
                    row["concept_subspace_dim"] / row["embedding_dim"],
                    row["language_subspace_dim"] / row["embedding_dim"],
                ),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
        ax.set_title(pretty(dataset))
        ax.set_xlabel("Concept 90% dim / embedding dim")
        ax.set_ylabel("Language 90% dim / embedding dim")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_figure(fig, output_dir / "subspace_dim_fraction_scatter", formats)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "dataset",
        "mean_squared_cosine",
        "max_cosine",
        "mean_principal_angle_degrees",
        "concept_subspace_dim",
        "language_subspace_dim",
        "concept_energy_explained",
        "language_energy_explained",
        "subspace_energy_threshold",
        "embedding_dim",
        "languages",
        "num_concepts",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], row["model"])))


def mean_angle_degrees(cosines: list[float]) -> float | None:
    if not cosines:
        return None
    values = np.clip(np.asarray(cosines, dtype=float), 0.0, 1.0)
    return float(np.mean(np.degrees(np.arccos(values))))


def add_value_labels(ax: plt.Axes) -> None:
    ymin, ymax = ax.get_ylim()
    padding = 0.01 * (ymax - ymin if ymax > ymin else 1.0)
    for patch in ax.patches:
        value = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + padding,
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
