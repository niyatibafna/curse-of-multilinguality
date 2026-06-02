from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

METRICS = {
    "concept_space_dim_growth_by_concept": (
        "concept_space_dim_growth_by_concept",
        "num_concepts",
        "Concept-space ratio by concepts",
    ),
    "concept_space_dim_growth_by_language": (
        "concept_space_dim_growth_by_language",
        "num_languages",
        "Concept-space ratio by languages",
    ),
    "language_space_dim_growth_by_language": (
        "language_subspace_scaling",
        "num_languages",
        "Language-space ratio by languages",
    ),
    "language_space_growth_by_concepts": (
        "language_space_growth_by_concepts",
        "num_concepts",
        "Language-space ratio by concepts",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scaling effective-dim ratios.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    rows = load_rows(args.input_dir)
    if not rows:
        raise ValueError(f"No scaling effective_dim_ratio results found in {args.input_dir}")

    output_dir = args.output_dir / "effective_dim_ratio"
    write_csv(rows, output_dir / "scaling_effective_dim_ratio.csv")
    for metric in sorted({row["metric"] for row in rows}):
        metric_rows = [row for row in rows if row["metric"] == metric]
        plot_by_dataset(metric, metric_rows, output_dir, args.formats)
        plot_dataset_grid(metric, metric_rows, output_dir, args.formats)
        plot_by_model(metric, metric_rows, output_dir, args.formats)

    print(f"Wrote scaling effective-dim ratio plots to {output_dir}")


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*/*/*.json")):
        with path.open() as handle:
            payload = json.load(handle)

        metric = str(payload.get("metric") or path.stem)
        if metric not in METRICS:
            continue

        result_key, x_key, _ = METRICS[metric]
        result = payload.get("result")
        if not isinstance(result, dict):
            continue

        model = str(payload.get("model") or path.parents[1].name)
        dataset = str(payload.get("dataset") or path.parent.name)
        for item in result.get(result_key, []):
            ratio = item.get("effective_dim_ratio")
            if ratio is None:
                continue
            rows.append({
                "metric": metric,
                "model": model,
                "dataset": dataset,
                "x_name": x_key,
                "x_value": int(item[x_key]),
                "effective_dim": float(item["effective_dim"]),
                "random_effective_dim_mean": float(item["random_effective_dim_mean"]),
                "random_effective_dim_std": float(item["random_effective_dim_std"]),
                "effective_dim_ratio": float(ratio),
            })
    return rows


def plot_by_dataset(
    metric: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    _, x_key, title = METRICS[metric]
    for dataset, subset in sorted(group_by(rows, "dataset").items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        add_ratio_baseline(ax)
        for model, model_rows in sorted(group_by(subset, "model").items()):
            plot_model_series(ax, model, model_rows)
        ax.set_title(f"{title} on {pretty(dataset)}")
        ax.set_xlabel(pretty(x_key))
        ax.set_ylabel("Effective dim / random baseline")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, output_dir / metric / f"{dataset}__effective_dim_ratio", formats)


def plot_dataset_grid(
    metric: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    _, x_key, title = METRICS[metric]
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(5 * len(datasets), 8), 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        add_ratio_baseline(ax)
        subset = [row for row in rows if row["dataset"] == dataset]
        for model, model_rows in sorted(group_by(subset, "model").items()):
            plot_model_series(ax, model, model_rows)
        ax.set_title(pretty(dataset))
        ax.set_xlabel(pretty(x_key))
        ax.set_ylabel("Eff dim / random")
    axes[0][-1].legend(fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_dir / metric / "all_datasets__effective_dim_ratio", formats)


def plot_by_model(
    metric: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    _, x_key, title = METRICS[metric]
    for model, subset in sorted(group_by(rows, "model").items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        add_ratio_baseline(ax)
        for dataset, dataset_rows in sorted(group_by(subset, "dataset").items()):
            plot_model_series(ax, dataset, dataset_rows)
        ax.set_title(f"{title} for {model}")
        ax.set_xlabel(pretty(x_key))
        ax.set_ylabel("Effective dim / random baseline")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, output_dir / metric / "detailed_plots" / f"{slugify(model)}__effective_dim_ratio", formats)


def plot_model_series(ax: plt.Axes, label: str, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: row["x_value"])
    ax.plot(
        [row["x_value"] for row in rows],
        [row["effective_dim_ratio"] for row in rows],
        marker="o",
        linewidth=1.5,
        markersize=3,
        label=label,
    )


def add_ratio_baseline(ax: plt.Axes) -> None:
    ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.7)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_by(rows: list[dict[str, Any]], *keys: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[item] for item in keys)
        if len(key) == 1:
            key = key[0]
        groups.setdefault(key, []).append(row)
    return groups


def save_figure(fig: plt.Figure, output_base: Path, formats: list[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    main()
