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
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results_plots"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metric results from outputs/.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    rows = load_results(args.input_dir)
    if not rows:
        raise ValueError(f"No scalar metric results found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(rows, args.output_dir / "results_summary.csv")
    plot_by_dataset_metric(rows, args.output_dir, args.formats)
    plot_by_model_metric(rows, args.output_dir, args.formats)
    plot_all_metrics(rows, args.output_dir, args.formats)
    print(f"Wrote plots to {args.output_dir}")


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*/*/*.json")):
        with path.open() as handle:
            payload = json.load(handle)

        result = payload.get("result")
        if not isinstance(result, (int, float)):
            continue

        rows.append(
            {
                "model": str(payload.get("model") or path.parents[1].name),
                "dataset": str(payload.get("dataset") or path.parent.name),
                "metric": str(payload.get("metric") or path.stem),
                "value": float(result),
            }
        )
    return rows


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "dataset", "metric", "value"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=row_key))


def plot_by_dataset_metric(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    metrics = sorted({row["metric"] for row in rows})

    for dataset in datasets:
        for metric in metrics:
            subset = sorted(
                [row for row in rows if row["dataset"] == dataset and row["metric"] == metric],
                key=lambda row: row["value"],
                reverse=True,
            )
            if not subset:
                continue
            plot_bars(
                labels=[row["model"] for row in subset],
                values=[row["value"] for row in subset],
                title=f"{pretty(metric)} on {pretty(dataset)}",
                xlabel="Model",
                ylabel=pretty(metric),
                output_base=output_dir / f"{dataset}__{metric}__by_model",
                formats=formats,
            )


def plot_by_model_metric(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    models = sorted({row["model"] for row in rows})
    metrics = sorted({row["metric"] for row in rows})

    for model in models:
        for metric in metrics:
            subset = sorted(
                [row for row in rows if row["model"] == model and row["metric"] == metric],
                key=lambda row: row["dataset"],
            )
            if not subset:
                continue
            plot_bars(
                labels=[row["dataset"] for row in subset],
                values=[row["value"] for row in subset],
                title=f"{pretty(metric)} for {model}",
                xlabel="Dataset",
                ylabel=pretty(metric),
                output_base=output_dir / f"{slugify(model)}__{metric}__by_dataset",
                formats=formats,
            )


def plot_all_metrics(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    metrics = sorted({row["metric"] for row in rows})
    if len(metrics) < 2:
        return

    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    fig, axes = plt.subplots(
        len(datasets),
        len(metrics),
        figsize=(max(5 * len(metrics), 8), max(3.5 * len(datasets), 4)),
        squeeze=False,
    )

    values = {(row["dataset"], row["metric"], row["model"]): row["value"] for row in rows}
    for row_index, dataset in enumerate(datasets):
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index][col_index]
            labels = [model for model in models if (dataset, metric, model) in values]
            y = [values[(dataset, metric, model)] for model in labels]
            ax.bar(labels, y, color="#4C78A8")
            ax.set_title(f"{pretty(dataset)}: {pretty(metric)}")
            ax.set_ylabel(pretty(metric))
            ax.tick_params(axis="x", labelrotation=45)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
            add_value_labels(ax)

    fig.tight_layout()
    save_figure(fig, output_dir / "all_datasets__all_metrics", formats)


def plot_bars(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_base: Path,
    formats: list[str],
) -> None:
    width = max(7, min(14, 0.65 * len(labels) + 3))
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    add_value_labels(ax)
    fig.tight_layout()
    save_figure(fig, output_base, formats)


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
    ax.set_ylim(top=max(ymax, max((patch.get_height() for patch in ax.patches), default=0) + 4 * padding))


def save_figure(fig: plt.Figure, output_base: Path, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["dataset"], row["metric"], row["model"]


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    main()
