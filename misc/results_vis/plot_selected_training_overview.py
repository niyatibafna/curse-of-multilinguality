from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt

from plot_training_scaling import aggregate_rows, load_rows, pretty, strategy_title
from metric_display import display_metric_label, sorted_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "misc" / "results_vis" / "plots"

SELECTED_METRICS = [
    "nearest_neighbor_overlap_against_monolingual_20",
    "mlm_loss",
    "alignment_condition",
    "alignment_condition_weak_view",
    "noncollapse",
    "language_space_dim_growth_by_language",
    "concept_language_principal_angle_overlap_20",
]

DATASET_COLORS = {
    "bouquet": "#4C78A8",
    "floresplus": "#F58518",
    "wmt24pp": "#54A24B",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=["v7", "v8"])
    parser.add_argument("--eval_stream", default="eval-all", choices=["eval-all", "eval-subset-n10"])
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    stream_suffix = "" if args.eval_stream == "eval-all" else f"_{args.eval_stream}"
    for version in args.versions:
        input_dir = PROJECT_ROOT / "outputs" / f"training_scaling_{version}{stream_suffix}"
        output_dir = args.output_root / f"scaling_{version}{stream_suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = aggregate_rows(load_rows(input_dir, args.eval_stream))
        selected_metric_set = {
            item[0] if isinstance(item, tuple) else item
            for item in SELECTED_METRICS
        }
        selected_rows = [
            row for row in rows
            if row["metric"] in selected_metric_set
        ]
        if not selected_rows:
            raise ValueError(f"No selected metric rows found under {input_dir}.")
        write_csv(selected_rows, output_dir / "selected_overview.csv")
        plot_selected_overview(selected_rows, version, args.eval_stream, output_dir, args.formats)
        print(output_dir / "selected_overview.png")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "strategy",
        "subset",
        "size",
        "dataset",
        "metric",
        "mean",
        "std",
        "stderr",
        "num_seeds",
        "seeds",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_selected_overview(
    rows: list[dict[str, Any]],
    version: str,
    eval_stream: str,
    output_dir: Path,
    formats: list[str],
) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    strategies = sorted({row["strategy"] for row in rows})
    sizes = sorted({row["size"] for row in rows})
    strategy_label = ", ".join(strategy_title(strategy, version) for strategy in strategies)
    stream_label = "eval-subset-n10" if eval_stream == "eval-subset-n10" else "eval-all"

    available_metrics = sorted_metrics({row["metric"] for row in rows})
    ncols = 3
    nrows = (len(available_metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.6 * nrows), squeeze=False)
    for ax, metric in zip(axes.ravel(), available_metrics):
        metric_rows = [row for row in rows if row["metric"] == metric]
        for dataset in datasets:
            dataset_rows = sorted(
                [row for row in metric_rows if row["dataset"] == dataset],
                key=lambda row: row["size"],
            )
            if not dataset_rows:
                continue
            ax.errorbar(
                [row["size"] for row in dataset_rows],
                [row["mean"] for row in dataset_rows],
                yerr=[row.get("std", 0.0) for row in dataset_rows],
                marker="o",
                linewidth=1.9,
                markersize=4,
                capsize=2.5,
                color=DATASET_COLORS.get(dataset),
                label=pretty(dataset),
            )
        ax.set_title(display_metric_label(metric))
        ax.set_xlabel("Training language group size")
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes.ravel()[len(available_metrics):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle(f"Selected Overview - {strategy_label} [{stream_label}]", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for fmt in formats:
        fig.savefig(output_dir / f"selected_overview.{fmt}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
