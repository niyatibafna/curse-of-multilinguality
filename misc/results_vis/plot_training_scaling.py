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
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "training_scaling"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots" / "scaling"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--metrics")
    args = parser.parse_args()

    rows = load_rows(args.input_dir)
    if args.metrics:
        requested_metrics = {item.strip() for item in args.metrics.split(",") if item.strip()}
        rows = [row for row in rows if row["metric"] in requested_metrics]
    if not rows:
        raise ValueError(f"No training-scaling metric outputs found in {args.input_dir}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = aggregate_rows(rows)
    write_csv(rows, args.output_dir / "training_scaling_raw.csv", raw=True)
    write_csv(summary_rows, args.output_dir / "training_scaling_summary.csv")
    for strategy in sorted({row["strategy"] for row in summary_rows}):
        strategy_rows = [row for row in summary_rows if row["strategy"] == strategy]
        strategy_dir = args.output_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        write_csv(strategy_rows, strategy_dir / "summary.csv")
    plot_rows(summary_rows, args.output_dir, args.formats)
    print(args.output_dir)


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("**/*.json")):
        rel = path.relative_to(input_dir)
        parts = rel.parts
        if len(parts) == 4:
            seed = ""
            strategy, subset, _, _ = parts
        elif len(parts) == 5:
            seed, strategy, subset, _, _ = parts
        else:
            continue
        if seed.startswith("_") or strategy.startswith("_"):
            continue
        with path.open() as handle:
            payload = json.load(handle)
        for metric_name, value in extract_values(payload):
            rows.append({
                "seed": seed,
                "strategy": strategy,
                "subset": subset,
                "size": int(subset.removeprefix("n")),
                "dataset": payload["dataset"],
                "metric": metric_name,
                "value": value,
                "path": str(path),
            })
    return sorted(rows, key=lambda row: (row["metric"], row["dataset"], row["strategy"], row["size"], row["seed"]))


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import math
    from statistics import mean, stdev

    grouped: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["strategy"], row["subset"], row["size"], row["dataset"], row["metric"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (strategy, subset, size, dataset, metric), group in grouped.items():
        values = [float(row["value"]) for row in group]
        seeds = sorted({row["seed"] or "unseeded" for row in group})
        std = stdev(values) if len(values) > 1 else 0.0
        summary.append({
            "strategy": strategy,
            "subset": subset,
            "size": size,
            "dataset": dataset,
            "metric": metric,
            "value": mean(values),
            "mean": mean(values),
            "std": std,
            "stderr": std / math.sqrt(len(values)) if values else 0.0,
            "num_seeds": len(values),
            "seeds": ",".join(seeds),
            "path": ";".join(row["path"] for row in group),
        })
    return sorted(summary, key=lambda row: (row["metric"], row["dataset"], row["strategy"], row["size"]))


def extract_values(payload: dict[str, Any]) -> list[tuple[str, float]]:
    value = extract_value(payload)
    values = []
    if value is not None:
        values.append((payload["metric"], value))
    if payload["metric"] == "monolingual_structure_condition":
        result = payload.get("result")
        if isinstance(result, dict):
            rmse = result.get("mean_rmse")
            if isinstance(rmse, (int, float)):
                values.append(("monolingual_structure_condition_rmse", float(rmse)))
    return values


def extract_value(payload: dict[str, Any]) -> float | None:
    metric = payload["metric"]
    result = payload["result"]
    if isinstance(result, (int, float)):
        return float(result)
    if not isinstance(result, dict):
        return None

    key_by_metric = {
        "alignment_condition": "score",
        "alignment_condition_weak_view": "score",
        "concept_language_principal_angle_overlap": "adjusted_mean_squared_cosine",
        "concept_language_principal_angle_overlap_20": "adjusted_mean_squared_cosine",
        "concept_language_principal_angle_overlap_50": "adjusted_mean_squared_cosine",
        "concept_language_principal_angle_overlap_90": "adjusted_mean_squared_cosine",
        "eff_langspace_dim_prop": "score",
        "individual_concept_dimensionality": "effective_dim_by_language",
        "monolingual_structure_condition": "score",
        "nearest_neighbor_overlap_against_monolingual": "score",
        "nearest_neighbor_overlap_against_monolingual_5": "score",
        "nearest_neighbor_overlap_against_monolingual_10": "score",
        "rmse_against_monolingual": "score",
    }
    if metric == "concept_space_dim_growth_by_language":
        return final_effective_dim(result.get("concept_space_dim_growth_by_language"))
    if metric == "language_space_dim_growth_by_language":
        return final_effective_dim(result.get("language_subspace_scaling"))
    if metric == "concept_space_dim_growth_by_concept":
        return final_effective_dim(result.get("concept_space_dim_growth_by_concept"))
    if metric == "language_space_growth_by_concepts":
        return final_effective_dim(result.get("language_space_growth_by_concepts"))

    key = key_by_metric.get(metric)
    if key is None:
        return None
    value = result.get(key)
    if isinstance(value, dict):
        values = [item for item in value.values() if isinstance(item, (int, float))]
        return sum(values) / len(values) if values else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def final_effective_dim(rows: Any) -> float | None:
    if not rows:
        return None
    value = rows[-1].get("effective_dim")
    return float(value) if isinstance(value, (int, float)) else None


def write_csv(rows: list[dict[str, Any]], path: Path, raw: bool = False) -> None:
    if not rows:
        return
    fieldnames = (
        ["seed", "strategy", "subset", "size", "dataset", "metric", "value", "path"]
        if raw
        else ["strategy", "subset", "size", "dataset", "metric", "mean", "std", "stderr", "num_seeds", "seeds", "path"]
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    metrics = sorted({row["metric"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})
    strategies = sorted({row["strategy"] for row in rows})
    sizes = sorted({row["size"] for row in rows})
    colors = {"bouquet": "#4C78A8", "floresplus": "#F58518", "wmt24pp": "#54A24B"}

    for strategy in strategies:
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        if not strategy_rows:
            continue
        strategy_dir = output_dir / strategy
        for metric in metrics:
            metric_rows = [row for row in strategy_rows if row["metric"] == metric]
            if not metric_rows:
                continue
            metric_dir = strategy_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)
            plot_metric_overlay(metric_rows, metric, datasets, sizes, colors, metric_dir / "all_datasets", formats)
            for dataset in datasets:
                dataset_rows = [
                    row for row in metric_rows
                    if row["dataset"] == dataset
                ]
                if dataset_rows:
                    plot_metric_dataset(dataset_rows, metric, dataset, sizes, colors[dataset], metric_dir / dataset, formats)
        plot_strategy_overview(strategy_rows, strategy, metrics, datasets, sizes, colors, strategy_dir / "overview", formats)


def plot_metric_overlay(
    rows: list[dict[str, Any]],
    metric: str,
    datasets: list[str],
    sizes: list[int],
    colors: dict[str, str],
    output_base: Path,
    formats: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for dataset in datasets:
        dataset_rows = sorted(
            [row for row in rows if row["dataset"] == dataset],
            key=lambda row: row["size"],
        )
        if not dataset_rows:
            continue
        ax.errorbar(
            [row["size"] for row in dataset_rows],
            [row["mean"] for row in dataset_rows],
            yerr=[row.get("std", 0.0) for row in dataset_rows],
            marker="o",
            linewidth=2,
            capsize=3,
            label=pretty(dataset),
            color=colors[dataset],
        )
    style_axis(ax, pretty(metric), pretty(metric), sizes)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, output_base, formats)


def plot_metric_dataset(
    rows: list[dict[str, Any]],
    metric: str,
    dataset: str,
    sizes: list[int],
    color: str,
    output_base: Path,
    formats: list[str],
) -> None:
    rows = sorted(rows, key=lambda row: row["size"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        [row["size"] for row in rows],
        [row["mean"] for row in rows],
        yerr=[row.get("std", 0.0) for row in rows],
        marker="o",
        linewidth=2,
        capsize=3,
        color=color,
    )
    style_axis(ax, f"{pretty(metric)} on {pretty(dataset)}", pretty(metric), sizes)
    fig.tight_layout()
    save(fig, output_base, formats)


def plot_strategy_overview(
    rows: list[dict[str, Any]],
    strategy: str,
    metrics: list[str],
    datasets: list[str],
    sizes: list[int],
    colors: dict[str, str],
    output_base: Path,
    formats: list[str],
) -> None:
    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
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
                linewidth=1.8,
                markersize=4,
                capsize=2,
                label=pretty(dataset),
                color=colors[dataset],
            )
        style_axis(ax, pretty(metric), "", sizes)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(datasets), frameon=False)
    fig.suptitle(f"{pretty(strategy)} Training Scaling", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, output_base, formats)


def style_axis(ax: Any, title: str, ylabel: str, sizes: list[int]) -> None:
    ax.set_title(title)
    ax.set_xlabel("Training language group size")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: Any, output_base: Path, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=200)
    plt.close(fig)


def pretty(value: str) -> str:
    return value.replace("_", " ").title()


if __name__ == "__main__":
    main()
