from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt

from metric_display import display_metric_label, extract_metric_value, sorted_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys_path = str(PROJECT_ROOT)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from src.training.check_eval_outputs import missing_outputs

DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "training_scaling"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots" / "scaling"
OVERVIEW_EXCLUDED_METRICS = {
    "anisotropy",
    "concept_space_dim_growth_by_concept",
    "concept_language_principal_angle_overlap",
    "language_space_growth_by_concepts",
    "nearest_neighbor_overlap_against_monolingual",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--metrics")
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--min_size", type=int)
    parser.add_argument("--eval_stream", choices=["eval-all", "eval-subset-n10"])
    parser.add_argument("--version")
    args = parser.parse_args()
    eval_stream = args.eval_stream or infer_eval_stream(args.input_dir, args.manifest_path)
    version = args.version or infer_version(args.input_dir, args.manifest_path)

    if args.manifest_path and not args.allow_partial:
        missing = missing_outputs(args.manifest_path, args.input_dir)
        if missing:
            preview = "\n".join(
                f"index={row['index']} {row['strategy']} {row['subset']} "
                f"{row['dataset']} {row['metric']}"
                for row in missing[:20]
            )
            raise RuntimeError(f"Refusing to plot incomplete eval outputs: {len(missing)} missing.\n{preview}")

    rows = load_rows(args.input_dir, eval_stream)
    if args.min_size is not None:
        rows = [row for row in rows if row["size"] >= args.min_size]
    if args.metrics:
        requested_metrics = {item.strip() for item in args.metrics.split(",") if item.strip()}
        rows = [row for row in rows if row["metric"] in requested_metrics]
    if not rows:
        raise ValueError(f"No training-scaling metric outputs found in {args.input_dir}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = aggregate_rows(rows)
    summary_rows = [row for row in summary_rows if row["metric"] not in OVERVIEW_EXCLUDED_METRICS]
    write_csv(rows, args.output_dir / "training_scaling_raw.csv", raw=True)
    write_csv(summary_rows, args.output_dir / "training_scaling_summary.csv")
    for strategy in sorted({row["strategy"] for row in summary_rows}):
        strategy_rows = [row for row in summary_rows if row["strategy"] == strategy]
        strategy_dir = args.output_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        write_csv(strategy_rows, strategy_dir / "summary.csv")
    plot_rows(summary_rows, args.output_dir, args.formats, version)
    print(args.output_dir)


def load_rows(input_dir: Path, eval_stream: str = "eval-all") -> list[dict[str, Any]]:
    rows = []
    desired_mlm_metric = mlm_metric_for_stream(eval_stream)
    for path in sorted(input_dir.glob("**/*.json")):
        if is_shadowed_legacy_mlm_loss(path, desired_mlm_metric):
            continue
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
        values.append((display_metric_name(payload), value))
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
    return extract_metric_value({"metric": metric, "result": result})


def display_metric_name(payload: dict[str, Any]) -> str:
    if payload.get("metric") in {"mlm_loss", "mlm_loss_all", "mlm_loss_fixed_subset"}:
        return "mlm_loss"
    return payload["metric"]


def is_shadowed_legacy_mlm_loss(path: Path, desired_mlm_metric: str) -> bool:
    if path.name == "mlm_loss.json":
        return path.with_name("mlm_loss_all.json").exists() or path.with_name("mlm_loss_fixed_subset.json").exists()
    if path.name in {"mlm_loss_all.json", "mlm_loss_fixed_subset.json"}:
        return path.stem != desired_mlm_metric
    return False


def mlm_metric_for_stream(eval_stream: str) -> str:
    if eval_stream == "eval-subset-n10":
        return "mlm_loss_fixed_subset"
    return "mlm_loss_all"


def infer_eval_stream(input_dir: Path, manifest_path: Path | None) -> str:
    if "eval-subset-n10" in input_dir.name:
        return "eval-subset-n10"
    if manifest_path and manifest_path.exists():
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        stream = manifest.get("eval_stream")
        if stream in {"eval-all", "eval-subset-n10"}:
            return stream
    return "eval-all"


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


def plot_rows(rows: list[dict[str, Any]], output_dir: Path, formats: list[str], version: str | None) -> None:
    metrics = sorted_metrics({row["metric"] for row in rows})
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
        plot_strategy_overview(strategy_rows, strategy, metrics, datasets, sizes, colors, strategy_dir / "overview", formats, version)


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
    style_axis(ax, display_metric_label(metric), display_metric_label(metric), sizes)
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
    style_axis(ax, f"{display_metric_label(metric)} on {pretty(dataset)}", display_metric_label(metric), sizes)
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
    version: str | None,
) -> None:
    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows + 0.6), squeeze=False)
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
        style_axis(ax, display_metric_label(metric), "", sizes)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(strategy_title(strategy, version), y=0.992)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(datasets),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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


def strategy_title(strategy: str, version: str | None) -> str:
    if "/" in strategy:
        strategy = strategy.split("/", 1)[1]

    if strategy.startswith("fixed"):
        regime = "Fixed"
        budget = token_budget_label(strategy, "total tokens")
    elif strategy.startswith("additive"):
        regime = "Additive"
        budget = token_budget_label(strategy, "max tokens/language")
    else:
        regime = pretty(strategy)
        budget = None

    balance = "imbalanced" if "resource" in strategy or "imbalanced" in strategy else "balanced"
    title = f"{regime} - {balance}"
    if budget:
        title = f"{title}, {budget}"
    if version:
        title = f"{title} ({version})"
    return title


def token_budget_label(strategy: str, suffix: str) -> str | None:
    match = re.search(r"(?:max)?(\d+)m", strategy)
    if not match:
        return None
    return f"{int(match.group(1))}M {suffix}"


def infer_version(input_dir: Path, manifest_path: Path | None) -> str | None:
    for value in [input_dir.name, manifest_path.stem if manifest_path else ""]:
        match = re.search(r"(v\d+)", value)
        if match:
            return match.group(1)
    return None


if __name__ == "__main__":
    main()
