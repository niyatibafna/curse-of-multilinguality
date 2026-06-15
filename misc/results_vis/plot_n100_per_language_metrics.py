from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots" / "preliminary_n100_per_language_metrics"

RUNS = {
    "v7": PROJECT_ROOT / "outputs" / "training_scaling_v7",
    "v8": PROJECT_ROOT / "outputs" / "training_scaling_v8",
}

STRATEGIES = {
    "v7": "fixed_500m_resource",
    "v8": "fixed_500m_balanced",
}

METRICS = {
    "individual_concept_dimensionality": ("effective_dim", "Concept effective dim"),
    "rmse_against_monolingual": ("rmse", "RMSE vs monolingual"),
    "nearest_neighbor_overlap_against_monolingual": ("mean_overlap", "NN overlap"),
    "nearest_neighbor_overlap_against_monolingual_5": ("mean_overlap", "NN overlap, k=5"),
    "nearest_neighbor_overlap_against_monolingual_10": ("mean_overlap", "NN overlap, k=10"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bouquet")
    parser.add_argument("--subset", default="n100")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = load_rows(args.dataset, args.subset)
    if not rows:
        raise ValueError(f"No per-language rows found for {args.subset}/{args.dataset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(rows)
    language_order = load_language_order(args.dataset, args.subset, rows)

    raw_path = args.output_dir / f"{args.subset}_{args.dataset}_per_language_raw.csv"
    summary_path = args.output_dir / f"{args.subset}_{args.dataset}_per_language_summary.csv"
    write_csv(rows, raw_path, ["version", "seed", "dataset", "subset", "metric", "language", "value", "path"])
    write_csv(
        summary_rows,
        summary_path,
        ["version", "dataset", "subset", "metric", "language", "mean", "std", "stderr", "num_seeds", "seeds"],
    )

    png_path = args.output_dir / f"{args.subset}_{args.dataset}_per_language_metrics.png"
    plot(summary_rows, language_order, args.dataset, args.subset, png_path)
    print(png_path)


def load_rows(dataset: str, subset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for version, root in RUNS.items():
        strategy = STRATEGIES[version]
        for seed_dir in sorted(root.glob("seed*")):
            metric_dir = seed_dir / strategy / subset / dataset
            if not metric_dir.exists():
                continue
            for metric, (field, _) in METRICS.items():
                path = metric_dir / f"{metric}.json"
                if not path.exists():
                    continue
                with path.open() as handle:
                    payload = json.load(handle)
                for language, value in extract_language_values(payload, metric, field).items():
                    rows.append({
                        "version": version,
                        "seed": seed_dir.name,
                        "dataset": dataset,
                        "subset": subset,
                        "metric": metric,
                        "language": language,
                        "value": value,
                        "path": str(path),
                    })
    return rows


def extract_language_values(payload: dict[str, Any], metric: str, field: str) -> dict[str, float]:
    result = payload.get("result")
    if not isinstance(result, dict):
        return {}
    if metric == "individual_concept_dimensionality":
        values = result.get("effective_dim_by_language")
        if isinstance(values, dict):
            return {str(language): float(value) for language, value in values.items() if is_number(value)}
        return {
            str(row["language"]): float(row[field])
            for row in result.get("sorted_effective_dims", [])
            if is_number(row.get(field))
        }
    return {
        str(row["language"]): float(row[field])
        for row in result.get("language_comparisons", [])
        if is_number(row.get(field))
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["version"], row["dataset"], row["subset"], row["metric"], row["language"])
        groups[key].append(row)

    summary = []
    for (version, dataset, subset, metric, language), group in sorted(groups.items()):
        values = [float(row["value"]) for row in group]
        std = stdev(values) if len(values) > 1 else 0.0
        seeds = sorted({row["seed"] for row in group})
        summary.append({
            "version": version,
            "dataset": dataset,
            "subset": subset,
            "metric": metric,
            "language": language,
            "mean": mean(values),
            "std": std,
            "stderr": std / math.sqrt(len(values)),
            "num_seeds": len(values),
            "seeds": ",".join(seeds),
        })
    return summary


def load_language_order(dataset: str, subset: str, rows: list[dict[str, Any]]) -> list[str]:
    for version, root in RUNS.items():
        path = root / "seed1" / STRATEGIES[version] / subset / dataset / "rmse_against_monolingual.json"
        if not path.exists():
            continue
        with path.open() as handle:
            payload = json.load(handle)
        languages = payload.get("result", {}).get("languages")
        if isinstance(languages, list) and languages:
            return [str(language) for language in languages]
    return sorted({str(row["language"]) for row in rows})


def plot(rows: list[dict[str, Any]], language_order: list[str], dataset: str, subset: str, path: Path) -> None:
    versions = ["v7", "v8"]
    colors = {"v7": "#4C78A8", "v8": "#F58518"}
    metrics = [metric for metric in METRICS if any(row["metric"] == metric for row in rows)]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(24, 0.28 * len(language_order)), 3.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(language_order))
    width = 0.38
    values = {
        (row["version"], row["metric"], row["language"]): row
        for row in rows
    }

    for ax, metric in zip(axes, metrics):
        _, label = METRICS[metric]
        for offset_index, version in enumerate(versions):
            offset = (offset_index - 0.5) * width
            means = [values.get((version, metric, language), {}).get("mean", np.nan) for language in language_order]
            stderrs = [values.get((version, metric, language), {}).get("stderr", 0.0) for language in language_order]
            ax.bar(x + offset, means, width=width, yerr=stderrs, label=version, color=colors[version], linewidth=0)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", ncols=2, frameon=False)

    axes[0].set_title(f"{subset} {dataset}: per-language metrics averaged across seeds")
    axes[-1].set_xticks(x, language_order, rotation=90, fontsize=7)
    axes[-1].set_xlabel("Language order from n100")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


if __name__ == "__main__":
    main()
