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
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "concept_space_plots"
INDIVIDUAL_METRIC = "individual_concept_dimensionality"
LANGUAGE_GROWTH_METRIC = "concept_space_dim_growth_by_language"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot concept-space dimensionality results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    individual_rows, growth_rows, order_rows = load_results(args.input_dir)
    if not individual_rows and not growth_rows:
        raise ValueError(f"No concept dimensionality results found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if individual_rows:
        write_csv(individual_rows, args.output_dir / "individual_language_dims.csv")
        plot_individual_language_dims(individual_rows, args.output_dir, args.formats)
        plot_language_dim_summary(individual_rows, args.output_dir, args.formats)
    if growth_rows:
        write_csv(growth_rows, args.output_dir / "concept_dim_growth_by_language.csv")
        write_csv(order_rows, args.output_dir / "concept_dim_growth_language_orders.csv")
        plot_growth_by_language(growth_rows, args.output_dir, args.formats)

    print(f"Wrote concept-space plots to {args.output_dir}")


def load_results(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    individual_rows = []
    growth_rows = []
    order_rows = []

    for path in sorted(input_dir.glob("*/*/*.json")):
        with path.open() as handle:
            payload = json.load(handle)

        model = str(payload.get("model") or path.parents[1].name)
        dataset = str(payload.get("dataset") or path.parent.name)
        metric = str(payload.get("metric") or path.stem)
        result = payload.get("result")
        if not isinstance(result, dict):
            continue

        if metric == INDIVIDUAL_METRIC:
            for item in result.get("sorted_effective_dims", []):
                individual_rows.append({
                    "model": model,
                    "dataset": dataset,
                    "language": item["language"],
                    "effective_dim": float(item["effective_dim"]),
                })
        elif metric == LANGUAGE_GROWTH_METRIC:
            for item in result.get("concept_space_dim_growth_by_language", []):
                growth_rows.append({
                    "model": model,
                    "dataset": dataset,
                    "num_languages": int(item["num_languages"]),
                    "effective_dim": float(item["effective_dim"]),
                })
            for index, language in enumerate(result.get("language_order", [])):
                order_rows.append({
                    "model": model,
                    "dataset": dataset,
                    "order_index": index,
                    "language": language,
                })

    return individual_rows, growth_rows, order_rows


def plot_individual_language_dims(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for (dataset, model), subset in sorted(group_by(rows, "dataset", "model").items()):
        subset = sorted(subset, key=lambda row: row["effective_dim"], reverse=True)
        width = max(8, min(22, 0.18 * len(subset) + 4))
        fig, ax = plt.subplots(figsize=(width, 5))
        ax.bar(
            [row["language"] for row in subset],
            [row["effective_dim"] for row in subset],
            color="#4C78A8",
        )
        ax.set_title(f"{model} on {pretty(dataset)}")
        ax.set_xlabel("Language")
        ax.set_ylabel("Effective dimensionality")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / f"{dataset}__{slugify(model)}__individual_language_dims",
            formats,
        )


def plot_language_dim_summary(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    summary = []
    for (dataset, model), subset in sorted(group_by(rows, "dataset", "model").items()):
        values = [row["effective_dim"] for row in subset]
        summary.append({
            "dataset": dataset,
            "model": model,
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        })

    for dataset, subset in sorted(group_by(summary, "dataset").items()):
        subset = sorted(subset, key=lambda row: row["mean"], reverse=True)
        fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(subset) + 3), 5))
        labels = [row["model"] for row in subset]
        means = [row["mean"] for row in subset]
        lower = [row["mean"] - row["min"] for row in subset]
        upper = [row["max"] - row["mean"] for row in subset]
        ax.bar(labels, means, color="#59A14F")
        ax.errorbar(labels, means, yerr=[lower, upper], fmt="none", color="#222222", capsize=3)
        ax.set_title(f"Per-language dimensionality on {pretty(dataset)}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Mean effective dimensionality")
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__individual_language_dim_summary", formats)


def plot_growth_by_language(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for dataset, subset in sorted(group_by(rows, "dataset").items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, model_rows in sorted(group_by(subset, "model").items()):
            model_rows = sorted(model_rows, key=lambda row: row["num_languages"])
            ax.plot(
                [row["num_languages"] for row in model_rows],
                [row["effective_dim"] for row in model_rows],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=model,
            )
        ax.set_title(f"Concept-space dimensionality by languages on {pretty(dataset)}")
        ax.set_xlabel("Number of languages")
        ax.set_ylabel("Effective dimensionality")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__concept_dim_growth_by_language", formats)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
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
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    main()
