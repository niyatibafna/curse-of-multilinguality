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
METRIC = "comness"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot COMness results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    rows = load_results(args.input_dir)
    if not rows:
        raise ValueError(f"No COMness results found in {args.input_dir}")

    output_dir = metric_dir(args.output_dir, METRIC)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "comness.csv")
    plot_score_by_dataset(rows, args.output_dir, args.formats, "raw_comness")
    if any(row["normalized_comness"] is not None for row in rows):
        plot_score_by_dataset(rows, args.output_dir, args.formats, "normalized_comness")
        plot_score_grid(rows, args.output_dir, args.formats)

    print(f"Wrote COMness plots to {output_dir}")


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*/*/*.json")):
        with path.open() as handle:
            payload = json.load(handle)

        metric = str(payload.get("metric") or path.stem)
        if metric != METRIC:
            continue

        result = payload.get("result")
        score = None
        details: dict[str, Any] = {}
        if isinstance(result, (int, float)):
            score = float(result)
        elif (
            isinstance(result, list)
            and len(result) == 2
            and isinstance(result[0], (int, float))
            and isinstance(result[1], dict)
        ):
            score = float(result[0])
            details = result[1]
        else:
            continue

        rows.append({
            "model": str(payload.get("model") or path.parents[1].name),
            "dataset": str(payload.get("dataset") or path.parent.name),
            "raw_comness": score,
            "normalized_comness": optional_float(details.get("normalized_comness")),
            "d_lang": optional_float(details.get("d_lang")),
            "d_concept": optional_float(details.get("d_concept")),
            "d_lang_ratio": optional_float(details.get("d_lang_ratio")),
            "d_concept_ratio": optional_float(details.get("d_concept_ratio")),
            "random_baseline_trials": details.get("random_baseline_trials"),
        })
    return rows


def plot_score_by_dataset(
    rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
    field: str,
) -> None:
    for dataset, subset in sorted(group_by(rows, "dataset").items()):
        subset = sorted(
            [row for row in subset if row[field] is not None],
            key=lambda row: row[field],
            reverse=True,
        )
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(subset) + 3), 5))
        ax.bar([row["model"] for row in subset], [row[field] for row in subset], color="#4C78A8")
        ax.set_title(f"{pretty(field)} on {pretty(dataset)}")
        ax.set_xlabel("Model")
        ax.set_ylabel(pretty(field))
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        add_value_labels(ax)
        fig.tight_layout()
        save_figure(fig, metric_dir(output_dir, METRIC) / f"{dataset}__{field}_by_model", formats)


def plot_score_grid(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    fields = ["raw_comness", "normalized_comness"]
    fig, axes = plt.subplots(
        len(fields),
        len(datasets),
        figsize=(max(5 * len(datasets), 8), 8),
        squeeze=False,
    )
    for row_index, field in enumerate(fields):
        for ax, dataset in zip(axes[row_index], datasets):
            subset = sorted(
                [row for row in rows if row["dataset"] == dataset and row[field] is not None],
                key=lambda row: row[field],
                reverse=True,
            )
            ax.bar([row["model"] for row in subset], [row[field] for row in subset], color="#4C78A8")
            ax.set_title(f"{pretty(dataset)}: {pretty(field)}")
            ax.set_ylabel(pretty(field))
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", labelrotation=45)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
            add_value_labels(ax)
    fig.tight_layout()
    save_figure(fig, metric_dir(output_dir, METRIC) / "all_datasets__raw_vs_normalized", formats)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "dataset",
        "raw_comness",
        "normalized_comness",
        "d_lang",
        "d_concept",
        "d_lang_ratio",
        "d_concept_ratio",
        "random_baseline_trials",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], row["model"])))


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


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    main()
