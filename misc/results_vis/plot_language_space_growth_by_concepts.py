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
METRIC = "language_space_growth_by_concepts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot language-space dimensionality growth over concepts.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    rows = load_results(args.input_dir)
    if not rows:
        raise ValueError(f"No {METRIC} results found in {args.input_dir}")

    output_dir = metric_dir(args.output_dir, METRIC)
    write_csv(rows, output_dir / "language_space_growth_by_concepts.csv")
    plot_by_dataset(rows, args.output_dir, args.formats)
    plot_dataset_grid(rows, args.output_dir, args.formats)
    plot_by_model(rows, args.output_dir, args.formats)
    print(f"Wrote language-space-by-concepts plots to {args.output_dir}")


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*/*/*.json")):
        with path.open() as handle:
            payload = json.load(handle)

        metric = str(payload.get("metric") or path.stem)
        if metric != METRIC:
            continue

        model = str(payload.get("model") or path.parents[1].name)
        dataset = str(payload.get("dataset") or path.parent.name)
        result = payload.get("result")
        if not isinstance(result, dict):
            continue

        for item in result.get("language_space_growth_by_concepts", []):
            rows.append({
                "model": model,
                "dataset": dataset,
                "num_concepts": int(item["num_concepts"]),
                "effective_dim": float(item["effective_dim"]),
            })

    return rows


def plot_by_dataset(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for dataset, subset in sorted(group_by(rows, "dataset").items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, model_rows in sorted(group_by(subset, "model").items()):
            model_rows = sorted(model_rows, key=lambda row: row["num_concepts"])
            ax.plot(
                [row["num_concepts"] for row in model_rows],
                [row["effective_dim"] for row in model_rows],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=model,
            )
        ax.set_title(f"Language-space dimensionality by concepts on {pretty(dataset)}")
        ax.set_xlabel("Number of concepts")
        ax.set_ylabel("Effective dimensionality")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, metric_dir(output_dir, METRIC) / f"{dataset}__language_space_growth_by_concepts", formats)


def plot_dataset_grid(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(5 * len(datasets), 8), 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        for model, model_rows in sorted(group_by(subset, "model").items()):
            model_rows = sorted(model_rows, key=lambda row: row["num_concepts"])
            ax.plot(
                [row["num_concepts"] for row in model_rows],
                [row["effective_dim"] for row in model_rows],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=model,
            )
        ax.set_title(pretty(dataset))
        ax.set_xlabel("Number of concepts")
        ax.set_ylabel("Effective dimensionality")
    axes[0][-1].legend(fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.suptitle("Language-space dimensionality by concepts")
    fig.tight_layout()
    save_figure(fig, metric_dir(output_dir, METRIC) / "all_datasets__language_space_growth_by_concepts", formats)


def plot_by_model(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for model, subset in sorted(group_by(rows, "model").items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        for dataset, dataset_rows in sorted(group_by(subset, "dataset").items()):
            dataset_rows = sorted(dataset_rows, key=lambda row: row["num_concepts"])
            ax.plot(
                [row["num_concepts"] for row in dataset_rows],
                [row["effective_dim"] for row in dataset_rows],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=dataset,
            )
        ax.set_title(f"Language-space dimensionality by concepts for {model}")
        ax.set_xlabel("Number of concepts")
        ax.set_ylabel("Effective dimensionality")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(
            fig,
            metric_dir(output_dir, METRIC) / "detailed_plots" / f"{slugify(model)}__language_space_growth_by_concepts",
            formats,
        )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
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


def metric_dir(output_dir: Path, metric: str) -> Path:
    return output_dir / slugify(metric)


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    main()
