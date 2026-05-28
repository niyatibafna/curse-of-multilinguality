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
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "language_subspace_plots"
METRIC = "language_space_dim_growth_by_language"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot language-subspace dimensionality scaling.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    scaling_rows, order_rows = load_results(args.input_dir)
    if not scaling_rows:
        raise ValueError(f"No {METRIC} results found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(scaling_rows, args.output_dir / "language_subspace_scaling.csv")
    write_csv(order_rows, args.output_dir / "language_orders.csv")
    plot_by_dataset(scaling_rows, args.output_dir, args.formats)
    plot_by_model(scaling_rows, args.output_dir, args.formats)
    print(f"Wrote language-subspace plots to {args.output_dir}")


def load_results(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scaling_rows = []
    order_rows = []
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

        for row in result.get("language_subspace_scaling", []):
            scaling_rows.append({
                "model": model,
                "dataset": dataset,
                "num_languages": int(row["num_languages"]),
                "effective_dim": float(row["effective_dim"]),
            })
        for index, language in enumerate(result.get("language_order", [])):
            order_rows.append({
                "model": model,
                "dataset": dataset,
                "order_index": index,
                "language": language,
            })

    return scaling_rows, order_rows


def plot_by_dataset(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
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
        ax.set_title(f"Language-subspace dimensionality on {pretty(dataset)}")
        ax.set_xlabel("Number of languages")
        ax.set_ylabel("Effective dimensionality")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__language_subspace_scaling", formats)


def plot_by_model(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for model, subset in sorted(group_by(rows, "model").items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        for dataset, dataset_rows in sorted(group_by(subset, "dataset").items()):
            dataset_rows = sorted(dataset_rows, key=lambda row: row["num_languages"])
            ax.plot(
                [row["num_languages"] for row in dataset_rows],
                [row["effective_dim"] for row in dataset_rows],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=dataset,
            )
        ax.set_title(f"Language-subspace dimensionality for {model}")
        ax.set_xlabel("Number of languages")
        ax.set_ylabel("Effective dimensionality")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{slugify(model)}__language_subspace_scaling", formats)


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
