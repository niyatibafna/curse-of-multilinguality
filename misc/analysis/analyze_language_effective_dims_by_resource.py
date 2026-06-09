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
UTILS_DIR = PROJECT_ROOT / "src" / "utils" / "language_sorting"
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "misc"
    / "results_vis"
    / "plots"
    / "individual_concept_dimensionality"
    / "individual_language_dims.csv"
)
DEFAULT_JSON_DIR = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "language_effective_dims_by_resource"
DEFAULT_LANGUAGE_TO_WIKI = UTILS_DIR / "language_to_wiki.csv"
DEFAULT_WIKI_COUNTS = UTILS_DIR / "wiki_counts.csv"
METRIC = "individual_concept_dimensionality"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-language effective dimensionality by reverse Wikipedia resource order."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--language-to-wiki", type=Path, default=DEFAULT_LANGUAGE_TO_WIKI)
    parser.add_argument("--wiki-counts", type=Path, default=DEFAULT_WIKI_COUNTS)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    rows = read_input_rows(args.input_csv) if args.input_csv.exists() else read_json_rows(args.json_dir)
    if not rows:
        raise ValueError(f"No {METRIC} rows found in {args.input_csv} or {args.json_dir}")

    language_to_wiki = read_mapping(args.language_to_wiki)
    wiki_counts = read_counts(args.wiki_counts)
    rows = add_resource_columns(rows, language_to_wiki, wiki_counts)
    mean_rows = average_by_dataset_language(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["dataset"], row["wiki_count"], row["language"], row["model"]))
    mean_rows = sorted(mean_rows, key=lambda row: (row["dataset"], row["wiki_count"], row["language"]))

    write_csv(rows, args.output_dir / "language_effective_dims_by_resource.csv", row_fieldnames())
    write_csv(
        mean_rows,
        args.output_dir / "language_effective_dims_by_resource_dataset_means.csv",
        mean_fieldnames(),
    )
    plot_dataset_means(mean_rows, args.output_dir, args.formats)
    plot_model_details(rows, args.output_dir, args.formats)

    print(f"Wrote language effective-dim analysis to {args.output_dir}")


def read_input_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "model": row["model"],
                "dataset": row["dataset"],
                "language": row["language"],
                "effective_dim": float(row["effective_dim"]),
            }
            for row in reader
        ]


def read_json_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*/*/*.json")):
        with path.open() as handle:
            payload = json.load(handle)
        if str(payload.get("metric") or path.stem) != METRIC:
            continue
        result = payload.get("result")
        if not isinstance(result, dict):
            continue
        model = str(payload.get("model") or path.parents[1].name)
        dataset = str(payload.get("dataset") or path.parent.name)
        for item in result.get("sorted_effective_dims", []):
            rows.append({
                "model": model,
                "dataset": dataset,
                "language": item["language"],
                "effective_dim": float(item["effective_dim"]),
            })
    return rows


def read_mapping(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["input_code"].strip(): row["wiki_code"].strip()
            for row in reader
            if row.get("input_code") and row.get("wiki_code")
        }


def read_counts(path: Path) -> dict[str, int]:
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t").delimiter if sample else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = set(reader.fieldnames or [])
        code_field = "wiki_code" if "wiki_code" in fieldnames else "prefix"
        count_field = "count" if "count" in fieldnames else "good"
        return {
            row[code_field].strip(): int(row[count_field].strip().replace(",", ""))
            for row in reader
            if row.get(code_field) and row.get(count_field)
        }


def add_resource_columns(
    rows: list[dict[str, Any]],
    language_to_wiki: dict[str, str],
    wiki_counts: dict[str, int],
) -> list[dict[str, Any]]:
    language_counts = {
        row["language"]: wiki_count(row["language"], language_to_wiki, wiki_counts)
        for row in rows
    }
    ranks = {
        language: index + 1
        for index, language in enumerate(
            sorted(language_counts, key=lambda language: (-language_counts[language], language))
        )
    }
    return [
        {
            **row,
            "wiki_count": language_counts[row["language"]],
            "resource_rank": ranks[row["language"]],
        }
        for row in rows
    ]


def average_by_dataset_language(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mean_rows = []
    for (dataset, language), subset in sorted(group_by(rows, "dataset", "language").items()):
        values = [row["effective_dim"] for row in subset]
        mean_rows.append({
            "dataset": dataset,
            "language": language,
            "wiki_count": subset[0]["wiki_count"],
            "resource_rank": subset[0]["resource_rank"],
            "num_models": len(values),
            "mean_effective_dim": sum(values) / len(values),
            "min_effective_dim": min(values),
            "max_effective_dim": max(values),
        })
    return mean_rows


def plot_dataset_means(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for dataset, subset in sorted(group_by(rows, "dataset").items()):
        subset = sort_reverse_resource(subset)
        fig, ax = plt.subplots(figsize=(bar_width(len(subset)), 5))
        ax.bar(
            [row["language"] for row in subset],
            [row["mean_effective_dim"] for row in subset],
            color="#4C78A8",
        )
        ax.set_title(f"Per-language effective dimensionality: {pretty(dataset)}")
        ax.set_xlabel("Language, least to most resourced by Wikipedia article count")
        ax.set_ylabel("Mean effective dimensionality across models")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__mean_language_effective_dims_by_reverse_resource", formats)


def plot_model_details(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for (dataset, model), subset in sorted(group_by(rows, "dataset", "model").items()):
        subset = sort_reverse_resource(subset)
        fig, ax = plt.subplots(figsize=(bar_width(len(subset)), 5))
        ax.bar(
            [row["language"] for row in subset],
            [row["effective_dim"] for row in subset],
            color="#59A14F",
        )
        ax.set_title(f"{model} on {pretty(dataset)}")
        ax.set_xlabel("Language, least to most resourced by Wikipedia article count")
        ax.set_ylabel("Effective dimensionality")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / "detailed_plots" / f"{dataset}__{slugify(model)}__language_effective_dims_by_reverse_resource",
            formats,
        )


def sort_reverse_resource(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["wiki_count"], row["language"]))


def wiki_count(language: str, language_to_wiki: dict[str, str], wiki_counts: dict[str, int]) -> int:
    wiki_code = language_to_wiki.get(language)
    return 0 if wiki_code is None else wiki_counts.get(wiki_code, 0)


def bar_width(num_bars: int) -> float:
    return max(8, min(40, 0.16 * num_bars + 4))


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, output_base: Path, formats: list[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def row_fieldnames() -> list[str]:
    return ["dataset", "model", "language", "wiki_count", "resource_rank", "effective_dim"]


def mean_fieldnames() -> list[str]:
    return [
        "dataset",
        "language",
        "wiki_count",
        "resource_rank",
        "num_models",
        "mean_effective_dim",
        "min_effective_dim",
        "max_effective_dim",
    ]


def group_by(rows: list[dict[str, Any]], *keys: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[key] for key in keys)
        if len(key) == 1:
            key = key[0]
        groups.setdefault(key, []).append(row)
    return groups


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def slugify(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
