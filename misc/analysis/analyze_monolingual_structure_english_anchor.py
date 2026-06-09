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
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PAIR_CSV = (
    PROJECT_ROOT
    / "misc"
    / "results_vis"
    / "plots"
    / "monolingual_structure_condition"
    / "monolingual_structure_condition_language_pairs.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "monolingual_structure_english_anchor"
)
DEFAULT_LANGUAGE_TO_WIKI = UTILS_DIR / "language_to_wiki.csv"
DEFAULT_WIKI_COUNTS = UTILS_DIR / "wiki_counts.csv"
DEFAULT_ENGLISH_LANGUAGES = ["eng_Latn", "en"]
METRIC = "monolingual_structure_condition"
MEASURES = [
    ("pearson", "Pearson", (-1, 1)),
    ("spearman", "Spearman", (-1, 1)),
    ("mae", "MAE", None),
    ("rmse", "RMSE", None),
    ("normalized_rmse", "Normalized RMSE", None),
    ("centered_rmse", "Centered RMSE", None),
    ("standardized_rmse", "Standardized RMSE", None),
    ("mean_distance_ratio_to_english", "Mean Distance / English", None),
    ("std_distance_ratio_to_english", "Std Distance / English", None),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot English-anchored monolingual-structure measures by Wikipedia count."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--language-to-wiki", type=Path, default=DEFAULT_LANGUAGE_TO_WIKI)
    parser.add_argument("--wiki-counts", type=Path, default=DEFAULT_WIKI_COUNTS)
    parser.add_argument("--english-language", action="append", default=[])
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    language_to_wiki = read_mapping(args.language_to_wiki)
    wiki_counts = read_counts(args.wiki_counts)
    english_languages = set(args.english_language or DEFAULT_ENGLISH_LANGUAGES)
    rows = load_pair_csv_rows(args.pair_csv, english_languages, language_to_wiki, wiki_counts)
    if not rows:
        rows = load_json_rows(args.input_dir, english_languages, language_to_wiki, wiki_counts)
    if not rows:
        raise ValueError(f"No English-anchored {METRIC} measures found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "monolingual_structure_english_anchor.csv")
    plot_dataset_measure_panels(rows, args.output_dir, args.formats)
    print(f"Wrote English-anchored monolingual-structure measures to {args.output_dir}")


def load_pair_csv_rows(
    pair_csv: Path,
    english_languages: set[str],
    language_to_wiki: dict[str, str],
    wiki_counts: dict[str, int],
) -> list[dict[str, Any]]:
    if not pair_csv.exists():
        return []

    rows = []
    with pair_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for pair in reader:
            target = english_target(pair, english_languages)
            if target is None:
                continue
            row = anchored_row(pair, target, language_to_wiki, wiki_counts)
            rows.append(row)

    return add_resource_ranks(rows)


def load_json_rows(
    input_dir: Path,
    english_languages: set[str],
    language_to_wiki: dict[str, str],
    wiki_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob(f"*/*/{METRIC}.json")):
        with path.open() as handle:
            payload = json.load(handle)

        result = payload.get("result", {})
        if not isinstance(result, dict):
            continue

        model = str(payload.get("model") or path.parents[1].name)
        dataset = str(payload.get("dataset") or path.parent.name)
        dataset_rows = []
        for pair in result.get("language_pairs", []):
            target = english_target(pair, english_languages)
            if target is None:
                continue
            pair = {**pair, "dataset": dataset, "model": model}
            dataset_rows.append(anchored_row(pair, target, language_to_wiki, wiki_counts))
        rows.extend(dataset_rows)
    return add_resource_ranks(rows)


def add_resource_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for (dataset, model), subset in sorted(group_by(rows, "dataset", "model").items()):
        for rank, row in enumerate(
            sorted(subset, key=lambda row: (-row["wiki_count"], row["language"])),
            start=1,
        ):
            ranked.append({**row, "rank": rank})
    return ranked


def anchored_row(
    pair: dict[str, Any],
    target: str,
    language_to_wiki: dict[str, str],
    wiki_counts: dict[str, int],
) -> dict[str, Any]:
    language_1 = str(pair.get("language_1", ""))
    language_2 = str(pair.get("language_2", ""))
    english_is_language_1 = language_1 != target

    mean_ratio = optional_float(pair.get("mean_distance_ratio"))
    std_ratio = optional_float(pair.get("std_distance_ratio"))
    if not english_is_language_1:
        mean_ratio = invert_ratio(mean_ratio)
        std_ratio = invert_ratio(std_ratio)

    return {
        "dataset": str(pair["dataset"]),
        "model": str(pair["model"]),
        "language": target,
        "wiki_count": wiki_count(target, language_to_wiki, wiki_counts),
        "pearson": optional_float(pair.get("pearson", pair.get("correlation"))),
        "spearman": optional_float(pair.get("spearman")),
        "mae": optional_float(pair.get("mae")),
        "rmse": optional_float(pair.get("rmse")),
        "normalized_rmse": optional_float(pair.get("normalized_rmse")),
        "centered_rmse": optional_float(pair.get("centered_rmse")),
        "standardized_rmse": optional_float(pair.get("standardized_rmse")),
        "mean_distance_ratio_to_english": mean_ratio,
        "std_distance_ratio_to_english": std_ratio,
    }


def english_target(pair: dict[str, Any], english_languages: set[str]) -> str | None:
    language_1 = str(pair.get("language_1", ""))
    language_2 = str(pair.get("language_2", ""))
    if language_1 in english_languages and language_2 not in english_languages:
        return language_2
    if language_2 in english_languages and language_1 not in english_languages:
        return language_1
    return None


def read_mapping(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["input_code"].strip(): row["wiki_code"].strip()
            for row in reader
            if row.get("input_code") and row.get("wiki_code")
        }


def read_counts(path: Path) -> dict[str, int]:
    counts = {}
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t").delimiter if sample else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = set(reader.fieldnames or [])
        code_field = "wiki_code" if "wiki_code" in fieldnames else "prefix"
        count_field = "count" if "count" in fieldnames else "good"
        for row in reader:
            code = row.get(code_field, "").strip()
            count = row.get(count_field, "").strip().replace(",", "")
            if code and count:
                counts[code] = int(count)
    return counts


def wiki_count(language: str, language_to_wiki: dict[str, str], wiki_counts: dict[str, int]) -> int:
    wiki_code = language_to_wiki.get(language)
    if wiki_code is None:
        return 0
    return wiki_counts.get(wiki_code, 0)


def plot_dataset_measure_panels(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for dataset, subset in sorted(group_by(rows, "dataset").items()):
        fig, axes = plt.subplots(3, 4, figsize=(20, 11), sharex=True)
        axes_flat = list(axes.flat)
        for ax, (field, label, ylim) in zip(axes_flat, MEASURES):
            for model, model_rows in sorted(group_by(subset, "model").items()):
                model_rows = [
                    row for row in sorted(model_rows, key=lambda row: row["rank"])
                    if row[field] is not None
                ]
                if not model_rows:
                    continue
                ax.plot(
                    [row["rank"] for row in model_rows],
                    [row[field] for row in model_rows],
                    linewidth=1.1,
                    alpha=0.72,
                    label=model,
                )
            ax.set_title(label)
            ax.set_ylabel(label)
            ax.grid(alpha=0.2, linewidth=0.6)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if "ratio" in field:
                ax.axhline(1, color="#333333", linewidth=0.8)
            elif field in {"pearson", "spearman"}:
                ax.axhline(0, color="#333333", linewidth=0.8)
        for ax in axes[-1]:
            ax.set_xlabel("Language rank by Wikipedia article count")
        for ax in axes_flat[len(MEASURES):]:
            ax.axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            axes_flat[-1].legend(handles, labels, fontsize=8, ncols=2, loc="center")
        fig.suptitle(f"English-Anchored Monolingual Structure Measures: {pretty(dataset)}")
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__english_anchor_measures_by_resource_rank", formats)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "model",
        "language",
        "wiki_count",
        "rank",
        *[field for field, _, _ in MEASURES],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (row["dataset"], row["model"], row["rank"], row["language"]))
        )


def save_figure(fig: plt.Figure, output_base: Path, formats: list[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def group_by(rows: list[dict[str, Any]], *keys: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[key] for key in keys)
        if len(key) == 1:
            key = key[0]
        groups.setdefault(key, []).append(row)
    return groups


def optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def invert_ratio(value: float | None) -> float | None:
    if value is None or value == 0.0:
        return None
    return 1.0 / value


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


if __name__ == "__main__":
    main()
