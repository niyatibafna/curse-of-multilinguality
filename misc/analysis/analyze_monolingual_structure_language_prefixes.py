from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = PROJECT_ROOT / "src" / "utils" / "language_sorting"
DEFAULT_PAIR_CSV = (
    PROJECT_ROOT
    / "misc"
    / "results_vis"
    / "plots"
    / "monolingual_structure_condition"
    / "monolingual_structure_condition_language_pairs.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "monolingual_structure_language_prefixes"
DEFAULT_LANGUAGE_TO_WIKI = UTILS_DIR / "language_to_wiki.csv"
DEFAULT_WIKI_COUNTS = UTILS_DIR / "wiki_counts.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MonolingualStructureCondition as languages are added by Wikipedia count."
    )
    parser.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--language-to-wiki", type=Path, default=DEFAULT_LANGUAGE_TO_WIKI)
    parser.add_argument("--wiki-counts", type=Path, default=DEFAULT_WIKI_COUNTS)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--order", choices=["wiki", "random"], default="wiki")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    if args.step < 2:
        raise ValueError("--step must be at least 2.")

    pair_rows = read_pair_rows(args.pair_csv)
    language_to_wiki = read_mapping(args.language_to_wiki)
    wiki_counts = read_counts(args.wiki_counts)

    rows = compute_prefix_scores(
        pair_rows,
        language_to_wiki,
        wiki_counts,
        args.step,
        args.order,
        args.random_seed,
    )
    avg_rows = average_by_dataset(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "monolingual_structure_language_prefix_scores.csv", prefix_fieldnames())
    write_csv(
        avg_rows,
        args.output_dir / "monolingual_structure_language_prefix_dataset_means.csv",
        mean_fieldnames(),
    )
    plot_dataset_curves(rows, avg_rows, args.output_dir, args.formats)
    plot_mean_curves(avg_rows, args.output_dir, args.formats)

    print(f"Wrote monolingual-structure language-prefix analysis to {args.output_dir}")


def read_pair_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            correlation = optional_float(row.get("correlation"))
            if correlation is None:
                continue
            rows.append({
                "model": row["model"],
                "dataset": row["dataset"],
                "language_1": row["language_1"],
                "language_2": row["language_2"],
                "correlation": correlation,
                "num_concept_pairs": int(row["num_concept_pairs"]),
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


def compute_prefix_scores(
    pair_rows: list[dict[str, Any]],
    language_to_wiki: dict[str, str],
    wiki_counts: dict[str, int],
    step: int,
    order: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for (dataset, model), subset in sorted(group_by(pair_rows, "dataset", "model").items()):
        languages = ordered_languages(
            sorted({row["language_1"] for row in subset} | {row["language_2"] for row in subset}),
            dataset,
            model,
            language_to_wiki,
            wiki_counts,
            order,
            random_seed,
        )
        ranks = {language: index + 1 for index, language in enumerate(languages)}
        prefix_sizes = list(range(step, len(languages) + 1, step))
        if prefix_sizes[-1] != len(languages):
            prefix_sizes.append(len(languages))

        events: dict[int, list[dict[str, Any]]] = {}
        for row in subset:
            required_size = max(ranks[row["language_1"]], ranks[row["language_2"]])
            events.setdefault(required_size, []).append(row)

        correlation_sum = 0.0
        num_language_pairs = 0
        event_size = 0
        for prefix_size in prefix_sizes:
            while event_size < prefix_size:
                event_size += 1
                for row in events.get(event_size, []):
                    correlation_sum += row["correlation"]
                    num_language_pairs += 1

            rows.append({
                "dataset": dataset,
                "model": model,
                "group_size": prefix_size,
                "num_languages": len(languages),
                "num_language_pairs": num_language_pairs,
                "score": score(correlation_sum, num_language_pairs),
                "order": order,
                "random_seed": random_seed if order == "random" else "",
            })
    return rows


def ordered_languages(
    languages: list[str],
    dataset: str,
    model: str,
    language_to_wiki: dict[str, str],
    wiki_counts: dict[str, int],
    order: str,
    random_seed: int,
) -> list[str]:
    if order == "wiki":
        return sorted(
            languages,
            key=lambda language: (-wiki_count(language, language_to_wiki, wiki_counts), language),
        )
    if order == "random":
        rng = random.Random(f"{random_seed}:{dataset}:{model}")
        shuffled = list(languages)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unknown order: {order}")


def wiki_count(language: str, language_to_wiki: dict[str, str], wiki_counts: dict[str, int]) -> int:
    wiki_code = language_to_wiki.get(language)
    if wiki_code is None:
        return 0
    return wiki_counts.get(wiki_code, 0)


def average_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    avg_rows = []
    for (dataset, group_size), subset in sorted(group_by(rows, "dataset", "group_size").items()):
        scores = [row["score"] for row in subset if row["score"] is not None]
        if not scores:
            continue
        avg_rows.append({
            "dataset": dataset,
            "group_size": group_size,
            "num_models": len(scores),
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
        })
    return avg_rows


def plot_dataset_curves(
    rows: list[dict[str, Any]],
    avg_rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    for dataset, subset in sorted(group_by(rows, "dataset").items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, model_rows in sorted(group_by(subset, "model").items()):
            model_rows = sorted(
                [row for row in model_rows if row["score"] is not None],
                key=lambda row: row["group_size"],
            )
            ax.plot(
                [row["group_size"] for row in model_rows],
                [row["score"] for row in model_rows],
                linewidth=1.2,
                alpha=0.6,
                label=model,
            )

        mean_rows = sorted(
            [row for row in avg_rows if row["dataset"] == dataset],
            key=lambda row: row["group_size"],
        )
        ax.plot(
            [row["group_size"] for row in mean_rows],
            [row["mean_score"] for row in mean_rows],
            color="black",
            linewidth=2.6,
            label="mean",
        )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"MonolingualStructureCondition by language-prefix size: {pretty(dataset)}")
        ax.set_xlabel("Number of languages, sorted by Wikipedia article count")
        ax.set_ylabel("Average correlation")
        ax.set_ylim(-1, 1)
        ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{dataset}__monolingual_structure_by_language_prefix", formats)


def plot_mean_curves(
    avg_rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, dataset_rows in sorted(group_by(avg_rows, "dataset").items()):
        dataset_rows = sorted(dataset_rows, key=lambda row: row["group_size"])
        ax.plot(
            [row["group_size"] for row in dataset_rows],
            [row["mean_score"] for row in dataset_rows],
            linewidth=2,
            label=pretty(dataset),
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Mean MonolingualStructureCondition by language-prefix size")
    ax.set_xlabel("Number of languages, sorted by Wikipedia article count")
    ax.set_ylabel("Mean correlation across models")
    ax.set_ylim(-1, 1)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "all_datasets__mean_monolingual_structure_by_language_prefix", formats)


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


def optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def score(correlation_sum: float, num_language_pairs: int) -> float | None:
    if num_language_pairs == 0:
        return None
    return correlation_sum / num_language_pairs


def prefix_fieldnames() -> list[str]:
    return [
        "dataset",
        "model",
        "group_size",
        "num_languages",
        "num_language_pairs",
        "score",
        "order",
        "random_seed",
    ]


def mean_fieldnames() -> list[str]:
    return ["dataset", "group_size", "num_models", "mean_score", "min_score", "max_score"]


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


if __name__ == "__main__":
    main()
