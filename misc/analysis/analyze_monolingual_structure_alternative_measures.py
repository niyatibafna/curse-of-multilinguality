from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = PROJECT_ROOT / "src" / "utils" / "language_sorting"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "monolingual_structure_alternative_measures"
)
DEFAULT_LANGUAGE_TO_WIKI = UTILS_DIR / "language_to_wiki.csv"
DEFAULT_WIKI_COUNTS = UTILS_DIR / "wiki_counts.csv"
METRIC = "monolingual_structure_condition"
PAIR_ID_FIELDS = ("pair_id", "concept_pair_id", "pair_index")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute English-vs-language alternative monolingual-structure diagnostics."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--distance-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--language-to-wiki", type=Path, default=DEFAULT_LANGUAGE_TO_WIKI)
    parser.add_argument("--wiki-counts", type=Path, default=DEFAULT_WIKI_COUNTS)
    parser.add_argument("--reference-language", default="eng_Latn")
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    language_to_wiki = read_mapping(args.language_to_wiki)
    wiki_counts = read_counts(args.wiki_counts)

    vectors = load_distance_csv(args.distance_csv) if args.distance_csv else {}
    report_rows, json_vectors = load_json_vectors(args.input_dir)
    write_csv(report_rows, args.output_dir / "monolingual_structure_vector_availability.csv", report_fieldnames())
    vectors.update(json_vectors)

    if not vectors:
        raise ValueError(
            f"No pairwise distance vectors found. Existing {METRIC} JSON files store aggregate "
            "Pearson language-pair correlations, not the per-language concept-pair distance vectors "
            "needed for Spearman, z-scored MAE/RMSE, or linear fits. Re-run with --distance-csv "
            "containing columns model,dataset,language,pair_id,distance, or provide JSON outputs "
            "that include result.distance_vectors."
        )

    rows = compute_rows(vectors, args.reference_language, language_to_wiki, wiki_counts)
    if not rows:
        raise ValueError(
            f"Distance vectors were loaded, but no model/dataset group contained reference language "
            f"{args.reference_language!r} plus at least one comparison language."
        )

    write_csv(
        rows,
        args.output_dir / "monolingual_structure_alternative_measures.csv",
        measure_fieldnames(),
    )
    plot_measures(rows, args.output_dir, args.formats)
    print(f"Wrote monolingual alternative measures to {args.output_dir}")


def load_json_vectors(input_dir: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, np.ndarray]]]:
    report_rows = []
    vectors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for path in sorted(input_dir.glob(f"*/*/{METRIC}.json")):
        with path.open() as handle:
            payload = json.load(handle)
        result = payload.get("result") if isinstance(payload, dict) else None
        result = result if isinstance(result, dict) else {}
        model = str(payload.get("model") or path.parents[1].name)
        dataset = str(payload.get("dataset") or path.parent.name)
        distance_vectors = result.get("distance_vectors")
        has_vectors = isinstance(distance_vectors, dict)
        report_rows.append({
            "model": model,
            "dataset": dataset,
            "path": str(path),
            "has_language_pair_correlations": isinstance(result.get("language_pairs"), list),
            "has_distance_vectors": has_vectors,
            "num_languages": int(result.get("num_languages", 0) or 0),
            "num_concept_pairs": int(result.get("num_concept_pairs", 0) or 0),
            "usable_for_alternative_measures": has_vectors,
        })
        if has_vectors:
            vectors[(model, dataset)] = {
                language: np.asarray(values, dtype=float)
                for language, values in distance_vectors.items()
            }
    return report_rows, vectors


def load_distance_csv(path: Path) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if {"model", "dataset", "language", "distance"}.issubset(fieldnames):
            pair_field = next((field for field in PAIR_ID_FIELDS if field in fieldnames), None)
            if pair_field is None:
                raise ValueError(f"{path} needs one of these pair-id columns: {', '.join(PAIR_ID_FIELDS)}")
            return load_long_distance_rows(reader, pair_field)
        if {"model", "dataset"}.issubset(fieldnames):
            pair_field = next((field for field in PAIR_ID_FIELDS if field in fieldnames), None)
            if pair_field is None:
                raise ValueError(f"{path} needs one of these pair-id columns: {', '.join(PAIR_ID_FIELDS)}")
            language_fields = [
                field for field in fieldnames
                if field not in {"model", "dataset", pair_field}
            ]
            return load_wide_distance_rows(reader, language_fields)
    raise ValueError(
        f"{path} must be either long CSV columns model,dataset,language,pair_id,distance "
        "or wide CSV columns model,dataset,pair_id,<language columns>."
    )


def load_long_distance_rows(
    reader: csv.DictReader,
    pair_field: str,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    values: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for row in reader:
        key = (row["model"], row["dataset"], row["language"])
        values.setdefault(key, []).append((row[pair_field], float(row["distance"])))
    vectors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    pair_ids_by_group: dict[tuple[str, str], list[str]] = {}
    for (model, dataset, language), items in values.items():
        items.sort(key=lambda item: item[0])
        pair_ids = [pair_id for pair_id, _ in items]
        group_key = (model, dataset)
        if group_key in pair_ids_by_group and pair_ids_by_group[group_key] != pair_ids:
            raise ValueError(
                f"Long distance CSV has mismatched pair IDs for {model}/{dataset}; "
                "provide the same ordered pair_id set for each language."
            )
        pair_ids_by_group[group_key] = pair_ids
        vectors.setdefault((model, dataset), {})[language] = np.array([value for _, value in items])
    return vectors


def load_wide_distance_rows(
    reader: csv.DictReader,
    language_fields: list[str],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    values: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in reader:
        group = values.setdefault((row["model"], row["dataset"]), {language: [] for language in language_fields})
        for language in language_fields:
            if row.get(language) not in {None, ""}:
                group[language].append(float(row[language]))
    return {
        key: {language: np.asarray(items, dtype=float) for language, items in group.items()}
        for key, group in values.items()
    }


def compute_rows(
    vectors: dict[tuple[str, str], dict[str, np.ndarray]],
    reference_language: str,
    language_to_wiki: dict[str, str],
    wiki_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for (model, dataset), language_vectors in sorted(vectors.items()):
        reference = language_vectors.get(reference_language)
        if reference is None:
            continue
        for language, distances in sorted(language_vectors.items()):
            if language == reference_language:
                continue
            count = min(len(reference), len(distances))
            if count == 0:
                continue
            ref = reference[:count]
            other = distances[:count]
            z_ref = zscore(ref)
            z_other = zscore(other)
            slope, intercept = linear_fit(ref, other)
            rows.append({
                "model": model,
                "dataset": dataset,
                "reference_language": reference_language,
                "language": language,
                "wiki_code": language_to_wiki.get(language, ""),
                "wiki_count": wiki_count(language, language_to_wiki, wiki_counts),
                "num_pairs": count,
                "spearman": spearman(ref, other),
                "z_mae": mean_abs_error(z_ref, z_other),
                "z_rmse": root_mean_squared_error(z_ref, z_other),
                "linear_slope": slope,
                "linear_intercept": intercept,
            })
    return sorted(rows, key=lambda row: (row["dataset"], row["model"], -row["wiki_count"], row["language"]))


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom == 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / denom)


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0.0:
        return np.full(len(values), np.nan)
    return (values - np.mean(values)) / std


def mean_abs_error(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return None
    return float(np.mean(np.abs(x - y)))


def root_mean_squared_error(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return None
    return float(np.sqrt(np.mean((x - y) ** 2)))


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    if len(x) < 2 or float(np.std(x)) == 0.0:
        return None, None
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def plot_measures(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
    for (dataset, model), subset in sorted(group_by(rows, "dataset", "model").items()):
        subset = sorted(subset, key=lambda row: (-row["wiki_count"], row["language"]))
        x = np.arange(len(subset))
        labels = [row["language"] for row in subset]
        for field, ylabel in [
            ("spearman", "Spearman correlation"),
            ("z_mae", "Z-scored MAE"),
            ("z_rmse", "Z-scored RMSE"),
            ("linear_slope", "Linear fit slope"),
            ("linear_intercept", "Linear fit intercept"),
        ]:
            values = [nan_if_none(row[field]) for row in subset]
            if np.all(np.isnan(values)):
                continue
            fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(subset) + 3), 5))
            ax.plot(x, values, marker="o", markersize=2.5, linewidth=1)
            ax.set_title(f"{pretty(model)} on {pretty(dataset)}: {ylabel}")
            ax.set_xlabel("Language, sorted by Wikipedia article count")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x[:: max(1, len(x) // 20)], labels[:: max(1, len(x) // 20)], rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            save_figure(fig, output_dir / f"{dataset}__{model}__{field}_by_wiki_count", formats)


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


def wiki_count(language: str, language_to_wiki: dict[str, str], wiki_counts: dict[str, int]) -> int:
    wiki_code = language_to_wiki.get(language)
    if wiki_code is None:
        return 0
    return wiki_counts.get(wiki_code, 0)


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


def report_fieldnames() -> list[str]:
    return [
        "model",
        "dataset",
        "path",
        "has_language_pair_correlations",
        "has_distance_vectors",
        "num_languages",
        "num_concept_pairs",
        "usable_for_alternative_measures",
    ]


def measure_fieldnames() -> list[str]:
    return [
        "model",
        "dataset",
        "reference_language",
        "language",
        "wiki_code",
        "wiki_count",
        "num_pairs",
        "spearman",
        "z_mae",
        "z_rmse",
        "linear_slope",
        "linear_intercept",
    ]


def group_by(rows: list[dict[str, Any]], *keys: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[key] for key in keys)
        if len(key) == 1:
            key = key[0]
        groups.setdefault(key, []).append(row)
    return groups


def nan_if_none(value: float | None) -> float:
    if value is None:
        return np.nan
    return value


def pretty(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


if __name__ == "__main__":
    main()
