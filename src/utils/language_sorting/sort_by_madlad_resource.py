from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    from .sort_by_wiki_count import read_dataset_langs, read_langs, require_columns
except ImportError:
    from sort_by_wiki_count import read_dataset_langs, read_langs, require_columns


UTILS_DIR = Path(__file__).resolve().parent
DEFAULT_LANGUAGE_TO_MADLAD = UTILS_DIR / "language_to_madlad.csv"
DEFAULT_MADLAD_COUNTS = UTILS_DIR / "madlad_counts.csv"
DEFAULT_DATASET_LANGUAGES = UTILS_DIR / "dataset_languages.csv"

SORT_KEYS = {
    "clean_docs",
    "clean_sents",
    "clean_tokens",
    "clean_chars",
    "clean_bytes",
    "noisy_docs",
    "noisy_sents",
    "noisy_tokens",
    "noisy_chars",
    "noisy_bytes",
}


def read_mapping(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        require_columns(reader, path, {"input_code", "madlad_code"})
        return {
            row["input_code"].strip(): row["madlad_code"].strip()
            for row in reader
            if row.get("input_code") and row.get("madlad_code")
        }


def read_counts(path: Path, key: str) -> dict[str, int]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        require_columns(reader, path, {"madlad_code", key})
        return {
            row["madlad_code"].strip(): int(row[key])
            for row in reader
            if row.get("madlad_code") and row.get(key)
        }


def sort_madlad_codes(madlad_counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(madlad_counts.items(), key=lambda row: row[1], reverse=True)


def sort_langs_by_madlad_resource(
    langs: list[str],
    language_to_madlad: dict[str, str],
    madlad_counts: dict[str, int],
    quiet: bool = False,
) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for lang in langs:
        madlad_code = language_to_madlad.get(lang)
        if madlad_code is None:
            warn(f"missing MADLAD mapping for {lang}; using count 0", quiet)
            rows.append((lang, 0))
            continue

        count = madlad_counts.get(madlad_code)
        if count is None:
            warn(f"missing MADLAD count for {lang} ({madlad_code}); using count 0", quiet)
            rows.append((lang, 0))
            continue

        rows.append((lang, count))

    return sorted(rows, key=lambda row: row[1], reverse=True)


def warn(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"warning: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort benchmark language codes by MADLAD resource size."
    )
    parser.add_argument(
        "langs",
        nargs="?",
        type=Path,
        help="File with one benchmark language code per line.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Use bundled languages for a current dataset, e.g. bouquet/floresplus/wmt24pp.",
    )
    parser.add_argument(
        "--all-madlad",
        action="store_true",
        help="Sort all MADLAD language codes directly.",
    )
    parser.add_argument(
        "--key",
        choices=sorted(SORT_KEYS),
        default="clean_bytes",
        help="MADLAD count column to sort by.",
    )
    parser.add_argument(
        "--language-to-madlad",
        type=Path,
        default=DEFAULT_LANGUAGE_TO_MADLAD,
        help=f"Mapping CSV. Default: {DEFAULT_LANGUAGE_TO_MADLAD}",
    )
    parser.add_argument(
        "--madlad-counts",
        type=Path,
        default=DEFAULT_MADLAD_COUNTS,
        help=f"MADLAD counts CSV. Default: {DEFAULT_MADLAD_COUNTS}",
    )
    parser.add_argument(
        "--dataset-languages",
        type=Path,
        default=DEFAULT_DATASET_LANGUAGES,
        help=f"Dataset language CSV. Default: {DEFAULT_DATASET_LANGUAGES}",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress zero-fill warnings for missing mappings/counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    langs: list[str] = []
    if args.langs is not None:
        langs.extend(read_langs(args.langs))
    for dataset in args.dataset or []:
        langs.extend(read_dataset_langs(args.dataset_languages, dataset))
    if not langs and not args.all_madlad:
        raise SystemExit("Provide langs.txt or --dataset.")

    madlad_counts = read_counts(args.madlad_counts, args.key)
    if args.all_madlad:
        rows = sort_madlad_codes(madlad_counts)
    else:
        language_to_madlad = read_mapping(args.language_to_madlad)
        rows = sort_langs_by_madlad_resource(
            langs,
            language_to_madlad,
            madlad_counts,
            quiet=args.quiet,
        )

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["lang", args.key])
    writer.writerows(rows)


if __name__ == "__main__":
    main()
