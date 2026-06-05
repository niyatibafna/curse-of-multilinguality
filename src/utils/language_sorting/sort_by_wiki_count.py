from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


UTILS_DIR = Path(__file__).resolve().parent
DEFAULT_LANGUAGE_TO_WIKI = UTILS_DIR / "language_to_wiki.csv"
DEFAULT_WIKI_COUNTS = UTILS_DIR / "wiki_counts.csv"
DEFAULT_DATASET_LANGUAGES = UTILS_DIR / "dataset_languages.csv"
WIKISTATS_CSV_URL = (
    "https://wikistats.wmcloud.org/api.php?action=dump&format=csv&table=wikipedias"
)


def read_langs(path: Path) -> list[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def read_dataset_langs(path: Path, dataset: str) -> list[str]:
    langs: list[str] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        require_columns(reader, path, {"dataset", "lang"})
        for row in reader:
            if row["dataset"].strip() == dataset:
                langs.append(row["lang"].strip())
    if not langs:
        raise ValueError(f"No languages found for dataset '{dataset}' in {path}")
    return langs


def read_mapping(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        require_columns(reader, path, {"input_code", "wiki_code"})
        return {
            row["input_code"].strip(): row["wiki_code"].strip()
            for row in reader
            if row.get("input_code") and row.get("wiki_code")
        }


def read_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=table_delimiter(path))
        code_column, count_column = count_columns(reader, path)
        for row in reader:
            wiki_code = row.get(code_column, "").strip()
            count = row.get(count_column, "").strip().replace(",", "")
            if wiki_code and count:
                counts[wiki_code] = int(count)
    return counts


def table_delimiter(path: Path) -> str:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(newline="") as f:
        sample = f.read(4096)
    if sample:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
        except csv.Error:
            pass
    return delimiter


def count_columns(reader: csv.DictReader, path: Path) -> tuple[str, str]:
    fieldnames = set(reader.fieldnames or [])
    if {"wiki_code", "count"} <= fieldnames:
        return "wiki_code", "count"
    if {"prefix", "good"} <= fieldnames:
        return "prefix", "good"
    raise ValueError(
        f"{path} must contain wiki_code,count or Wikistats prefix,good columns"
    )


def require_columns(reader: csv.DictReader, path: Path, columns: set[str]) -> None:
    fieldnames = set(reader.fieldnames or [])
    missing = columns - fieldnames
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required column(s): {missing_text}")


def sort_langs_by_wiki_count(
    langs: list[str],
    language_to_wiki: dict[str, str],
    wiki_counts: dict[str, int],
    quiet: bool = False,
) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for lang in langs:
        wiki_code = language_to_wiki.get(lang)
        if wiki_code is None:
            warn(f"missing mapping for {lang}; using count 0", quiet)
            rows.append((lang, 0))
            continue

        count = wiki_counts.get(wiki_code)
        if count is None:
            warn(f"missing Wikipedia count for {lang} ({wiki_code}); using count 0", quiet)
            rows.append((lang, 0))
            continue

        rows.append((lang, count))

    return sorted(rows, key=lambda row: row[1], reverse=True)


def warn(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"warning: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort benchmark language codes by Wikipedia article count.",
        epilog=(
            "Default counts come from the normalized local Wikistats snapshot at "
            f"{DEFAULT_WIKI_COUNTS}. Refresh source: {WIKISTATS_CSV_URL}"
        ),
    )
    parser.add_argument(
        "langs",
        nargs="?",
        type=Path,
        help="File with one benchmark language code per line.",
    )
    parser.add_argument(
        "language_to_wiki_pos",
        nargs="?",
        type=Path,
        help="Optional positional mapping CSV, for backward compatibility.",
    )
    parser.add_argument(
        "wiki_counts_pos",
        nargs="?",
        type=Path,
        help="Optional positional counts CSV/TSV, for backward compatibility.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Use bundled languages for a current dataset, e.g. bouquet/floresplus/wmt24pp.",
    )
    parser.add_argument(
        "--language-to-wiki",
        type=Path,
        default=DEFAULT_LANGUAGE_TO_WIKI,
        help=f"Mapping CSV. Default: {DEFAULT_LANGUAGE_TO_WIKI}",
    )
    parser.add_argument(
        "--wiki-counts",
        type=Path,
        default=DEFAULT_WIKI_COUNTS,
        help=f"Counts CSV/TSV. Default: {DEFAULT_WIKI_COUNTS}",
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
        help="Suppress skip warnings for unmapped languages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    langs: list[str] = []
    if args.langs is not None:
        langs.extend(read_langs(args.langs))
    for dataset in args.dataset or []:
        langs.extend(read_dataset_langs(args.dataset_languages, dataset))
    if not langs:
        raise SystemExit("Provide langs.txt or --dataset.")

    language_to_wiki_path = args.language_to_wiki_pos or args.language_to_wiki
    wiki_counts_path = args.wiki_counts_pos or args.wiki_counts
    language_to_wiki = read_mapping(language_to_wiki_path)
    wiki_counts = read_counts(wiki_counts_path)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["lang", "count"])
    writer.writerows(
        sort_langs_by_wiki_count(
            langs,
            language_to_wiki,
            wiki_counts,
            quiet=args.quiet,
        )
    )


if __name__ == "__main__":
    main()
