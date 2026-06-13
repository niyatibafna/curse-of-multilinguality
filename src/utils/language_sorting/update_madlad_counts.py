from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path


UTILS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = UTILS_DIR / "madlad_counts.csv"
MADLAD_CARD_URL = (
    "https://huggingface.co/datasets/allenai/MADLAD-400/raw/main/README.md"
)

COUNT_COLUMNS = {
    "docs (noisy)": "noisy_docs",
    "docs (clean)": "clean_docs",
    "sents (noisy)": "noisy_sents",
    "sents (clean)": "clean_sents",
    "toks (noisy)": "noisy_tokens",
    "toks (clean)": "clean_tokens",
    "chars (noisy)": "noisy_chars",
    "chars (clean)": "clean_chars",
    "clean": "clean_bytes",
    "noisy": "noisy_bytes",
}


def read_text(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url) as response:
            return response.read().decode("utf-8")
    return Path(path_or_url).read_text()


def parse_madlad_counts(markdown: str) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    lines = iter(markdown.splitlines())
    for line in lines:
        if line.startswith("BCP-47"):
            header = split_row(line)
            break
    else:
        raise ValueError("Could not find MADLAD final dataset table.")

    for line in lines:
        if not line.strip():
            break
        if set(line.replace("|", "").strip()) <= {"-", ":"}:
            continue

        values = split_row(line)
        if len(values) != len(header):
            continue
        raw = dict(zip(header, values))
        lang = raw["BCP-47"].rstrip("*")
        if lang == "total":
            continue

        row: dict[str, int | str] = {"madlad_code": lang}
        for source, target in COUNT_COLUMNS.items():
            row[target] = parse_count(raw[source])
        rows.append(row)

    if not rows:
        raise ValueError("MADLAD final dataset table contained no language rows.")
    return rows


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_count(value: str) -> int:
    text = value.strip().replace(",", "")
    if not text:
        return 0

    parts = text.split()
    if len(parts) == 2:
        number, suffix = parts
    else:
        number = text[:-1] if text[-1].isalpha() else text
        suffix = text[-1] if text[-1].isalpha() else ""

    multipliers = {
        "": 1,
        "K": 1_000,
        "M": 1_000_000,
        "G": 1_000_000_000,
        "B": 1_000_000_000,
        "T": 1_000_000_000_000,
    }
    return int(round(float(number) * multipliers[suffix.upper()]))


def write_counts(rows: list[dict[str, int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["madlad_code", *COUNT_COLUMNS.values()]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh MADLAD resource counts from the dataset-card table."
    )
    parser.add_argument("--input", default=MADLAD_CARD_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = parse_madlad_counts(read_text(args.input))
    write_counts(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
