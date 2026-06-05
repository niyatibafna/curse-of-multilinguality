from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path


UTILS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = UTILS_DIR / "wiki_counts.csv"
WIKISTATS_CSV_URL = (
    "https://wikistats.wmcloud.org/api.php?action=dump&format=csv&table=wikipedias"
)


def download_counts(url: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def write_counts(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["wiki_code", "count"])
        for row in sorted(rows, key=lambda row: row["prefix"]):
            writer.writerow([row["prefix"], int(row["good"] or 0)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh language_sorting/wiki_counts.csv from one Wikistats CSV download."
    )
    parser.add_argument("--url", default=WIKISTATS_CSV_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_counts(download_counts(args.url), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
