from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import parse_sizes, read_json, write_json


def main(
    language_plan_path: str,
    clean_bytes_path: str = "src/utils/language_sorting/language_plan_clean_bytes_ordered.csv",
    output_path: str | None = None,
    total_tokens: int = 500_000_000,
    floor_tokens_per_language: int = 10_000,
    sizes: str | None = None,
) -> None:
    plan = read_json(language_plan_path)
    clean_bytes = read_clean_bytes(clean_bytes_path)
    requested_sizes = parse_sizes(sizes) if sizes else subset_sizes(plan)
    subsets: dict[str, dict[str, int]] = {}
    missing: dict[str, list[str]] = {}
    for size in requested_sizes:
        subset = f"n{size}"
        languages = list(plan["subsets"][subset])
        missing_languages = [language for language in languages if language not in clean_bytes]
        if missing_languages:
            missing[subset] = missing_languages
            raise ValueError(f"Missing clean bytes for {subset}: {missing_languages}")
        subsets[subset] = allocate_targets(
            languages=languages,
            clean_bytes=clean_bytes,
            total_tokens=total_tokens,
            floor_tokens_per_language=floor_tokens_per_language,
        )

    payload: dict[str, Any] = {
        "language_plan_path": language_plan_path,
        "clean_bytes_path": clean_bytes_path,
        "total_tokens": total_tokens,
        "floor_tokens_per_language": floor_tokens_per_language,
        "sizes": requested_sizes,
        "missing_clean_bytes": missing,
        "subsets": subsets,
    }
    output = Path(output_path) if output_path else Path(language_plan_path).with_suffix(".resource_token_targets.json")
    write_json(output, payload)
    print(output)
    for subset, targets in subsets.items():
        print(
            f"{subset}: languages={len(targets)} target={sum(targets.values())} "
            f"min={min(targets.values())} max={max(targets.values())}"
        )
    if missing:
        print(f"WARNING: missing clean bytes for {missing}", flush=True)


def read_clean_bytes(path: str | Path) -> dict[str, int]:
    values = {}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            values[row["lang"]] = int(row["clean_bytes"])
    return values


def subset_sizes(plan: dict[str, Any]) -> list[int]:
    return sorted(int(name[1:]) for name in plan["subsets"] if name.startswith("n"))


def allocate_targets(
    languages: list[str],
    clean_bytes: dict[str, int],
    total_tokens: int,
    floor_tokens_per_language: int,
) -> dict[str, int]:
    if floor_tokens_per_language * len(languages) > total_tokens:
        raise ValueError("Floor tokens exceed total token budget.")
    floor_total = floor_tokens_per_language * len(languages)
    remainder = total_tokens - floor_total
    weights = {language: max(0, clean_bytes.get(language, 0)) for language in languages}
    weight_total = sum(weights.values())
    if weight_total == 0:
        return allocate_equal(languages, total_tokens)

    targets = {language: floor_tokens_per_language for language in languages}
    fractional = []
    allocated = 0
    for language in languages:
        exact = remainder * weights[language] / weight_total
        whole = int(exact)
        targets[language] += whole
        allocated += whole
        fractional.append((exact - whole, language))
    leftover = remainder - allocated
    for _, language in sorted(fractional, reverse=True)[:leftover]:
        targets[language] += 1
    return targets


def allocate_equal(languages: list[str], total_tokens: int) -> dict[str, int]:
    base = total_tokens // len(languages)
    extra = total_tokens % len(languages)
    return {
        language: base + (1 if index < extra else 0)
        for index, language in enumerate(languages)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--language_plan_path", required=True)
    parser.add_argument("--clean_bytes_path", default="src/utils/language_sorting/language_plan_clean_bytes_ordered.csv")
    parser.add_argument("--output_path")
    parser.add_argument("--total_tokens", type=int, default=500_000_000)
    parser.add_argument("--floor_tokens_per_language", type=int, default=10_000)
    parser.add_argument("--sizes")
    main(**vars(parser.parse_args()))
