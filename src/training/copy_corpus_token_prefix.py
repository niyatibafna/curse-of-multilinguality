from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
import random
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import count_tokens, read_json, str_to_bool, write_json
from src.training.sample_madlad import load_tokenizer, to_jsonl


def main(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
    target_tokens: int = 500_000_000,
    source_manifest_path: str | None = None,
    strategy: str = "fixed",
    subset: str | None = None,
    seed: int | None = None,
    allow_underfilled: bool = True,
    balance_by_language: bool = False,
    shuffle_output: bool = True,
) -> None:
    tokenizer = load_tokenizer(tokenizer_path)
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = read_json(source_manifest_path) if source_manifest_path else {}
    languages = source_manifest.get("languages")
    if languages is None:
        languages = languages_in_jsonl(source)

    if balance_by_language:
        selected_rows, per_language, actual_tokens, rows = select_balanced_prefix(
            source=source,
            tokenizer=tokenizer,
            languages=languages,
            target_tokens=target_tokens,
        )
    else:
        selected_rows, per_language, actual_tokens, rows = select_global_prefix(
            source=source,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
        )

    if shuffle_output:
        rng = random.Random(seed)
        rng.shuffle(selected_rows)
    write_rows(output, selected_rows)

    if actual_tokens < target_tokens and not allow_underfilled:
        raise RuntimeError(f"{source} ended at {actual_tokens} tokens before target {target_tokens}.")

    target_per_language = max(1, target_tokens // len(languages)) if balance_by_language else 0
    manifest_per_language = {
        language: {
            "rows": per_language.get(language, {}).get("rows", 0),
            "tokens": per_language.get(language, {}).get("tokens", 0),
            "target_tokens": target_per_language,
            "underfilled": balance_by_language and per_language.get(language, {}).get("tokens", 0) < target_per_language,
        }
        for language in languages
    }
    manifest: dict[str, Any] = {
        "strategy": strategy,
        "subset": subset or output.stem,
        "seed": seed,
        "languages": languages,
        "source_jsonl": str(source),
        "source_manifest": source_manifest_path,
        "tokenizer_path": tokenizer_path,
        "balance_by_language": balance_by_language,
        "shuffle_output": shuffle_output,
        "target_tokens_total": target_tokens,
        "actual_tokens_total": actual_tokens,
        "token_deficit": max(0, target_tokens - actual_tokens),
        "underfilled": actual_tokens < target_tokens,
        "underfilled_languages": [
            language
            for language, item in manifest_per_language.items()
            if item["underfilled"]
        ],
        "rows": rows,
        "files": {"jsonl": str(output), "txt": str(output.with_suffix(".txt"))},
        "per_language": manifest_per_language,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(output)
    print(f"target={target_tokens} actual={actual_tokens} deficit={manifest['token_deficit']} rows={rows}")
    if actual_tokens < target_tokens:
        print(f"WARNING: {source} ended before target {target_tokens}.", flush=True)


def select_global_prefix(source: Path, tokenizer: Any, target_tokens: int):
    selected_rows = []
    per_language: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "tokens": 0})
    actual_tokens = 0
    rows = 0
    for row in iter_rows(source):
        language = row["language"]
        text = row["text"]
        tokens = count_tokens(text, tokenizer)
        actual_tokens += tokens
        rows += 1
        per_language[language]["rows"] += 1
        per_language[language]["tokens"] += tokens
        selected_rows.append({"language": language, "text": text})
        if actual_tokens >= target_tokens:
            break
    return selected_rows, per_language, actual_tokens, rows


def select_balanced_prefix(source: Path, tokenizer: Any, languages: list[str], target_tokens: int):
    target_per_language = max(1, target_tokens // len(languages))
    active = set(languages)
    selected_rows = []
    per_language: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "tokens": 0})
    for row in iter_rows(source):
        language = row["language"]
        if language not in active:
            continue
        text = row["text"]
        tokens = count_tokens(text, tokenizer)
        per_language[language]["rows"] += 1
        per_language[language]["tokens"] += tokens
        selected_rows.append({"language": language, "text": text})
        if per_language[language]["tokens"] >= target_per_language:
            active.remove(language)
            if not active:
                break
    actual_tokens = sum(item["tokens"] for item in per_language.values())
    rows = sum(item["rows"] for item in per_language.values())
    return selected_rows, per_language, actual_tokens, rows


def iter_rows(path: Path):
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row["text"]).strip()
            if text:
                yield {"language": row["language"], "text": text}


def languages_in_jsonl(path: Path) -> list[str]:
    seen = set()
    languages = []
    for row in iter_rows(path):
        language = row["language"]
        if language not in seen:
            seen.add(language)
            languages.append(language)
    return languages


def write_rows(output: Path, rows: list[dict[str, str]]) -> None:
    with output.open("w") as jsonl_handle, output.with_suffix(".txt").open("w") as txt_handle:
        for row in rows:
            jsonl_handle.write(to_jsonl(row))
            txt_handle.write(row["text"].replace("\n", " ") + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--target_tokens", type=int, default=500_000_000)
    parser.add_argument("--source_manifest_path")
    parser.add_argument("--strategy", default="fixed")
    parser.add_argument("--subset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow_underfilled", type=str_to_bool, default=True)
    parser.add_argument("--balance_by_language", type=str_to_bool, default=False)
    parser.add_argument("--shuffle_output", type=str_to_bool, default=True)
    main(**vars(parser.parse_args()))
