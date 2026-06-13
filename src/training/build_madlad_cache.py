from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import as_list, count_tokens, read_json, str_to_bool, training_dir, write_json
from src.training.sample_madlad import (
    cache_language_path,
    cache_manifest_path,
    load_language_stream,
    load_tokenizer,
    to_jsonl,
)


def main(
    language_plan_path: str | None = None,
    output_root: str | None = None,
    languages: str | None = None,
    subset: str | None = None,
    language_index: int | None = None,
    dataset_name: str | None = None,
    split: str = "clean",
    text_field: str = "text",
    config_per_language: bool = True,
    trust_remote_code: bool = True,
    tokenizer_path: str | None = None,
    max_tokens_per_language: int = 750_000_000,
    overwrite: bool = False,
    skip_existing: bool = False,
    write_index: bool = True,
) -> None:
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        raise ValueError("Pass --tokenizer_path so cache token counts are explicit.")

    plan = read_json(language_plan_path or training_dir() / "language_plan.json")
    dataset = dataset_name or plan["dataset_name"]
    selected_languages = select_languages(plan, languages, subset)
    if language_index is not None:
        selected_languages = [selected_languages[language_index]]

    root = Path(output_root) if output_root else training_dir() / "madlad_cache"
    root.mkdir(parents=True, exist_ok=True)

    manifests = []
    for language in selected_languages:
        manifest = build_language_cache(
            root=root,
            dataset_name=dataset,
            language=language,
            split=split,
            text_field=text_field,
            config_per_language=config_per_language,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            max_tokens=max_tokens_per_language,
            overwrite=overwrite,
            skip_existing=skip_existing,
        )
        manifests.append(manifest)

    if write_index:
        index_path = root / split / "manifest.json"
        existing = []
        if index_path.exists():
            existing = read_json(index_path)
        by_language = {row["language"]: row for row in existing}
        for row in manifests:
            by_language[row["language"]] = row
        write_json(index_path, sorted(by_language.values(), key=lambda row: row["language"]))
        print(index_path)


def select_languages(plan: dict[str, Any], languages: str | None, subset: str | None) -> list[str]:
    explicit = as_list(languages)
    if explicit:
        return explicit
    if subset:
        return list(plan["subsets"][subset])
    return list(plan["languages"])


def build_language_cache(
    root: Path,
    dataset_name: str,
    language: str,
    split: str,
    text_field: str,
    config_per_language: bool,
    trust_remote_code: bool,
    tokenizer: Any,
    tokenizer_path: str | None,
    max_tokens: int,
    overwrite: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    output = cache_language_path(root, split, language)
    manifest_path = cache_manifest_path(root, split, language)
    if not overwrite and manifest_path.exists() and output.exists():
        manifest = read_json(manifest_path)
        if skip_existing:
            print(output)
            return manifest
        if manifest.get("tokens", 0) >= max_tokens or manifest.get("exhausted"):
            print(output)
            return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    token_count = 0
    rows = 0
    exhausted = True

    stream = load_language_stream(
        dataset_name=dataset_name,
        language=language,
        split=split,
        config_per_language=config_per_language,
        trust_remote_code=trust_remote_code,
    )
    with tmp_output.open("w") as handle:
        for row in stream:
            text = str(row[text_field]).strip()
            if not text:
                continue
            token_count += count_tokens(text, tokenizer)
            handle.write(to_jsonl({"language": language, "text": text}))
            rows += 1
            if token_count >= max_tokens:
                exhausted = False
                break

    tmp_output.replace(output)
    manifest = {
        "dataset_name": dataset_name,
        "split": split,
        "language": language,
        "file": str(output),
        "rows": rows,
        "tokens": token_count,
        "target_tokens": max_tokens,
        "underfilled": token_count < max_tokens,
        "exhausted": exhausted,
        "tokenizer_path": tokenizer_path,
    }
    write_json(manifest_path, manifest)
    print(output)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--language_plan_path")
    parser.add_argument("--output_root")
    parser.add_argument("--languages")
    parser.add_argument("--subset")
    parser.add_argument("--language_index", type=int)
    parser.add_argument("--dataset_name")
    parser.add_argument("--split", default="clean")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--config_per_language", type=str_to_bool, default=True)
    parser.add_argument("--trust_remote_code", type=str_to_bool, default=True)
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--max_tokens_per_language", type=int, default=750_000_000)
    parser.add_argument("--overwrite", type=str_to_bool, default=False)
    parser.add_argument("--skip_existing", type=str_to_bool, default=False)
    parser.add_argument("--write_index", type=str_to_bool, default=True)
    main(**vars(parser.parse_args()))
