from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import count_tokens, read_json, str_to_bool, training_dir, write_json


def main(
    language_plan_path: str | None = None,
    strategy: str = "fixed",
    subset: str = "n1",
    output_path: str | None = None,
    dataset_name: str | None = None,
    split: str = "clean",
    text_field: str = "text",
    config_per_language: bool = True,
    trust_remote_code: bool = True,
    fixed_total_tokens: int = 20_000_000,
    additive_tokens_per_language: int = 1_000_000,
    tokenizer_path: str | None = None,
    allow_underfilled: bool = True,
    seed: int = 13,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` to sample MADLAD400.") from exc

    tokenizer = load_tokenizer(tokenizer_path)
    plan_path = Path(language_plan_path) if language_plan_path else training_dir() / "language_plan.json"
    plan = read_json(plan_path)
    dataset = dataset_name or plan["dataset_name"]
    languages = plan["subsets"][subset]

    if strategy == "fixed":
        target_per_language = max(1, fixed_total_tokens // len(languages))
    elif strategy in {"additive", "tokenizer"}:
        target_per_language = additive_tokens_per_language
    else:
        raise ValueError("strategy must be one of: fixed, additive, tokenizer")

    output = Path(output_path) if output_path else default_output(strategy, subset)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset_name": dataset,
        "split": split,
        "strategy": strategy,
        "subset": subset,
        "languages": languages,
        "target_tokens_per_language": target_per_language,
        "seed": seed,
        "files": {"jsonl": str(output), "txt": str(output.with_suffix(".txt"))},
        "per_language": {},
    }

    with output.open("w") as jsonl_handle, output.with_suffix(".txt").open("w") as txt_handle:
        for language in languages:
            token_count = 0
            rows = 0
            stream = load_language_stream(
                dataset_name=dataset,
                language=language,
                split=split,
                config_per_language=config_per_language,
                trust_remote_code=trust_remote_code,
            )
            for row in stream:
                text = str(row[text_field]).strip()
                if not text:
                    continue
                token_count += count_tokens(text, tokenizer)
                jsonl_handle.write(to_jsonl({"language": language, "text": text}))
                txt_handle.write(text.replace("\n", " ") + "\n")
                rows += 1
                if token_count >= target_per_language:
                    break
            if token_count < target_per_language and not allow_underfilled:
                raise RuntimeError(
                    f"{language} ended at {token_count} tokens before target {target_per_language}."
                )
            if token_count < target_per_language:
                print(
                    f"WARNING: {language} ended at {token_count} tokens before "
                    f"target {target_per_language}.",
                    flush=True,
                )
            manifest["per_language"][language] = {
                "rows": rows,
                "tokens": token_count,
                "target_tokens": target_per_language,
                "underfilled": token_count < target_per_language,
            }

    write_json(output.with_suffix(".manifest.json"), manifest)
    print(output)


def load_tokenizer(path: str | None):
    if path is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install `transformers` to count sampled tokens with a tokenizer.") from exc
    return AutoTokenizer.from_pretrained(path)


def load_language_stream(
    dataset_name: str,
    language: str,
    split: str,
    config_per_language: bool,
    trust_remote_code: bool,
):
    from datasets import load_dataset

    if config_per_language:
        return iter(load_dataset(dataset_name, language, split=split, streaming=True, trust_remote_code=trust_remote_code))
    return iter(
        load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=trust_remote_code).filter(
            lambda row: row["language"] == language
        )
    )


def default_output(strategy: str, subset: str) -> Path:
    return training_dir() / "corpora" / strategy / f"{subset}.jsonl"


def to_jsonl(row: dict[str, Any]) -> str:
    import json

    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--language_plan_path")
    parser.add_argument("--strategy", default="fixed")
    parser.add_argument("--subset", default="n1")
    parser.add_argument("--output_path")
    parser.add_argument("--dataset_name")
    parser.add_argument("--split", default="clean")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--config_per_language", type=str_to_bool, default=True)
    parser.add_argument("--trust_remote_code", type=str_to_bool, default=True)
    parser.add_argument("--fixed_total_tokens", type=int, default=20_000_000)
    parser.add_argument("--additive_tokens_per_language", type=int, default=1_000_000)
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--allow_underfilled", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=13)
    main(**vars(parser.parse_args()))
