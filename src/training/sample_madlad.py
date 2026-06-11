from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import hashlib

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
    shuffle_output: bool = False,
    sample_shuffle_buffer_size: int = 0,
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
        "sample_shuffle_buffer_size": sample_shuffle_buffer_size,
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
                shuffle_buffer_size=sample_shuffle_buffer_size,
                seed=language_seed(seed, language),
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
    if shuffle_output:
        shuffle_jsonl(output, seed)
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
    shuffle_buffer_size: int = 0,
    seed: int = 13,
):
    from datasets import load_dataset

    if config_per_language:
        dataset = load_dataset(dataset_name, language, split=split, streaming=True, trust_remote_code=trust_remote_code)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=trust_remote_code).filter(
            lambda row: row["language"] == language
        )
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    return iter(dataset)


def language_seed(seed: int, language: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{language}".encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def default_output(strategy: str, subset: str) -> Path:
    return training_dir() / "corpora" / strategy / f"{subset}.jsonl"


def to_jsonl(row: dict[str, Any]) -> str:
    import json

    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def shuffle_jsonl(path: Path, seed: int) -> None:
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(path), split="train").shuffle(seed=seed)
    tmp_jsonl = path.with_suffix(path.suffix + ".shuffle_tmp")
    tmp_txt = path.with_suffix(".txt.shuffle_tmp")
    with tmp_jsonl.open("w") as jsonl_handle, tmp_txt.open("w") as txt_handle:
        for row in dataset:
            jsonl_handle.write(to_jsonl({"language": row["language"], "text": row["text"]}))
            txt_handle.write(str(row["text"]).replace("\n", " ") + "\n")
    tmp_jsonl.replace(path)
    tmp_txt.replace(path.with_suffix(".txt"))


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
    parser.add_argument("--shuffle_output", type=str_to_bool, default=False)
    parser.add_argument("--sample_shuffle_buffer_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    main(**vars(parser.parse_args()))
