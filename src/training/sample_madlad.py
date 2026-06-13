from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import hashlib
import random

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
    target_tokens_path: str | None = None,
    tokenizer_path: str | None = None,
    allow_underfilled: bool = True,
    shuffle_output: bool = False,
    sample_shuffle_buffer_size: int = 0,
    madlad_cache_root: str | None = None,
    cache_fallback_to_hf: bool = True,
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

    target_by_language = load_target_tokens(target_tokens_path, subset) if target_tokens_path else None
    if target_by_language is not None:
        missing = [language for language in languages if language not in target_by_language]
        if missing:
            raise ValueError(f"Missing target tokens for {subset}: {missing}")
    elif strategy == "fixed":
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
        "target_tokens_per_language": None if target_by_language is not None else target_per_language,
        "target_tokens_path": target_tokens_path,
        "target_tokens_total": sum(target_by_language[language] for language in languages)
        if target_by_language is not None
        else target_per_language * len(languages),
        "seed": seed,
        "sample_shuffle_buffer_size": sample_shuffle_buffer_size,
        "madlad_cache_root": madlad_cache_root,
        "cache_fallback_to_hf": cache_fallback_to_hf,
        "files": {"jsonl": str(output), "txt": str(output.with_suffix(".txt"))},
        "per_language": {},
    }

    with output.open("w") as jsonl_handle, output.with_suffix(".txt").open("w") as txt_handle:
        for language in languages:
            target_tokens = target_by_language[language] if target_by_language is not None else target_per_language
            token_count = 0
            rows = 0
            stream, source, cache_rows, cache_exhausted = load_sample_stream(
                cache_root=madlad_cache_root,
                dataset_name=dataset,
                language=language,
                split=split,
                text_field=text_field,
                config_per_language=config_per_language,
                trust_remote_code=trust_remote_code,
                cache_fallback_to_hf=cache_fallback_to_hf,
                shuffle_buffer_size=sample_shuffle_buffer_size,
                seed=language_seed(seed, language),
            )
            token_count, rows, reached_target = write_until_target(
                stream=stream,
                language=language,
                text_field=text_field,
                tokenizer=tokenizer,
                target_tokens=target_tokens,
                jsonl_handle=jsonl_handle,
                txt_handle=txt_handle,
                token_count=token_count,
                rows=rows,
            )
            source_used = source
            if not reached_target and source == "cache" and cache_fallback_to_hf:
                if cache_exhausted:
                    print(
                        f"WARNING: {language} cache is exhausted at {token_count} tokens before "
                        f"target {target_tokens}; not falling back to HF.",
                        flush=True,
                    )
                elif sample_shuffle_buffer_size > 0:
                    print(
                        f"WARNING: {language} cache ended at {token_count} tokens before "
                        f"target {target_tokens}; not falling back to shuffled HF "
                        "because duplicate-free continuation is not guaranteed.",
                        flush=True,
                    )
                else:
                    print(
                        f"WARNING: {language} cache ended at {token_count} tokens before "
                        f"target {target_tokens}; falling back to HF.",
                        flush=True,
                    )
                    hf_stream = load_language_stream(
                        dataset_name=dataset,
                        language=language,
                        split=split,
                        config_per_language=config_per_language,
                        trust_remote_code=trust_remote_code,
                        shuffle_buffer_size=0,
                        seed=language_seed(seed, language),
                    )
                    if cache_rows:
                        hf_stream = skip_rows(hf_stream, cache_rows)
                    token_count, rows, reached_target = write_until_target(
                        stream=hf_stream,
                        language=language,
                        text_field=text_field,
                        tokenizer=tokenizer,
                        target_tokens=target_tokens,
                        jsonl_handle=jsonl_handle,
                        txt_handle=txt_handle,
                        token_count=token_count,
                        rows=rows,
                    )
                    source_used = "cache+hf"
            if token_count < target_tokens and not allow_underfilled:
                raise RuntimeError(
                    f"{language} ended at {token_count} tokens before target {target_tokens}."
                )
            if token_count < target_tokens:
                print(
                    f"WARNING: {language} ended at {token_count} tokens before "
                    f"target {target_tokens}.",
                    flush=True,
                )
            manifest["per_language"][language] = {
                "rows": rows,
                "tokens": token_count,
                "target_tokens": target_tokens,
                "underfilled": token_count < target_tokens,
                "source": source_used,
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


def load_target_tokens(path: str | None, subset: str) -> dict[str, int] | None:
    if path is None:
        return None
    payload = read_json(path)
    targets = payload.get("subsets", {}).get(subset)
    if targets is None:
        raise ValueError(f"No targets for subset {subset} in {path}.")
    return {language: int(tokens) for language, tokens in targets.items()}


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


def load_sample_stream(
    cache_root: str | None,
    dataset_name: str,
    language: str,
    split: str,
    text_field: str,
    config_per_language: bool,
    trust_remote_code: bool,
    cache_fallback_to_hf: bool,
    shuffle_buffer_size: int,
    seed: int,
) -> tuple[Any, str, int, bool]:
    if cache_root:
        cache_path = cache_language_path(Path(cache_root), split, language)
        if cache_path.exists():
            rows, exhausted = cache_info(cache_path)
            stream = load_cached_language_stream(cache_path, shuffle_buffer_size, seed)
            return stream, "cache", rows, exhausted
        if not cache_fallback_to_hf:
            raise FileNotFoundError(f"Cache missing for {language}: {cache_path}")
        print(f"WARNING: cache missing for {language}: {cache_path}; falling back to HF.", flush=True)
    return (
        load_language_stream(
            dataset_name=dataset_name,
            language=language,
            split=split,
            config_per_language=config_per_language,
            trust_remote_code=trust_remote_code,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        ),
        "hf",
        0,
        False,
    )


def cache_language_path(cache_root: Path, split: str, language: str) -> Path:
    return cache_root / split / f"{language}.jsonl"


def cache_manifest_path(cache_root: Path, split: str, language: str) -> Path:
    return cache_root / split / f"{language}.manifest.json"


def cache_info(cache_path: Path) -> tuple[int, bool]:
    manifest = cache_path.with_suffix(".manifest.json")
    if not manifest.exists():
        return 0, False
    try:
        payload = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return 0, False
    return int(payload.get("rows", 0)), bool(payload.get("exhausted", False))


def load_cached_language_stream(path: Path, shuffle_buffer_size: int, seed: int):
    rows = iter_cached_rows(path)
    if shuffle_buffer_size > 0:
        rows = buffer_shuffle(rows, shuffle_buffer_size, seed)
    return rows


def iter_cached_rows(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def buffer_shuffle(rows, buffer_size: int, seed: int):
    rng = random.Random(seed)
    buffer = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    rng.shuffle(buffer)
    yield from buffer


def skip_rows(rows, num_rows: int):
    for index, row in enumerate(rows):
        if index >= num_rows:
            yield row


def write_until_target(
    stream,
    language: str,
    text_field: str,
    tokenizer: Any | None,
    target_tokens: int,
    jsonl_handle,
    txt_handle,
    token_count: int = 0,
    rows: int = 0,
) -> tuple[int, int, bool]:
    for row in stream:
        text = str(row[text_field]).strip()
        if not text:
            continue
        token_count += count_tokens(text, tokenizer)
        jsonl_handle.write(to_jsonl({"language": language, "text": text}))
        txt_handle.write(text.replace("\n", " ") + "\n")
        rows += 1
        if token_count >= target_tokens:
            return token_count, rows, True
    return token_count, rows, False


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
    parser.add_argument("--target_tokens_path")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--allow_underfilled", type=str_to_bool, default=True)
    parser.add_argument("--shuffle_output", type=str_to_bool, default=False)
    parser.add_argument("--sample_shuffle_buffer_size", type=int, default=0)
    parser.add_argument("--madlad_cache_root")
    parser.add_argument("--cache_fallback_to_hf", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=13)
    main(**vars(parser.parse_args()))
