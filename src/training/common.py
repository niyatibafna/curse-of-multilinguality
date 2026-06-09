from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


DEFAULT_SIZES = [1, 5, 10, 25, 50, 75, 100]
PROJECT_NAME = "curse-of-multilinguality"


def project_datadir() -> Path:
    try:
        datadir = os.environ["DATADIR"]
    except KeyError as exc:
        raise ValueError("Set DATADIR before running training scripts.") from exc
    return Path(datadir) / "projects" / PROJECT_NAME


def training_dir() -> Path:
    return project_datadir() / "training"


def as_list(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def read_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def parse_sizes(sizes: str | list[int] | None) -> list[int]:
    if sizes is None:
        return DEFAULT_SIZES
    if isinstance(sizes, str):
        return [int(item) for item in sizes.split(",") if item.strip()]
    return list(sizes)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}.")


def nested_subsets(languages: list[str], sizes: list[int]) -> dict[str, list[str]]:
    return {f"n{size}": languages[:size] for size in sizes}


def shuffled(items: list[str], seed: int) -> list[str]:
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def count_tokens(text: str, tokenizer: Any | None = None) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))
