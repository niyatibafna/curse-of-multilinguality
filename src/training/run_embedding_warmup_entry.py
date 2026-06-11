from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.run_metrics import main as run_metrics_main
from src.training.common import read_json


def main(
    manifest_path: str | None = None,
    index: int | None = None,
    output_dir: str | Path = "outputs/training_scaling/_embedding_warmup",
    batch_size: int = 32,
    layer: int = -1,
    pooling: str = "cls",
    device: str | None = "cuda",
    max_texts: int | None = None,
) -> None:
    if index is None:
        import os

        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        index = int(env_index) if env_index is not None else None
    manifest = read_json(manifest_path or default_manifest_path())
    entries = unique_embedding_entries(manifest["entries"])
    if index is None:
        raise ValueError("Pass --index or set SLURM_ARRAY_TASK_ID.")
    if index < 0 or index >= len(entries):
        raise ValueError(f"Index {index} out of range for {len(entries)} entries.")

    entry = entries[index]
    print(
        "warming embeddings "
        f"index={index} strategy={entry['strategy']} subset={entry['subset']} "
        f"dataset={entry['dataset']} eval_languages={len(entry['eval_languages'])}",
        flush=True,
    )
    run_metrics_main(
        models=entry["checkpoint_path"],
        model_type="mbert",
        model_aliases=entry["subset"],
        datasets=entry["dataset"],
        dataset_splits=entry.get("dataset_split", "dev"),
        metrics="anisotropy",
        output_dir=Path(output_dir) / entry["strategy"],
        dataset_languages=entry["eval_languages"],
        eval_languages=entry["eval_languages"],
        max_texts=max_texts,
        layer=layer,
        batch_size=batch_size,
        pooling=pooling,
        device=device,
    )


def default_manifest_path() -> Path:
    import os

    try:
        datadir = os.environ["DATADIR"]
    except KeyError as exc:
        raise ValueError("Set DATADIR or pass --manifest_path.") from exc
    return Path(datadir) / "projects" / "curse-of-multilinguality" / "training" / "eval_manifest.json"


def unique_embedding_entries(entries: list[dict]) -> list[dict]:
    by_key = {}
    for entry in entries:
        key = (
            entry["strategy"],
            entry["subset"],
            entry["checkpoint_path"],
            entry["dataset"],
            entry.get("dataset_split", "dev"),
            tuple(entry["eval_languages"]),
        )
        by_key.setdefault(key, {
            "strategy": entry["strategy"],
            "size": entry["size"],
            "subset": entry["subset"],
            "checkpoint_path": entry["checkpoint_path"],
            "dataset": entry["dataset"],
            "dataset_split": entry.get("dataset_split", "dev"),
            "train_languages": entry["train_languages"],
            "eval_languages": entry["eval_languages"],
        })
    return sorted(
        by_key.values(),
        key=lambda row: (row["strategy"], row["size"], row["dataset"]),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path")
    parser.add_argument("--index", type=int)
    parser.add_argument("--output_dir", default="outputs/training_scaling/_embedding_warmup")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_texts", type=int)
    main(**vars(parser.parse_args()))
