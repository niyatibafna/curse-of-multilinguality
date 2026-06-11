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
    output_dir: str | Path = "outputs/training_scaling",
    batch_size: int = 32,
    layer: int = -1,
    pooling: str = "cls",
    device: str | None = None,
    max_texts: int | None = None,
    random_baseline_trials: int = 1,
    random_baseline_seed: int = 0,
    alignment_batch_size: int = 64,
) -> None:
    if index is None:
        import os

        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        index = int(env_index) if env_index is not None else None
    manifest = read_json(manifest_path or default_manifest_path())
    entries = manifest["entries"]
    if index is None:
        raise ValueError("Pass --index or set SLURM_ARRAY_TASK_ID.")
    if index < 0 or index >= len(entries):
        raise ValueError(f"Index {index} out of range for {len(entries)} entries.")

    entry = entries[index]
    strategy = entry["strategy"]
    subset = entry["subset"]
    dataset = entry["dataset"]
    metric = entry["metric"]
    eval_languages = entry["eval_languages"]

    print(
        "running "
        f"index={index} strategy={strategy} subset={subset} "
        f"dataset={dataset} metric={metric} eval_languages={len(eval_languages)}",
        flush=True,
    )
    run_metrics_main(
        models=entry["checkpoint_path"],
        model_type="mbert",
        model_aliases=subset,
        datasets=dataset,
        dataset_splits=entry.get("dataset_split", "dev"),
        metrics=metric,
        output_dir=Path(output_dir) / strategy,
        dataset_languages=eval_languages,
        eval_languages=eval_languages,
        max_texts=max_texts,
        layer=layer,
        batch_size=batch_size,
        pooling=pooling,
        device=device,
        random_baseline_trials=random_baseline_trials,
        random_baseline_seed=random_baseline_seed,
        alignment_batch_size=alignment_batch_size,
    )


def default_manifest_path() -> Path:
    import os

    try:
        datadir = os.environ["DATADIR"]
    except KeyError as exc:
        raise ValueError("Set DATADIR or pass --manifest_path.") from exc
    return Path(datadir) / "projects" / "curse-of-multilinguality" / "training" / "eval_manifest.json"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path")
    parser.add_argument("--index", type=int)
    parser.add_argument("--output_dir", default="outputs/training_scaling")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--device")
    parser.add_argument("--max_texts", type=int)
    parser.add_argument("--random_baseline_trials", type=int, default=1)
    parser.add_argument("--random_baseline_seed", type=int, default=0)
    parser.add_argument("--alignment_batch_size", type=int, default=64)
    main(**vars(parser.parse_args()))
