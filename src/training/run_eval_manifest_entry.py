from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.run_metrics import main as run_metrics_main
from src.training.common import read_json, str_to_bool, write_json


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
    monolingual_reference_language: str = "eng_Latn",
    monolingual_reference_dataset_language: str | None = None,
    monolingual_reference_model: str | None = None,
    monolingual_reference_model_type: str = "mbert",
    monolingual_reference_pooling: str | None = None,
    monolingual_reference_neighbor_k: int | None = None,
    max_seq_length: int = 128,
    mlm_probability: float = 0.15,
    mask_seed: int = 0,
    fp16: bool = True,
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

    run_entry(
        entries[index],
        index,
        output_dir=output_dir,
        batch_size=batch_size,
        layer=layer,
        pooling=pooling,
        device=device,
        max_texts=max_texts,
        random_baseline_trials=random_baseline_trials,
        random_baseline_seed=random_baseline_seed,
        alignment_batch_size=alignment_batch_size,
        monolingual_reference_language=monolingual_reference_language,
        monolingual_reference_dataset_language=monolingual_reference_dataset_language,
        monolingual_reference_model=monolingual_reference_model,
        monolingual_reference_model_type=monolingual_reference_model_type,
        monolingual_reference_pooling=monolingual_reference_pooling,
        monolingual_reference_neighbor_k=monolingual_reference_neighbor_k,
        max_seq_length=max_seq_length,
        mlm_probability=mlm_probability,
        mask_seed=mask_seed,
        fp16=fp16,
    )


def run_entry(
    entry: dict,
    index: int,
    output_dir: str | Path,
    batch_size: int,
    layer: int,
    pooling: str,
    device: str | None,
    max_texts: int | None,
    random_baseline_trials: int,
    random_baseline_seed: int,
    alignment_batch_size: int,
    monolingual_reference_language: str,
    monolingual_reference_dataset_language: str | None,
    monolingual_reference_model: str | None,
    monolingual_reference_model_type: str,
    monolingual_reference_pooling: str | None,
    monolingual_reference_neighbor_k: int | None,
    max_seq_length: int,
    mlm_probability: float,
    mask_seed: int,
    fp16: bool,
) -> None:
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
        monolingual_reference_language=monolingual_reference_language,
        monolingual_reference_dataset_language=monolingual_reference_dataset_language,
        monolingual_reference_model=monolingual_reference_model,
        monolingual_reference_model_type=monolingual_reference_model_type,
        monolingual_reference_pooling=monolingual_reference_pooling,
        monolingual_reference_neighbor_k=monolingual_reference_neighbor_k,
        max_seq_length=max_seq_length,
        mlm_probability=mlm_probability,
        mask_seed=mask_seed,
        fp16=fp16,
    )
    output_path = Path(output_dir) / strategy / subset / dataset / f"{metric}.json"
    payload = read_json(output_path)
    payload.update({
        "eval_stream": entry.get("eval_stream"),
        "eval_language_source_subset": entry.get("eval_language_source_subset"),
        "requested_subset": subset,
        "size": entry.get("size"),
        "train_languages": entry.get("train_languages"),
    })
    write_json(output_path, payload)
    if metric == "mlm_loss":
        output_metric = mlm_output_metric(entry)
        payload.update({
            "metric": output_metric,
            "base_metric": "mlm_loss",
            "eval_language_mode": entry.get("eval_language_mode"),
        })
        write_json(output_path, payload)
        write_json(output_path.with_name(f"{output_metric}.json"), payload)


def mlm_output_metric(entry: dict) -> str:
    if entry.get("eval_language_mode") == "train_subset":
        return "mlm_loss_all"
    return "mlm_loss_fixed_subset"


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
    parser.add_argument("--monolingual_reference_language", default="eng_Latn")
    parser.add_argument("--monolingual_reference_dataset_language")
    parser.add_argument("--monolingual_reference_model")
    parser.add_argument("--monolingual_reference_model_type", default="mbert")
    parser.add_argument("--monolingual_reference_pooling")
    parser.add_argument("--monolingual_reference_neighbor_k", type=int)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--mask_seed", type=int, default=0)
    parser.add_argument("--fp16", type=str_to_bool, default=True)
    main(**vars(parser.parse_args()))
