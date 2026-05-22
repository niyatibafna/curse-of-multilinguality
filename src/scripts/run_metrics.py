from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fire
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset
from src.metrics import Anisotropy
from src.models import MODEL_REGISTRY, EmbeddingModel, load_model


METRICS = {
    "anisotropy": Anisotropy,
}


def main(
    models: str | list[str],
    datasets: str | list[str],
    metrics: str | list[str],
    model_type: str = "llama",
    output_dir: str | Path = "outputs",
    split: str = "dev",
    dataset_languages: str | list[str] | None = None,
    eval_languages: str | list[str] | None = None,
    cache_dir: str | None = None,
    max_texts: int | None = None,
    layer: int = -1,
    batch_size: int = 32,
    pooling: str = "last_token",
    device: str | None = None,
    return_details: bool = False,
    normalize: bool = True,
) -> None:
    models = as_list(models)
    datasets = as_list(datasets)
    metrics = as_list(metrics)
    dataset_languages = as_list(dataset_languages)
    eval_languages = as_list(eval_languages)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_choices("model_type", [model_type], MODEL_REGISTRY)
    validate_choices("metrics", metrics, METRICS)

    for dataset_name in datasets:
        texts = load_texts(dataset_name, split, dataset_languages, cache_dir, max_texts)
        languages = select_languages(texts, eval_languages)

        for model_name in models:
            model = load_model(
                model_type,
                model_name_or_path=model_name,
                layer=layer,
                device=device,
            )
            embeddings = embed_texts(model, texts, languages, batch_size, pooling)

            for metric_name in metrics:
                result = compute_metric(metric_name, embeddings, return_details, normalize)
                output_path = output_file(output_dir, model_name, dataset_name, metric_name)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(
                    output_path,
                    {
                        "model": model_name,
                        "model_type": model_type,
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "split": split,
                        "languages": languages,
                        "num_concepts": len(next(iter(embeddings.values()))),
                        "embedding_dim": int(next(iter(embeddings.values())).shape[-1]),
                        "result": result,
                    },
                )
                print(f"Wrote {output_path}")


def load_texts(
    dataset_name: str,
    split: str,
    dataset_languages: list[str] | None,
    cache_dir: str | None,
    max_texts: int | None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(
        dataset_name,
        split=split,
        languages=dataset_languages,
        cache_dir=cache_dir,
    )
    texts = dataset.multiparallel_format()
    if max_texts is not None:
        texts = texts[:max_texts]
    if not texts:
        raise ValueError(f"Dataset '{dataset_name}' returned no texts.")
    return texts


def select_languages(texts: list[dict[str, Any]], requested: list[str] | None) -> list[str]:
    if requested:
        languages = requested
    else:
        languages = sorted(set.intersection(*(set(text["data"]) for text in texts)))

    if not languages:
        raise ValueError("No shared languages found across texts.")

    missing = [
        (text["id"], language)
        for text in texts
        for language in languages
        if language not in text["data"]
    ]
    if missing:
        preview = ", ".join(f"{row_id}:{language}" for row_id, language in missing[:5])
        raise ValueError(f"Requested languages are missing in some texts: {preview}")

    return languages


def embed_texts(
    model: EmbeddingModel,
    texts: list[dict[str, Any]],
    languages: list[str],
    batch_size: int,
    pooling: str,
) -> dict[str, np.ndarray]:
    embeddings = {}
    for language in languages:
        inputs = [text["data"][language] for text in texts]
        encoded = model.encode(inputs, batch_size=batch_size, pooling=pooling)
        embeddings[language] = encoded.detach().cpu().numpy()
    return embeddings


def compute_metric(
    metric_name: str,
    embeddings: dict[str, np.ndarray],
    return_details: bool,
    normalize: bool,
) -> Any:
    first = next(iter(embeddings.values()))
    metric = METRICS[metric_name](
        embeddings,
        num_concepts=first.shape[0],
        num_languages=len(embeddings),
        embedding_dim=first.shape[-1],
        return_details=return_details,
        normalize=normalize,
    )
    return to_jsonable(metric.compute())


def as_list(value: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def validate_choices(name: str, values: list[str], choices: dict[str, Any]) -> None:
    unknown = [value for value in values if value not in choices]
    if unknown:
        valid = ", ".join(sorted(choices))
        raise ValueError(f"Unknown {name} {unknown}. Available: {valid}")


def output_file(output_dir: Path, model_name: str, dataset_name: str, metric_name: str) -> Path:
    model_slug = slugify(model_name)
    return output_dir / model_slug / dataset_name / f"{metric_name}.json"


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    fire.Fire(main)
