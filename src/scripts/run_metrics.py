from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset
from src.metrics import (
    AlignmentCondition,
    Anisotropy,
    Comness,
    ConceptLanguagePrincipalAngleOverlap,
    ConceptSpaceDimGrowthByConcept,
    ConceptSpaceDimGrowthByLanguage,
    IndividualLanguageConceptDimensionality,
    LanguageSpaceDimGrowthByLanguage,
    LanguageSpaceGrowthByConcepts,
    MonolingualStructureCondition,
)
from src.models import MODEL_REGISTRY, EmbeddingModel, load_model


METRICS = {
    "alignment_condition": AlignmentCondition,
    "alignment_condition_weak_view": AlignmentCondition,
    "anisotropy": Anisotropy,
    "comness": Comness,
    "concept_language_principal_angle_overlap": ConceptLanguagePrincipalAngleOverlap,
    "concept_language_principal_angle_overlap_20": ConceptLanguagePrincipalAngleOverlap,
    "concept_language_principal_angle_overlap_50": ConceptLanguagePrincipalAngleOverlap,
    "concept_language_principal_angle_overlap_90": ConceptLanguagePrincipalAngleOverlap,
    "concept_space_dim_growth_by_concept": ConceptSpaceDimGrowthByConcept,
    "concept_space_dim_growth_by_language": ConceptSpaceDimGrowthByLanguage,
    "individual_concept_dimensionality": IndividualLanguageConceptDimensionality,
    "language_space_dim_growth_by_language": LanguageSpaceDimGrowthByLanguage,
    "language_space_growth_by_concepts": LanguageSpaceGrowthByConcepts,
    "monolingual_structure_condition": MonolingualStructureCondition,
}

METRIC_DEFAULT_KWARGS = {
    "alignment_condition_weak_view": {"negative_view": "weak_view"},
    "concept_language_principal_angle_overlap_20": {"subspace_energy_threshold": 0.2},
    "concept_language_principal_angle_overlap_50": {"subspace_energy_threshold": 0.5},
    "concept_language_principal_angle_overlap_90": {"subspace_energy_threshold": 0.9},
}

DEFAULT_POOLING = {
    "mbert": "cls",
}


def main(
    models: str | list[str],
    datasets: str | list[str],
    metrics: str | list[str],
    model_type: str | None = None,
    model_aliases: str | list[str] | None = None,
    dataset_splits: str | list[str] | None = None,
    output_dir: str | Path = "outputs",
    dataset_languages: str | list[str] | None = None,
    eval_languages: str | list[str] | None = None,
    max_texts: int | None = None,
    layer: int = -1,
    batch_size: int = 32,
    pooling: str | None = None,
    device: str | None = None,
    return_details: bool = False,
    normalize: bool = True,
    random_baseline_trials: int = 1,
    random_baseline_seed: int = 0,
    similarity: str = "cosine",
    negative_view: str | None = None,
    alignment_batch_size: int = 64,
) -> None:
    models = as_list(models)
    model_aliases = as_list(model_aliases)
    datasets = as_list(datasets)
    dataset_splits = as_list(dataset_splits)
    metrics = as_list(metrics)
    dataset_languages = as_list(dataset_languages)
    eval_languages = as_list(eval_languages)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_type is None:
        validate_choices("models", models, MODEL_REGISTRY)
    else:
        validate_choices("model_type", [model_type], MODEL_REGISTRY)
    if model_aliases is not None and len(model_aliases) != len(models):
        raise ValueError("--model_aliases must have the same length as --models.")
    if dataset_splits is not None and len(dataset_splits) != len(datasets):
        raise ValueError("--dataset_splits must have the same length as --datasets.")
    validate_choices("metrics", metrics, METRICS)

    for dataset_index, dataset_name in enumerate(datasets):
        requested_split = dataset_splits[dataset_index] if dataset_splits else None
        log(f"loading dataset={dataset_name}")
        texts, dataset_split = load_texts(dataset_name, dataset_languages, requested_split)
        languages = select_languages(texts, eval_languages or dataset_languages)
        texts = filter_texts_for_languages(texts, languages)
        log(f"loaded {len(texts)} texts with {len(languages)} languages: {', '.join(languages)}")

        for model_index, model_name in enumerate(models):
            output_model_name = (
                model_aliases[model_index]
                if model_aliases is not None
                else model_name
            )
            model_key = model_type or model_name
            model_pooling = resolve_pooling(model_key, pooling)
            model_kwargs = {"layer": layer, "device": device}
            if model_type is not None:
                model_kwargs["model_name_or_path"] = model_name

            cache_metadata = embedding_cache_metadata(
                model_name=model_name,
                model_type=model_key,
                dataset_name=dataset_name,
                split=dataset_split,
                dataset_languages=dataset_languages,
                eval_languages=languages,
                texts=texts,
                layer=layer,
                batch_size=batch_size,
                pooling=model_pooling,
            )
            cache_path = embedding_cache_path(cache_metadata)
            embeddings = read_embedding_cache(cache_path, cache_metadata)
            if embeddings is None:
                lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
                while True:
                    try:
                        lock_path.mkdir(parents=True)
                        break
                    except FileExistsError:
                        log(f"waiting for embedding cache {cache_path}")
                        time.sleep(30)
                        embeddings = read_embedding_cache(cache_path, cache_metadata)
                        if embeddings is not None:
                            break

                if embeddings is None:
                    try:
                        log(f"loading model={model_name} model_type={model_key}")
                        model = load_model(model_key, **model_kwargs)
                        log(
                            f"encoding model={model_name} dataset={dataset_name} "
                            f"pooling={model_pooling}"
                        )
                        embeddings = embed_texts(model, texts, languages, batch_size, model_pooling)
                        write_embedding_cache(cache_path, cache_metadata, embeddings)
                        log(f"wrote embedding cache {cache_path}")
                    finally:
                        if lock_path.exists():
                            lock_path.rmdir()
            else:
                log(f"loaded embedding cache {cache_path}")

            log(f"finished encoding model={model_name} dataset={dataset_name}")
            metric_embeddings = slice_embeddings(embeddings, max_texts)
            if max_texts is not None:
                log(
                    f"using first {len(next(iter(metric_embeddings.values())))} "
                    f"of {len(next(iter(embeddings.values())))} cached concepts"
                )

            for metric_name in metrics:
                log(f"computing metric={metric_name} model={model_name} dataset={dataset_name}")
                extra_metric_kwargs = {
                    "random_baseline_trials": random_baseline_trials,
                    "random_baseline_seed": random_baseline_seed,
                    "similarity": similarity,
                    "alignment_batch_size": alignment_batch_size,
                }
                if negative_view is not None:
                    extra_metric_kwargs["negative_view"] = negative_view

                result = compute_metric(
                    metric_name,
                    metric_embeddings,
                    return_details,
                    normalize,
                    **extra_metric_kwargs,
                )
                output_path = output_file(output_dir, output_model_name, dataset_name, metric_name)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(
                    output_path,
                    {
                        "model": output_model_name,
                        "model_path": model_name,
                        "model_type": model_key,
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "result": result,
                        "split": dataset_split,
                        "languages": len(languages),
                        "num_concepts": len(next(iter(metric_embeddings.values()))),
                        "num_cached_concepts": len(next(iter(embeddings.values()))),
                        "embedding_dim": int(next(iter(metric_embeddings.values())).shape[-1]),
                    },
                )
                log(f"wrote {output_path}")


def load_texts(
    dataset_name: str,
    dataset_languages: list[str] | None,
    split: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    kwargs = {"languages": None}
    if split is not None:
        kwargs["split"] = split
    dataset = load_dataset(dataset_name, **kwargs)
    cache_path = formatted_cache_path(dataset_name, dataset.split)
    texts = read_jsonl(cache_path)
    if texts is None:
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        while True:
            try:
                lock_path.mkdir(parents=True)
                break
            except FileExistsError:
                log(f"waiting for dataset cache {cache_path}")
                time.sleep(30)
                texts = read_jsonl(cache_path)
                if texts is not None:
                    break

        if texts is None:
            try:
                log(f"building dataset cache {cache_path}")
                texts = dataset.multiparallel_format()
                write_jsonl(cache_path, texts)
            finally:
                if lock_path.exists():
                    lock_path.rmdir()

    else:
        log(f"loaded dataset cache {cache_path}")

    if not texts:
        raise ValueError(f"Dataset '{dataset_name}' returned no texts.")
    return texts, dataset.split


def select_languages(texts: list[dict[str, Any]], requested: list[str] | None) -> list[str]:
    if requested:
        available = set().union(*(set(text["data"]) for text in texts))
        languages = [language for language in requested if language in available]
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
        available_rows = filter_texts_for_languages(texts, languages)
        if not available_rows:
            raise ValueError(f"Requested languages are missing in all texts: {preview}")

    return languages


def filter_texts_for_languages(
    texts: list[dict[str, Any]],
    languages: list[str],
) -> list[dict[str, Any]]:
    filtered = [
        {
            "id": text["id"],
            "data": {language: text["data"][language] for language in languages},
            "metadata": text.get("metadata", {}),
        }
        for text in texts
        if all(language in text["data"] for language in languages)
    ]
    if not filtered:
        raise ValueError(
            "No dataset rows contain all requested languages: "
            f"{', '.join(languages)}"
        )
    return filtered


def embed_texts(
    model: EmbeddingModel,
    texts: list[dict[str, Any]],
    languages: list[str],
    batch_size: int,
    pooling: str,
) -> dict[str, np.ndarray]:
    embeddings = {}
    language_iter = tqdm(languages, desc="encoding languages", leave=True)
    for index, language in enumerate(language_iter, start=1):
        log(f"encoding language {index}/{len(languages)}: {language}")
        inputs = [text["data"][language] for text in texts]
        encoded = model.encode(inputs, batch_size=batch_size, pooling=pooling)
        embeddings[language] = as_numpy(encoded)
        log(f"encoded language {index}/{len(languages)}: {language} shape={embeddings[language].shape}")
    return embeddings


def slice_embeddings(
    embeddings: dict[str, np.ndarray],
    max_texts: int | None,
) -> dict[str, np.ndarray]:
    if max_texts is None:
        return embeddings
    return {
        language: language_embeddings[:max_texts]
        for language, language_embeddings in embeddings.items()
    }


def compute_metric(
    metric_name: str,
    embeddings: dict[str, np.ndarray],
    return_details: bool,
    normalize: bool,
    **metric_kwargs: Any,
) -> Any:
    first = next(iter(embeddings.values()))
    metric_kwargs = {
        **METRIC_DEFAULT_KWARGS.get(metric_name, {}),
        **metric_kwargs,
    }
    metric = METRICS[metric_name](
        embeddings,
        num_concepts=first.shape[0],
        num_languages=len(embeddings),
        embedding_dim=first.shape[-1],
        return_details=return_details,
        normalize=normalize,
        **metric_kwargs,
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


def resolve_pooling(model_key: str, pooling: str | None) -> str:
    if pooling is not None:
        return pooling
    return DEFAULT_POOLING.get(model_key, "last_token")


def output_file(output_dir: Path, model_name: str, dataset_name: str, metric_name: str) -> Path:
    model_slug = slugify(model_name)
    return output_dir / model_slug / dataset_name / f"{metric_name}.json"


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def formatted_cache_path(
    dataset_name: str,
    split: str,
) -> Path:
    name = "__".join([slugify(dataset_name), slugify(split), "all"])
    return project_datadir() / "multiparallel" / f"{name}.jsonl"


def embedding_cache_metadata(
    model_name: str,
    model_type: str,
    dataset_name: str,
    split: str,
    dataset_languages: list[str] | None,
    eval_languages: list[str],
    texts: list[dict[str, Any]],
    layer: int,
    batch_size: int,
    pooling: str,
) -> dict[str, Any]:
    return {
        "cache_version": 2,
        "model": model_name,
        "model_type": model_type,
        "dataset": dataset_name,
        "split": split,
        "dataset_languages": dataset_languages,
        "eval_languages": eval_languages,
        "num_texts": len(texts),
        "text_ids_hash": stable_hash([text["id"] for text in texts]),
        "text_data_hash": stable_hash([
            {language: text["data"][language] for language in eval_languages}
            for text in texts
        ]),
        "layer": layer,
        "batch_size": batch_size,
        "pooling": pooling,
    }


def embedding_cache_path(metadata: dict[str, Any]) -> Path:
    key = stable_hash(metadata)
    name = "__".join([
        slugify(metadata["model"]),
        slugify(metadata["model_type"]),
        slugify(metadata["dataset"]),
        slugify(metadata["split"]),
        str(metadata["num_texts"]),
        key[:16],
    ])
    return project_datadir() / "embeddings" / f"{name}.npz"


def project_datadir() -> Path:
    try:
        datadir = os.environ["DATADIR"]
    except KeyError as exc:
        raise ValueError("Set DATADIR before running metrics.") from exc
    return Path(datadir) / "projects" / "curse-of-multilinguality"


def read_embedding_cache(
    path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["__metadata__"].item()))
            if metadata != expected_metadata:
                return None
            return {
                language: np.asarray(data[f"lang_{index}"])
                for index, language in enumerate(metadata["eval_languages"])
            }
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def write_embedding_cache(
    path: Path,
    metadata: dict[str, Any],
    embeddings: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    arrays = {
        f"lang_{index}": embeddings[language]
        for index, language in enumerate(metadata["eval_languages"])
    }
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            __metadata__=json.dumps(metadata, sort_keys=True),
            **arrays,
        )
    tmp_path.replace(path)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp_path.open("w") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    tmp_path.replace(path)


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


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
