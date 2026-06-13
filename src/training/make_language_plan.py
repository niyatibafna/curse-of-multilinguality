from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bouquet import Bouquet
from src.data.floresplus import FloresPlus
from src.data.wmt24pp import WMT24PP
from src.training.common import as_list, nested_subsets, parse_sizes, shuffled, training_dir, write_json


EVAL_DATASETS = ("bouquet", "floresplus", "wmt24pp")
DATASET_CLASSES = {
    "bouquet": Bouquet,
    "floresplus": FloresPlus,
    "wmt24pp": WMT24PP,
}
MADLAD_COUNTS_PATH = PROJECT_ROOT / "src" / "utils" / "language_sorting" / "madlad_counts.csv"
MADLAD_RESOURCE_COLUMNS = {
    "clean": "clean_bytes",
    "noisy": "noisy_bytes",
    "clean_docs": "clean_docs",
    "clean_sents": "clean_sents",
    "clean_tokens": "clean_tokens",
    "clean_chars": "clean_chars",
    "clean_bytes": "clean_bytes",
    "noisy_docs": "noisy_docs",
    "noisy_sents": "noisy_sents",
    "noisy_tokens": "noisy_tokens",
    "noisy_chars": "noisy_chars",
    "noisy_bytes": "noisy_bytes",
}
LANGUAGE_ALIASES = {
    "ar": {"arb"},
    "arb": {"ar"},
    "cmn": {"zh"},
    "he": {"iw"},
    "id": {"in"},
    "jv": {"jw"},
    "ko": {"kor"},
    "kor": {"ko"},
    "ro": {"mo"},
    "yi": {"ji"},
    "zh": {"cmn"},
}
MANUAL_CODE_MATCHES = {
    "fa": {"pes_Arab"},
    "no": {"nno_Latn", "nob_Latn"},
}


def main(
    dataset_name: str = "allenai/madlad-400",
    output_path: str | None = None,
    languages: str | None = None,
    include_languages: str = "en",
    exclude_languages: str = "",
    prioritize_eval_coverage: bool = True,
    eval_datasets: str = ",".join(EVAL_DATASETS),
    order_by_madlad_resource: bool = False,
    resource_split: str = "clean_bytes",
    num_languages: int = 100,
    sizes: str | None = None,
    seed: int = 13,
) -> None:
    selected = as_list(languages)
    included = as_list(include_languages) or []
    excluded = set(as_list(exclude_languages) or [])
    included = [language for language in included if language not in excluded]

    if selected is None:
        try:
            from datasets import get_dataset_config_names
        except ImportError as exc:
            raise ImportError("Install `datasets` to infer MADLAD400 language configs.") from exc

        configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
        config_set = set(configs)
        missing_included = [language for language in included if language not in config_set]
        if missing_included:
            raise ValueError(f"Included languages are not dataset configs: {missing_included}")
        candidates = shuffled([
            config
            for config in configs
            if config not in excluded and config not in included
        ], seed)
        eval_coverage = {}
        if prioritize_eval_coverage:
            eval_coverage = madlad_eval_coverage(configs, as_list(eval_datasets) or [])
            candidates = sort_by_eval_coverage(candidates, eval_coverage)
        resource_scores = {}
        if order_by_madlad_resource:
            resource_scores = madlad_resource_scores(dataset_name, resource_split)
            candidates = sort_by_resource(candidates, resource_scores, eval_coverage)
        selected = included + candidates[: num_languages - len(included)]
    else:
        eval_coverage = {}
        resource_scores = {}
        selected = [language for language in selected if language not in excluded][:num_languages]

    if len(selected) < num_languages:
        raise ValueError(f"Need {num_languages} languages, found {len(selected)}.")

    group_sizes = parse_sizes(sizes)
    if max(group_sizes) > len(selected):
        raise ValueError("Largest subset size exceeds selected language count.")

    payload = {
        "dataset_name": dataset_name,
        "seed": seed,
        "num_languages": num_languages,
        "languages": selected,
        "subsets": nested_subsets(selected, group_sizes),
        "eval_coverage": {
            language: eval_coverage.get(language, {})
            for language in selected
        },
        "resource_split": resource_split,
        "resource_scores": {
            language: resource_scores.get(language, 0)
            for language in selected
        },
    }
    output = Path(output_path) if output_path else training_dir() / "language_plan.json"
    write_json(output, payload)
    print(output)


def madlad_eval_coverage(
    madlad_languages: list[str],
    dataset_names: list[str],
) -> dict[str, dict[str, list[str]]]:
    eval_languages = load_eval_languages(dataset_names)
    madlad_features = {language: code_features(language) for language in madlad_languages}
    coverage: dict[str, dict[str, list[str]]] = {language: {} for language in madlad_languages}

    for dataset_name, languages in eval_languages.items():
        for eval_language in languages:
            eval_features = code_features(eval_language)
            for madlad_language, features in madlad_features.items():
                if manual_code_match(madlad_language, eval_language) or codes_match(features, eval_features):
                    coverage[madlad_language].setdefault(dataset_name, []).append(eval_language)

    return {
        language: datasets
        for language, datasets in coverage.items()
        if datasets
    }


def load_eval_languages(dataset_names: list[str]) -> dict[str, list[str]]:
    try:
        from datasets import get_dataset_config_names
    except ImportError as exc:
        raise ImportError("Install `datasets` to infer evaluation dataset languages.") from exc

    result = {}
    for dataset_name in dataset_names:
        dataset_cls = DATASET_CLASSES[dataset_name]
        configs = get_dataset_config_names(dataset_cls.hf_dataset_name)
        if dataset_name == "wmt24pp":
            languages = sorted({
                language
                for config in configs
                if dataset_cls._is_language_pair_config(config)
                for language in config.split("-", maxsplit=1)
            })
        else:
            languages = sorted(
                config
                for config in configs
                if dataset_cls._is_language_config(config)
            )
        result[dataset_name] = languages
    return result


def sort_by_eval_coverage(
    languages: list[str],
    coverage: dict[str, dict[str, list[str]]],
) -> list[str]:
    return sorted(
        languages,
        key=lambda language: (
            -len(coverage.get(language, {})),
            -sum(len(matches) for matches in coverage.get(language, {}).values()),
            language,
        ),
    )


def madlad_resource_scores(dataset_name: str, split: str) -> dict[str, int]:
    del dataset_name
    column = madlad_resource_column(split)
    with MADLAD_COUNTS_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or {"madlad_code", column} - set(reader.fieldnames):
            raise ValueError(f"{MADLAD_COUNTS_PATH} must contain madlad_code and {column} columns.")
        return {
            row["madlad_code"].strip(): int(row[column])
            for row in reader
            if row.get("madlad_code") and row.get(column)
        }


def madlad_resource_column(split: str) -> str:
    try:
        return MADLAD_RESOURCE_COLUMNS[split]
    except KeyError as exc:
        choices = ", ".join(sorted(MADLAD_RESOURCE_COLUMNS))
        raise ValueError(f"Unknown MADLAD resource split/key {split!r}. Choices: {choices}") from exc


def sort_by_resource(
    languages: list[str],
    resource_scores: dict[str, int],
    coverage: dict[str, dict[str, list[str]]],
) -> list[str]:
    return sorted(
        languages,
        key=lambda language: (
            language not in coverage,
            -resource_scores.get(language, 0),
            -len(coverage.get(language, {})),
            -sum(len(matches) for matches in coverage.get(language, {}).values()),
            language,
        ),
    )


def codes_match(left: dict[str, set[str]], right: dict[str, set[str]]) -> bool:
    if not left["languages"] or not right["languages"]:
        return False
    if not left["languages"] & right["languages"]:
        return False
    if left["scripts"] and right["scripts"]:
        return bool(left["scripts"] & right["scripts"])
    return True


def manual_code_match(madlad_language: str, eval_language: str) -> bool:
    return eval_language in MANUAL_CODE_MATCHES.get(madlad_language, set())


def code_features(code: str) -> dict[str, set[str]]:
    scripts = explicit_scripts(code)
    return {
        "languages": language_keys(code),
        "scripts": scripts or inferred_scripts(code),
    }


def inferred_scripts(code: str) -> set[str]:
    scripts = langcodes_scripts(code.replace("_", "-"))
    if "Kore" in scripts:
        scripts.add("Hang")
    return scripts


def explicit_scripts(code: str) -> set[str]:
    scripts = set()
    for part in code.replace("-", "_").split("_")[1:]:
        if len(part) == 4 and part[0].isalpha() and part[1:].islower():
            scripts.add(part.title())
    return scripts


def language_keys(code: str) -> set[str]:
    normalized = code.replace("_", "-")
    parts = code.replace("-", "_").split("_")
    keys = {code.lower(), normalized.lower(), parts[0].lower()}
    keys.update(langcodes_keys(normalized))
    keys.update(langcodes_keys(parts[0]))
    for key in list(keys):
        keys.update(LANGUAGE_ALIASES.get(key, set()))
    return {key for key in keys if key}


def langcodes_scripts(code: str) -> set[str]:
    try:
        import langcodes
    except ImportError:
        return set()

    try:
        language = langcodes.Language.get(code)
        maximized = language.maximize()
    except Exception:
        return set()

    scripts = set()
    for item in (language, maximized):
        if item.script:
            scripts.add(item.script.title())
    return scripts


def langcodes_keys(code: str) -> set[str]:
    try:
        import langcodes
    except ImportError:
        return set()

    try:
        language = langcodes.Language.get(code)
        maximized = language.maximize()
    except Exception:
        return set()

    keys: set[str] = set()
    for item in (language, maximized):
        if item.language:
            keys.add(item.language.lower())
        try:
            keys.add(item.to_tag().lower())
        except Exception:
            pass
    return keys


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="allenai/madlad-400")
    parser.add_argument("--output_path")
    parser.add_argument("--languages")
    parser.add_argument("--include_languages", default="en")
    parser.add_argument("--exclude_languages", default="")
    parser.add_argument("--prioritize_eval_coverage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_datasets", default=",".join(EVAL_DATASETS))
    parser.add_argument("--order_by_madlad_resource", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resource_split", default="clean_bytes")
    parser.add_argument("--num_languages", type=int, default=100)
    parser.add_argument("--sizes")
    parser.add_argument("--seed", type=int, default=13)
    main(**vars(parser.parse_args()))
