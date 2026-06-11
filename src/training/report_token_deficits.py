from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import read_json, training_dir, write_json


def main(
    manifest_glob: str | None = None,
    output_path: str | None = None,
) -> None:
    pattern = manifest_glob or str(training_dir() / "corpora_v2" / "*" / "*.manifest.json")
    manifests = sorted(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern[1:]))
    rows = []
    for path in manifests:
        manifest = read_json(path)
        target = sum(item["target_tokens"] for item in manifest["per_language"].values())
        actual = sum(item["tokens"] for item in manifest["per_language"].values())
        deficit = max(0, target - actual)
        rows.append({
            "manifest": str(path),
            "strategy": manifest["strategy"],
            "subset": manifest["subset"],
            "languages": len(manifest["languages"]),
            "target_tokens": target,
            "actual_tokens": actual,
            "token_deficit": deficit,
            "underfilled_languages": [
                language
                for language, item in manifest["per_language"].items()
                if item.get("underfilled")
            ],
        })

    output = Path(output_path) if output_path else training_dir() / "corpora_v2" / "token_deficit_report.json"
    write_json(output, rows)
    print(output)
    for row in rows:
        print(
            f"{row['strategy']}/{row['subset']}: "
            f"target={row['target_tokens']} actual={row['actual_tokens']} "
            f"deficit={row['token_deficit']} underfilled={len(row['underfilled_languages'])}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_glob")
    parser.add_argument("--output_path")
    main(**vars(parser.parse_args()))
