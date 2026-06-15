from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import read_json, write_json


def main(
    manifest_path: str,
    output_dir: str,
    report_path: str | None = None,
    allow_missing: bool = False,
) -> None:
    missing = missing_outputs(manifest_path, output_dir)
    report = {
        "manifest_path": manifest_path,
        "output_dir": output_dir,
        "expected": expected_count(manifest_path),
        "missing": len(missing),
        "missing_entries": missing,
    }
    if report_path:
        write_json(report_path, report)
    print(f"expected={report['expected']} missing={report['missing']}")
    if missing and not allow_missing:
        preview = "\n".join(format_missing(row) for row in missing[:20])
        raise RuntimeError(f"Missing {len(missing)} eval outputs.\n{preview}")


def expected_count(manifest_path: str | Path) -> int:
    return len(read_json(manifest_path)["entries"])


def missing_outputs(manifest_path: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    root = Path(output_dir)
    missing = []
    for index, entry in enumerate(manifest["entries"]):
        path = output_path(root, entry)
        if not path.exists():
            missing.append({
                "index": index,
                "strategy": entry["strategy"],
                "subset": entry["subset"],
                "dataset": entry["dataset"],
                "metric": entry["metric"],
                "path": str(path),
            })
    return missing


def output_path(root: Path, entry: dict[str, Any]) -> Path:
    return root / entry["strategy"] / entry["subset"] / entry["dataset"] / f"{entry['metric']}.json"


def format_missing(row: dict[str, Any]) -> str:
    return (
        f"index={row['index']} strategy={row['strategy']} subset={row['subset']} "
        f"dataset={row['dataset']} metric={row['metric']} path={row['path']}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_path")
    parser.add_argument("--allow_missing", action="store_true")
    main(**vars(parser.parse_args()))
