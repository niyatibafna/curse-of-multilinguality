from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import parse_sizes, training_dir, write_json


def main(
    output_path: str | None = None,
    sizes: str | None = None,
    strategies: str = "fixed,additive",
) -> None:
    entries = []
    for strategy in [item.strip() for item in strategies.split(",") if item.strip()]:
        for size in parse_sizes(sizes):
            subset = f"n{size}"
            entries.append({
                "strategy": strategy,
                "subset": subset,
                "train_file": str(training_dir() / "corpora" / strategy / f"{subset}.jsonl"),
                "output_dir": str(training_dir() / "checkpoints" / strategy / subset),
            })

    output = Path(output_path) if output_path else training_dir() / "training_manifest.json"
    write_json(output, {"experiments": entries})
    print(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path")
    parser.add_argument("--sizes")
    parser.add_argument("--strategies", default="fixed,additive")
    main(**vars(parser.parse_args()))
