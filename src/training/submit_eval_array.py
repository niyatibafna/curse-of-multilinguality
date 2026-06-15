from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import read_json


def main(
    manifest_path: str,
    eval_script: str,
    plot_script: str | None = None,
    dependency: str | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    dry_run: bool = False,
) -> None:
    total = len(read_json(manifest_path)["entries"])
    if total == 0:
        raise ValueError(f"No eval entries in {manifest_path}.")
    end = total - 1 if end_index is None else end_index
    if start_index < 0 or end >= total or start_index > end:
        raise ValueError(f"Invalid eval array range {start_index}-{end} for {total} entries.")

    eval_cmd = ["sbatch", "--parsable", f"--array={start_index}-{end}"]
    if dependency:
        eval_cmd.append(f"--dependency={dependency}")
    eval_cmd.append(eval_script)
    eval_job = run(eval_cmd, dry_run)
    print(f"eval_job={eval_job}")
    print(f"eval_array={start_index}-{end}")
    print(f"eval_entries={total}")

    if plot_script:
        plot_cmd = ["sbatch", "--parsable", f"--dependency=afterok:{eval_job}", plot_script]
        plot_job = run(plot_cmd, dry_run)
        print(f"plot_job={plot_job}")


def run(command: list[str], dry_run: bool) -> str:
    if dry_run:
        print(" ".join(command))
        return "DRY_RUN"
    return subprocess.check_output(command, text=True).strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--eval_script", required=True)
    parser.add_argument("--plot_script")
    parser.add_argument("--dependency")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--dry_run", action="store_true")
    main(**vars(parser.parse_args()))
