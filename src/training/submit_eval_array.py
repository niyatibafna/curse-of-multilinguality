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
    mlm_eval_script: str | None = None,
    mlm_plot_script: str | None = None,
    version: str | None = None,
    eval_stream: str = "eval-all",
    eval_output_dir: str | None = None,
    plot_output_dir: str | None = None,
    mlm_probability: float = 0.3,
    mlm_eval_language_source_subset: str = "n10",
    mlm_eval_language_mode: str | None = None,
    min_plot_size: int = 10,
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

    inferred_version = version or infer_version(manifest_path)
    stream_suffix = "" if eval_stream == "eval-all" else f"_{eval_stream}"
    resolved_eval_output_dir = eval_output_dir or f"outputs/training_scaling_{inferred_version}{stream_suffix}"
    resolved_plot_output_dir = plot_output_dir or f"misc/results_vis/plots/scaling_{inferred_version}{stream_suffix}"
    resolved_mlm_language_mode = mlm_eval_language_mode or (
        "train_subset" if eval_stream == "eval-all" else "source_subset"
    )

    eval_cmd = [
        "sbatch",
        "--parsable",
        f"--array={start_index}-{end}",
        (
            f"--export=ALL,MANIFEST_PATH={manifest_path},"
            f"EVAL_STREAM={eval_stream},OUTPUT_DIR={resolved_eval_output_dir}"
        ),
    ]
    if dependency:
        eval_cmd.append(f"--dependency={dependency}")
    eval_cmd.append(eval_script)
    eval_job = run(eval_cmd, dry_run)
    print(f"eval_job={eval_job}")
    print(f"eval_array={start_index}-{end}")
    print(f"eval_entries={total}")

    mlm_job = None
    if mlm_eval_script:
        mlm_cmd = [
            "sbatch",
            "--parsable",
            (
                f"--export=ALL,VERSION={inferred_version},"
                f"BASE_MANIFEST_PATH={manifest_path},"
                f"OUTPUT_DIR={resolved_eval_output_dir},"
                f"MLM_PROBABILITY={mlm_probability},"
                f"EVAL_LANGUAGE_SOURCE_SUBSET={mlm_eval_language_source_subset},"
                f"EVAL_LANGUAGE_MODE={resolved_mlm_language_mode}"
            ),
        ]
        if dependency:
            mlm_cmd.append(f"--dependency={dependency}")
        mlm_cmd.append(mlm_eval_script)
        mlm_job = run(mlm_cmd, dry_run)
        print(f"mlm_eval_job={mlm_job}")

        if mlm_plot_script:
            mlm_plot_cmd = [
                "sbatch",
                "--parsable",
                f"--dependency=afterok:{mlm_job}",
                (
                    f"--export=ALL,VERSION={inferred_version},"
                    f"INPUT_DIR={resolved_eval_output_dir},"
                    f"OUTPUT_DIR=misc/results_vis/plots/mlm_loss_scaling/{inferred_version}{stream_suffix},"
                    f"MIN_SIZE={min_plot_size}"
                ),
                mlm_plot_script,
            ]
            mlm_plot_job = run(mlm_plot_cmd, dry_run)
            print(f"mlm_plot_job={mlm_plot_job}")

    if plot_script:
        plot_dependency = f"afterok:{eval_job}"
        if mlm_job:
            plot_dependency = f"{plot_dependency}:{mlm_job}"
        plot_cmd = [
            "sbatch",
            "--parsable",
            f"--dependency={plot_dependency}",
            (
                f"--export=ALL,MIN_SIZE={min_plot_size},"
                f"INPUT_DIR={resolved_eval_output_dir},"
                f"OUTPUT_DIR={resolved_plot_output_dir},"
                f"MANIFEST_PATH={manifest_path},"
                f"EVAL_STREAM={eval_stream}"
            ),
            plot_script,
        ]
        plot_job = run(plot_cmd, dry_run)
        print(f"plot_job={plot_job}")


def run(command: list[str], dry_run: bool) -> str:
    if dry_run:
        print(" ".join(command))
        return "DRY_RUN"
    return subprocess.check_output(command, text=True).strip()


def infer_version(manifest_path: str) -> str:
    stem = Path(manifest_path).stem
    prefix = "eval_manifest_"
    if stem.startswith(prefix):
        return stem[len(prefix):].split("_eval-", 1)[0]
    raise ValueError("Pass --version when it cannot be inferred from manifest_path.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--eval_script", required=True)
    parser.add_argument("--plot_script")
    parser.add_argument("--mlm_eval_script")
    parser.add_argument("--mlm_plot_script")
    parser.add_argument("--version")
    parser.add_argument("--eval_stream", default="eval-all", choices=["eval-all", "eval-subset-n10"])
    parser.add_argument("--eval_output_dir")
    parser.add_argument("--plot_output_dir")
    parser.add_argument("--mlm_probability", type=float, default=0.3)
    parser.add_argument("--mlm_eval_language_source_subset", default="n10")
    parser.add_argument("--mlm_eval_language_mode")
    parser.add_argument("--min_plot_size", type=int, default=10)
    parser.add_argument("--dependency")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--dry_run", action="store_true")
    main(**vars(parser.parse_args()))
