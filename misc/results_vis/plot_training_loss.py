from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "slurm_logs" / "training_train_mlm"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "misc" / "results_vis" / "plots" / "training_loss"


def main(
    log_dir: str | Path = DEFAULT_LOG_DIR,
    loss_root: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    job_id: str | None = None,
) -> None:
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl_loss_rows(Path(loss_root)) if loss_root else read_loss_rows(log_dir, job_id)
    if not rows:
        source = loss_root or log_dir
        raise ValueError(f"No loss rows found under {source}.")

    df = pd.DataFrame(rows).sort_values(["strategy", "subset_size", "step"])
    csv_path = output_dir / "training_loss_curves.csv"
    df.to_csv(csv_path, index=False)

    plot_curves(df, output_dir / "training_loss_curves.png")
    plot_curves(df, output_dir / "training_loss_curves.pdf")
    plot_reductions(df, output_dir / "training_loss_reduction.png")
    plot_reductions(df, output_dir / "training_loss_reduction.pdf")
    write_summary(df, output_dir / "training_loss_summary.json")
    print(output_dir)


def read_loss_rows(log_dir: Path, job_id: str | None) -> list[dict[str, Any]]:
    pattern = f"{job_id}_*.out" if job_id else "*.out"
    rows = []
    for path in sorted(log_dir.glob(pattern), key=log_sort_key):
        strategy, subset = read_run_name(path)
        if strategy is None or subset is None:
            continue
        for line in path.read_text(errors="replace").splitlines():
            if not line.startswith("{'loss':"):
                continue
            payload = ast.literal_eval(line)
            rows.append({
                "log_file": path.name,
                "strategy": strategy,
                "subset": subset,
                "subset_size": int(subset[1:]),
                "step": len([row for row in rows if row.get("log_file") == path.name]) + 1,
                "epoch": float(payload["epoch"]),
                "loss": float(payload["loss"]),
                "learning_rate": float(payload["learning_rate"]),
                "grad_norm": float(payload["grad_norm"]),
            })
    return rows


def read_jsonl_loss_rows(loss_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(loss_root.glob("**/training_loss.jsonl"), key=jsonl_sort_key):
        rel = path.relative_to(loss_root)
        if len(rel.parts) < 3:
            continue
        subset = path.parent.name
        strategy = "/".join(rel.parts[:-2])
        subset = path.parent.name
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                rows.append({
                    "log_file": str(path),
                    "strategy": strategy,
                    "subset": subset,
                    "subset_size": int(subset[1:]),
                    "step": int(payload["step"]),
                    "epoch": float(payload["epoch"]),
                    "loss": float(payload["loss"]),
                    "learning_rate": none_or_float(payload.get("learning_rate")),
                    "grad_norm": none_or_float(payload.get("grad_norm")),
                })
    return rows


def jsonl_sort_key(path: Path) -> tuple[str, int]:
    try:
        size = int(path.parent.name[1:])
    except ValueError:
        size = 0
    return (str(path.parent.parent), size)


def none_or_float(value: Any) -> float | None:
    return None if value is None else float(value)


def read_run_name(path: Path) -> tuple[str | None, str | None]:
    pattern = re.compile(r"### Finished training_train_mlm strategy=(\w+) subset=(n\d+) ###")
    for line in reversed(path.read_text(errors="replace").splitlines()):
        match = pattern.match(line.strip())
        if match:
            return match.group(1), match.group(2)
    return None, None


def log_sort_key(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 0


def plot_curves(df: pd.DataFrame, path: Path) -> None:
    strategies = list(df["strategy"].drop_duplicates())
    fig, axes = plt.subplots(1, len(strategies), figsize=(6.5 * len(strategies), 4.5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    for ax, strategy in zip(axes, strategies, strict=True):
        subset = df[df["strategy"] == strategy]
        for name, group in subset.groupby("subset"):
            ax.plot(group["epoch"], group["loss"], label=name, linewidth=1.8)
        ax.set_title(strategy)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    axes[0].set_ylabel("logged training loss")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_reductions(df: pd.DataFrame, path: Path) -> None:
    summary = summarize(df)
    summary["label"] = summary["strategy"] + "/" + summary["subset"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    palette = {
        strategy: color
        for strategy, color in zip(
            summary["strategy"].drop_duplicates(),
            ["#4c78a8", "#f58518", "#54a24b", "#e45756"],
            strict=False,
        )
    }
    colors = summary["strategy"].map(palette)
    ax.bar(summary["label"], summary["loss_reduction_pct"], color=colors)
    ax.set_ylabel("loss reduction (%)")
    ax.set_xlabel("run")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, subset), group in df.groupby(["strategy", "subset"]):
        group = group.sort_values("step")
        first = float(group.iloc[0]["loss"])
        last = float(group.iloc[-1]["loss"])
        rows.append({
            "strategy": strategy,
            "subset": subset,
            "subset_size": int(subset[1:]),
            "first_loss": first,
            "last_loss": last,
            "loss_reduction": first - last,
            "loss_reduction_pct": 100 * (first - last) / first,
            "num_logged_points": len(group),
        })
    return pd.DataFrame(rows).sort_values(["strategy", "subset_size"])


def write_summary(df: pd.DataFrame, path: Path) -> None:
    summary = summarize(df)
    with path.open("w") as handle:
        json.dump(summary.to_dict(orient="records"), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--loss_root")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job_id")
    main(**vars(parser.parse_args()))
