from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(input_dir: str | Path) -> None:
    input_path = Path(input_dir)
    count = 0
    for path in sorted(input_path.glob("**/mlm_loss.json")):
        with path.open() as handle:
            payload = json.load(handle)
        metric = output_metric(payload)
        payload["metric"] = metric
        payload.setdefault("base_metric", "mlm_loss")
        output_path = path.with_name(f"{metric}.json")
        write_json(output_path, payload)
        count += 1
    print(f"materialized={count}")


def output_metric(payload: dict[str, Any]) -> str:
    metric = payload.get("metric")
    if metric in {"mlm_loss_all", "mlm_loss_fixed_subset"}:
        return metric
    if payload.get("eval_language_mode") == "train_subset":
        return "mlm_loss_all"
    return "mlm_loss_fixed_subset"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    main(**vars(parser.parse_args()))
