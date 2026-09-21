#!/usr/bin/env python3
"""Single public entry point for the NASA RQ-SAIL workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_nasa import evaluate
from src.attack import build_poison_metadata
from src.dataset import prepare_nasa
from src.rq_sail import train_rq_sail
from src.runtime import load_public_config
from src.task1 import train_task1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nasa_bt1.yaml")
    parser.add_argument(
        "--stage",
        choices=["check", "prepare", "task1", "poison", "update", "evaluate", "all"],
        default="check",
    )
    args = parser.parse_args()

    config = load_public_config(args.config)

    if args.stage == "check":
        print(f"config={Path(args.config).resolve()}")
        print(f"attack={config['attack']['type']}")
        print(f"seed={config['experiment']['seed']}")
        print("stages=prepare -> task1 -> poison -> update -> evaluate")
        return 0

    if args.stage in {"prepare", "all"}:
        print(prepare_nasa(config))

    if args.stage in {"task1", "all"}:
        print(train_task1(config))

    if args.stage in {"poison", "all"}:
        poison_path = build_poison_metadata(config)
        config["data"]["poisoned_metadata_path"] = poison_path
        print(poison_path)

    if args.stage in {"update", "all"}:
        print(train_rq_sail(config))

    if args.stage in {"evaluate", "all"}:
        print(evaluate(config))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
