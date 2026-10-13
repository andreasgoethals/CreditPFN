"""Prepare or execute the paired, five-fold context-size experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from src.eval.context_curve import prepare, run_task
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--config", default="config/experiment4/context.yaml")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        path, count = prepare(args.config)
        print(f"Prepared {count} tasks: {path}", flush=True)
    else:
        if args.plan is None or args.task is None:
            parser.error("run requires --plan and --task")
        run_task(args.plan, args.task)


if __name__ == "__main__":
    main()
