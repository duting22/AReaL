"""Collect a running paired experiment; never launch another generation attempt."""

import argparse
import contextlib
import io
import json
import time
from collections import Counter
from pathlib import Path

from examples.blender_overfit.control import Evaluation
from examples.blender_overfit.report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--site-base", required=True)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--direct-api", action="store_true")
    args = parser.parse_args()
    while True:
        operation = Evaluation(args)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                operation.status()
                operation.download()
                operation.replay_deliveries()
                summary = build_report(args.dataset, args.run_dir, args.site_base)
            counts = {
                mode: dict(
                    Counter(
                        item["status"]
                        for item in operation.state["attempts"]
                        if item.get("include_in_comparison") and item["mode"] == mode
                    )
                )
                for mode in ("sft", "agent")
            }
            print(
                json.dumps(
                    {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "counts": counts,
                        "scores": summary["modes"],
                        "paired": summary["paired"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if all(
                value["scored"] == summary["task_count"]
                for value in summary["modes"].values()
            ):
                print("COMPLETED_PAIRED_EVALUATION", flush=True)
                return
        finally:
            operation._lock.close()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
