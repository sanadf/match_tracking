"""Inspect a YOLO dataset and report football class coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .yolo_dataset import inspect_dataset, save_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect YOLO labels for the football detector classes.")
    parser.add_argument("--dataset", required=True, help="Dataset root to inspect.")
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_dataset(args.dataset)
    payload = report.to_dict()
    if args.report:
        save_report(report, args.report)
        print(f"Report saved to {Path(args.report)}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
