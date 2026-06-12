"""Normalize labels into player, ball, goalkeeper, and referee classes."""

from __future__ import annotations

import argparse
from pathlib import Path

from .yolo_dataset import clean_dataset, save_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean YOLO labels to the final football detector classes.")
    parser.add_argument("--dataset", required=True, help="Source YOLO dataset root.")
    parser.add_argument("--output", default="data/football_clean", help="Output cleaned dataset root.")
    parser.add_argument("--keep-unknown", action="store_true", help="Fail instead of dropping unknown labels.")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--report", default=None, help="JSON inspection report for the cleaned dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Cleaning labels from {Path(args.dataset)} to {Path(args.output)}", flush=True)
    report = clean_dataset(
        args.dataset,
        args.output,
        drop_unknown=not args.keep_unknown,
        copy_mode=args.copy_mode,
        progress_every=1000,
        logger=lambda message: print(f"[clean_labels] {message}", flush=True),
    )
    report_path = Path(args.report) if args.report else Path(args.output) / "label_cleaning_report.json"
    save_report(report, report_path)
    print(f"Cleaned dataset saved to {Path(args.output)}")
    print(f"Final counts: {dict(report.final_class_counts)}")
    if report.unknown_labels:
        print(f"Remaining unknown labels: {dict(report.unknown_labels)}")
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
