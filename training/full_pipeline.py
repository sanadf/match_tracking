"""Prepare a football dataset, then train and compare detector models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .yolo_dataset import clean_dataset, has_split, inspect_dataset, save_report, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect, clean, split, augment, and train football YOLO detectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Raw or cleaned YOLO dataset root.")
    parser.add_argument("--workdir", default="training_runs/football_dataset", help="Prepared dataset/work directory.")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--augment", action="store_true", help="Add offline motion blur and lighting variants to train split.")
    parser.add_argument("--model", action="append", default=[], help="Single model checkpoint to train. Repeatable.")
    parser.add_argument("--models", nargs="+", default=None, help="Model checkpoints to train and compare.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Prepare/check commands without starting training.")
    parser.add_argument("--skip-train", action="store_true", help="Stop after dataset preparation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    workdir = Path(args.workdir).expanduser()
    _log("Football detector full pipeline")
    _log(f"Source dataset: {dataset}")
    _log(f"Workdir: {workdir}")

    if not dataset.exists():
        _log(f"Stopping: dataset path does not exist: {dataset}")
        return 1

    workdir.mkdir(parents=True, exist_ok=True)
    _log("Inspecting source dataset labels...")
    raw_report = inspect_dataset(dataset)
    raw_report_path = workdir / "raw_dataset_report.json"
    save_report(raw_report, raw_report_path)
    _log(f"Source report: {raw_report_path}")
    _log(f"Source final counts: {dict(raw_report.final_class_counts)}")
    if raw_report.unknown_labels:
        _log(f"Source unknown/extra labels: {dict(raw_report.unknown_labels)}")

    cleaned = workdir / "cleaned"
    _log(f"Cleaning labels into final classes at: {cleaned}")
    clean_report = clean_dataset(dataset, cleaned, copy_mode=args.copy_mode, progress_every=1000, logger=_log)
    clean_report_path = cleaned / "label_cleaning_report.json"
    save_report(clean_report, clean_report_path)
    _log(f"Cleaned report: {clean_report_path}")
    _log(f"Cleaned final counts: {dict(clean_report.final_class_counts)}")
    if clean_report.unknown_labels:
        _log(f"Cleaned dataset still has unknown labels: {dict(clean_report.unknown_labels)}")

    if has_split(cleaned):
        prepared = cleaned
        _log("Cleaned dataset already has train/val folders; using it directly.")
    else:
        prepared = workdir / "prepared"
        _log(f"Creating train/val/test split at: {prepared}")
        split_dataset(cleaned, prepared, copy_mode=args.copy_mode)

    data_yaml = prepared / "data.yaml"
    _log(f"Prepared data.yaml: {data_yaml}")
    if not data_yaml.exists():
        _log(f"Stopping: prepared data.yaml was not created: {data_yaml}")
        return 1

    if args.augment:
        _log("Running offline augmentation on train split...")
        augment_cmd = [sys.executable, "-m", "training.augment_yolo", "--dataset", str(prepared)]
        _run_subprocess(augment_cmd)
    else:
        _log("Offline augmentation skipped. Use --augment to enable it.")

    if args.skip_train:
        _log("--skip-train set; dataset preparation complete.")
        return 0

    train_cmd = [
        sys.executable,
        "-u",
        "-m",
        "training.train_yolo",
        "--data",
        str(data_yaml),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
    ]
    if args.device:
        train_cmd.extend(["--device", args.device])
    for model in _resolve_models(args):
        train_cmd.extend(["--model", model])
    if args.dry_run:
        train_cmd.append("--dry-run")

    _log(f"Starting training command: {' '.join(train_cmd)}")
    _run_subprocess(train_cmd)
    _log("Full pipeline complete.")
    return 0


def _resolve_models(args: argparse.Namespace) -> list[str]:
    models: list[str] = []
    if args.models:
        models.extend(args.models)
    models.extend(args.model)
    return models


def _run_subprocess(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _log(message: str) -> None:
    print(f"[full_pipeline] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
