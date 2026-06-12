"""Train and compare YOLO detector variants for football analytics."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from .classes import FINAL_CLASS_NAMES

DEFAULT_MODELS = ["yolo11s.pt", "yolov8s.pt", "yolov10s.pt", "yolo11m.pt", "yolov8m.pt"]


@dataclass
class ModelResult:
    model: str
    status: str
    best_pt: str = ""
    run_dir: str = ""
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    ball_map50_95: float = 0.0
    score: float = 0.0
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate YOLO models for football detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Clean YOLO data.yaml path.")
    parser.add_argument("--model", action="append", default=[], help="Single model checkpoint to train. Repeatable.")
    parser.add_argument("--models", nargs="+", default=None, help="Model checkpoints to train and compare.")
    parser.add_argument("--project", default="runs/detect", help="Ultralytics project directory.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960, help="Higher resolution helps ball detection.")
    parser.add_argument("--batch", default="auto", help="Batch size or 'auto'.")
    parser.add_argument("--device", default=None, help="Ultralytics device value, e.g. 0, cpu, mps.")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--promote-to", default="models/football_detector_best.pt", help="Where to copy the winning best.pt.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned training without starting Ultralytics.")
    parser.add_argument("--no-promote", action="store_true", help="Do not copy the winner to models/football_detector_best.pt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = _resolve_models(args)
    data_path = Path(args.data).expanduser().resolve()
    project = Path(args.project).expanduser().resolve()

    _log("Football detector training")
    _log(f"Data YAML: {data_path}")
    _log(f"Class order: {FINAL_CLASS_NAMES}")
    _log(f"Models: {models}")
    _log(f"Project: {project}")
    _log(f"Epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}, device: {args.device or 'auto'}")

    if not data_path.exists():
        _log(f"Skipping training: data.yaml not found at {data_path}")
        return 1

    if args.dry_run:
        _log("Dry run requested; Ultralytics training will not start.")
        return 0

    try:
        from ultralytics import YOLO
    except Exception as exc:
        _log(f"Skipping training: Ultralytics import failed: {exc}")
        _log("Install with: python -m pip install -r requirements-ml.txt")
        return 1

    project.mkdir(parents=True, exist_ok=True)
    results: list[ModelResult] = []
    for model_name in models:
        results.append(_train_one_model(YOLO, model_name, data_path, project, args))

    comparison_paths = _write_comparison(project, results)
    _log(f"Comparison CSV: {comparison_paths[0]}")
    _log(f"Comparison JSON: {comparison_paths[1]}")

    winner = _best_result(results)
    if winner is None:
        _log("No model completed successfully; nothing to promote.")
        return 1

    _log(f"Best model: {winner.model} score={winner.score:.4f} best_pt={winner.best_pt}")
    if args.no_promote:
        _log("--no-promote was set; leaving best.pt in its run directory.")
        return 0

    promote_to = Path(args.promote_to).expanduser()
    promote_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner.best_pt, promote_to)
    metadata_path = promote_to.parent / "football_detector_best_metadata.json"
    metadata_path.write_text(json.dumps(asdict(winner), indent=2), encoding="utf-8")
    _log(f"Promoted best weights to: {promote_to}")
    _log(f"Promoted metadata to: {metadata_path}")
    return 0


def _train_one_model(YOLO: Any, model_name: str, data_path: Path, project: Path, args: argparse.Namespace) -> ModelResult:
    _log(f"Starting model: {model_name}")
    result = ModelResult(model=model_name, status="failed")
    try:
        model = YOLO(model_name)
    except Exception as exc:
        result.reason = f"could not load checkpoint: {exc}"
        _log(f"Skipping {model_name}: {result.reason}")
        return result

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": _parse_batch(args.batch),
        "patience": args.patience,
        "workers": args.workers,
        "project": str(project),
        "name": _safe_name(model_name),
        "exist_ok": True,
        "pretrained": True,
        "plots": True,
        "close_mosaic": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.55,
        "hsv_v": 0.35,
        "degrees": 3.0,
        "translate": 0.08,
        "scale": 0.65,
        "shear": 1.0,
        "perspective": 0.0008,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.08,
        "copy_paste": 0.05,
    }
    if args.device:
        train_kwargs["device"] = args.device

    _log(f"Calling YOLO({model_name!r}).train(...)")
    _log(f"Train run directory: {project / _safe_name(model_name)}")
    try:
        train_result = model.train(**train_kwargs)
        run_dir = _resolve_run_dir(train_result, model, project, model_name)
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            result.reason = f"training finished but best.pt was not found at {best_pt}"
            _log(f"Skipping validation for {model_name}: {result.reason}")
            return result

        _log(f"Training finished for {model_name}; best weights: {best_pt}")
        _log(f"Validating {model_name} on val split...")
        val_kwargs: dict[str, Any] = {"data": str(data_path), "imgsz": args.imgsz, "split": "val", "plots": True}
        if args.device:
            val_kwargs["device"] = args.device
        metrics = YOLO(str(best_pt)).val(**val_kwargs)
        _populate_metrics(result, metrics)
        result.status = "ok"
        result.best_pt = str(best_pt)
        result.run_dir = str(run_dir)
        result.score = _score_result(result)
        _log(
            f"Metrics for {model_name}: "
            f"precision={result.precision:.4f}, recall={result.recall:.4f}, "
            f"mAP50={result.map50:.4f}, mAP50-95={result.map50_95:.4f}, "
            f"ball_mAP50-95={result.ball_map50_95:.4f}"
        )
    except Exception as exc:
        result.reason = str(exc)
        _log(f"Skipping {model_name}: training or validation failed: {exc}")
    return result


def _resolve_models(args: argparse.Namespace) -> list[str]:
    models: list[str] = []
    if args.models:
        models.extend(args.models)
    models.extend(args.model)
    if not models:
        models = DEFAULT_MODELS.copy()
    seen: set[str] = set()
    unique: list[str] = []
    for model in models:
        if model not in seen:
            unique.append(model)
            seen.add(model)
    return unique


def _parse_batch(value: object) -> object:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        return int(text)
    except ValueError:
        return value


def _resolve_run_dir(train_result: Any, model: Any, project: Path, model_name: str) -> Path:
    save_dir = getattr(train_result, "save_dir", None)
    if save_dir:
        return Path(save_dir)
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        return Path(save_dir)
    return project / _safe_name(model_name)


def _populate_metrics(result: ModelResult, metrics: Any) -> None:
    box = getattr(metrics, "box", None)
    if box is None:
        result.reason = "validation returned no box metrics"
        return
    result.map50_95 = float(getattr(box, "map", 0.0) or 0.0)
    result.map50 = float(getattr(box, "map50", 0.0) or 0.0)
    result.precision = _mean_metric(getattr(box, "p", 0.0))
    result.recall = _mean_metric(getattr(box, "r", 0.0))
    maps = getattr(box, "maps", None)
    if maps is not None and len(maps) > 1:
        result.ball_map50_95 = float(maps[1])


def _mean_metric(value: Any) -> float:
    try:
        import numpy as np

        return float(np.mean(value))
    except Exception:
        try:
            return float(value)
        except Exception:
            return 0.0


def _score_result(result: ModelResult) -> float:
    ball = result.ball_map50_95 if result.ball_map50_95 > 0 else result.map50_95
    return 0.6 * result.map50_95 + 0.2 * result.map50 + 0.2 * ball


def _best_result(results: list[ModelResult]) -> Optional[ModelResult]:
    successful = [result for result in results if result.status == "ok" and result.best_pt]
    return max(successful, key=lambda item: item.score) if successful else None


def _write_comparison(project: Path, results: list[ModelResult]) -> tuple[Path, Path]:
    csv_path = project / "model_comparison.csv"
    json_path = project / "model_comparison.json"
    rows = [asdict(result) for result in results]
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def _safe_name(model_name: str) -> str:
    return Path(model_name).stem.replace(".", "_")


def _log(message: str) -> None:
    print(f"[train_yolo] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
