"""Command-line entrypoint for the football analytics pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a football match video and export scouting stats.")
    parser.add_argument("--input", default="video/faisaly_match.mp4", help="Input match video path.")
    parser.add_argument("--output", default="output/analyzed_match.mp4", help="Annotated output video path.")
    parser.add_argument("--stats-dir", default="output/stats", help="Directory for CSV/XLSX stats.")
    parser.add_argument(
        "--detector",
        choices=["auto", "opencv", "yolo"],
        default="auto",
        help="Detection backend. Use yolo with --model-path for best results.",
    )
    parser.add_argument("--model-path", default=None, help="Optional YOLO model path.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames.")
    parser.add_argument("--display", action="store_true", help="Show an OpenCV preview window while processing.")
    parser.add_argument("--team-one", default="Team 1", help="Display name for team cluster 1.")
    parser.add_argument("--team-two", default="Team 2", help="Display name for team cluster 2.")
    parser.add_argument(
        "--max-missing-seconds",
        type=float,
        default=30.0,
        help="How long to keep lost player identities available for re-identification.",
    )
    parser.add_argument("--enable-jersey-ocr", action="store_true", help="Try optional OCR for jersey numbers.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from analytics import PipelineConfig, run_pipeline
    except ModuleNotFoundError as exc:
        missing = exc.name or "a dependency"
        print(f"Missing dependency: {missing}")
        print("Install dependencies with: python3.11 -m pip install -r requirements.txt")
        return 1

    config = PipelineConfig(
        input_video=Path(args.input),
        output_video=Path(args.output),
        stats_dir=Path(args.stats_dir),
        detector=args.detector,
        model_path=args.model_path,
        confidence=args.confidence,
        max_frames=args.max_frames,
        display=args.display,
        team_one=args.team_one,
        team_two=args.team_two,
        max_missing_seconds=args.max_missing_seconds,
        enable_jersey_ocr=args.enable_jersey_ocr,
    )
    try:
        outputs = run_pipeline(config)
    except ModuleNotFoundError as exc:
        missing = exc.name or "a dependency"
        print(f"Missing dependency: {missing}")
        print("Install dependencies with: python3.11 -m pip install -r requirements.txt")
        return 1

    print("Analysis complete.")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
