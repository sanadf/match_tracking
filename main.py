"""Command-line entry point for football match analytics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analytics import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run football video analytics and export annotated video plus stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input match video path.")
    parser.add_argument("--output", required=True, help="Annotated output video path.")
    parser.add_argument("--stats-dir", required=True, help="Directory for CSV/JSON stats exports.")
    parser.add_argument("--detector", choices=["auto", "yolo", "classical", "none"], default="auto")
    parser.add_argument("--model-path", default="models/football_detector_best.pt", help="YOLO weights path.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame processing limit.")
    parser.add_argument("--display", action="store_true", help="Show a live preview window.")
    parser.add_argument("--no-video", action="store_true", help="Export stats without writing annotated video.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        stats_dir=Path(args.stats_dir),
        detector=args.detector,
        model_path=Path(args.model_path) if args.model_path else None,
        confidence=args.confidence,
        max_frames=args.max_frames,
        display=args.display,
        write_video=not args.no_video,
    )
    result = run_pipeline(config)
    payload = {
        "status": result.status,
        "processed_frames": result.processed_frames,
        "output_path": str(result.output_path),
        "stats_dir": str(result.stats_dir),
        "summary_path": str(result.summary_path) if result.summary_path else None,
        "warnings": result.warnings,
    }
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
