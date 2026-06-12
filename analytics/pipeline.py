"""End-to-end football analytics pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .detection import create_detector
from .models import PipelineResult
from .stats import StatsAccumulator
from .team import TeamAssigner
from .tracker import CentroidTracker
from .visualization import draw_frame, draw_placeholder


@dataclass(slots=True)
class PipelineConfig:
    input_path: str | Path
    output_path: str | Path
    stats_dir: str | Path
    detector: str = "auto"
    model_path: str | Path | None = "models/football_detector_best.pt"
    confidence: float = 0.25
    max_frames: int | None = None
    display: bool = False
    write_video: bool = True
    placeholder_size: tuple[int, int] = (1280, 720)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Analyze a match video, write an annotated video, and export stats."""

    input_path = Path(config.input_path).expanduser()
    output_path = Path(config.output_path).expanduser()
    stats_dir = Path(config.stats_dir).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    if config.max_frames is not None and config.max_frames < 0:
        raise ValueError("max_frames must be positive or omitted")
    if not 0.0 <= float(config.confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    try:
        import cv2
    except Exception as exc:
        stats = StatsAccumulator(25.0)
        warnings.append(f"OpenCV is unavailable: {exc}")
        summary_path = stats.export(
            stats_dir,
            input_path=input_path,
            output_path=output_path,
            processed_frames=0,
            status="opencv_unavailable",
            warnings=warnings,
        )
        return PipelineResult(input_path, output_path, stats_dir, 0, "opencv_unavailable", summary_path, warnings)

    if not input_path.exists():
        stats = StatsAccumulator(25.0)
        warnings.append(f"Input video not found: {input_path}")
        _write_placeholder_video(cv2, output_path, config.placeholder_size, "Input video not found")
        summary_path = stats.export(
            stats_dir,
            input_path=input_path,
            output_path=output_path,
            processed_frames=0,
            status="missing_input",
            warnings=warnings,
        )
        return PipelineResult(input_path, output_path, stats_dir, 0, "missing_input", summary_path, warnings)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        stats = StatsAccumulator(25.0)
        warnings.append(f"Could not open input video: {input_path}")
        _write_placeholder_video(cv2, output_path, config.placeholder_size, "Could not open input video")
        summary_path = stats.export(
            stats_dir,
            input_path=input_path,
            output_path=output_path,
            processed_frames=0,
            status="no_frames",
            warnings=warnings,
        )
        return PipelineResult(input_path, output_path, stats_dir, 0, "no_frames", summary_path, warnings)

    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS), default=25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or config.placeholder_size[0])
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or config.placeholder_size[1])
    detector = create_detector(config.detector, config.model_path, float(config.confidence))
    warnings.extend(getattr(detector, "warnings", []))
    tracker = CentroidTracker()
    team_assigner = TeamAssigner()
    stats = StatsAccumulator(fps)
    writer = _make_writer(cv2, output_path, fps, width, height) if config.write_video else None
    if config.write_video and writer is None:
        warnings.append(f"Could not open output writer for {output_path}; stats will still be exported.")

    processed = 0
    status = "ok"
    while True:
        if config.max_frames is not None and processed >= config.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None:
            continue
        processed += 1
        detections = detector.detect(frame)
        tracker.update(detections)
        team_assigner.update(frame, detections)
        for detection in detections:
            tracker.remember_team(detection.track_id, detection.team)
        stats.update(processed - 1, detections)
        annotated = draw_frame(
            frame,
            detections,
            frame_index=processed - 1,
            possession=stats.possession_snapshot(),
            status=f"Detector: {getattr(detector, 'name', config.detector)}",
        )
        if writer is not None:
            writer.write(annotated)
        if config.display:
            cv2.imshow("Football analytics", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if config.display:
        cv2.destroyAllWindows()

    if processed == 0:
        status = "no_frames"
        warnings.append(f"No frames were read from {input_path}")
        _write_placeholder_video(cv2, output_path, config.placeholder_size, "No frames were read")

    summary_path = stats.export(
        stats_dir,
        input_path=input_path,
        output_path=output_path,
        processed_frames=processed,
        status=status,
        warnings=warnings,
    )
    return PipelineResult(input_path, output_path, stats_dir, processed, status, summary_path, warnings)


def _make_writer(cv2: object, output_path: Path, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 25.0, (width, height))
    if writer.isOpened():
        return writer
    writer.release()
    return None


def _write_placeholder_video(cv2: object, output_path: Path, size: tuple[int, int], message: str) -> None:
    width, height = size
    writer = _make_writer(cv2, output_path, 25.0, width, height)
    if writer is None:
        output_path.write_text(message + "\n", encoding="utf-8")
        return
    try:
        import numpy as np
    except Exception:
        writer.release()
        output_path.write_text(message + "\n", encoding="utf-8")
        return
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame = draw_placeholder(frame, message)
    for _ in range(25):
        writer.write(frame)
    writer.release()


def _safe_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default
