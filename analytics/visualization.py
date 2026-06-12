"""Frame annotation helpers."""

from __future__ import annotations

from typing import Any

from .models import Detection

COLORS = {
    "player": (40, 210, 80),
    "ball": (50, 220, 255),
    "goalkeeper": (255, 160, 40),
    "referee": (80, 80, 240),
}


def draw_frame(
    frame: Any,
    detections: list[Detection],
    *,
    frame_index: int,
    possession: dict[str, float] | None = None,
    status: str = "",
) -> Any:
    try:
        import cv2
    except Exception:
        return frame

    annotated = frame.copy()
    for detection in detections:
        color = COLORS.get(detection.label, (230, 230, 230))
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = _label_text(detection)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - text_height - 8)
        cv2.rectangle(annotated, (x1, top), (x1 + text_width + 8, top + text_height + 8), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, top + text_height + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    overlay_lines = [f"Frame {frame_index}"]
    if status:
        overlay_lines.append(status)
    if possession:
        possession_text = "  ".join(f"{team}: {share * 100:.1f}%" for team, share in sorted(possession.items()))
        overlay_lines.append(f"Possession {possession_text}")
    _draw_overlay(annotated, overlay_lines)
    return annotated


def draw_placeholder(frame: Any, message: str) -> Any:
    try:
        import cv2
    except Exception:
        return frame
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    cv2.putText(
        annotated,
        message,
        (max(20, width // 12), max(40, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _label_text(detection: Detection) -> str:
    parts = [detection.label]
    if detection.track_id is not None:
        parts.append(f"#{detection.track_id}")
    if detection.team:
        parts.append(f"T{detection.team}")
    parts.append(f"{detection.confidence:.2f}")
    return " ".join(parts)


def _draw_overlay(frame: Any, lines: list[str]) -> None:
    try:
        import cv2
    except Exception:
        return
    if not lines:
        return
    line_height = 24
    width = 420
    height = line_height * len(lines) + 12
    cv2.rectangle(frame, (12, 12), (12 + width, 12 + height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (22, 38 + line_height * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
