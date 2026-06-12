"""Drawing helpers for legacy scripts."""

from __future__ import annotations

from typing import Any

from analytics.models import Detection
from analytics.visualization import draw_frame


def draw_detections(frame: Any, detections: list[Detection], frame_index: int = 0) -> Any:
    return draw_frame(frame, detections, frame_index=frame_index)
