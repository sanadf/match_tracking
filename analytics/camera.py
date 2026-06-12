"""Camera-motion hooks used by the analytics pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CameraMotion:
    dx: float = 0.0
    dy: float = 0.0
    confidence: float = 0.0


class CameraMotionEstimator:
    """Placeholder estimator that keeps the public API stable."""

    def __init__(self) -> None:
        self.previous_gray: Any | None = None

    def update(self, frame: Any) -> CameraMotion:
        try:
            import cv2
        except Exception:
            return CameraMotion()
        self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return CameraMotion()
