"""Video IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def open_video(path: str | Path) -> Any:
    import cv2

    return cv2.VideoCapture(str(path))


def video_metadata(path: str | Path) -> dict[str, float]:
    cap = open_video(path)
    try:
        return {
            "fps": float(cap.get(5) or 0.0),
            "frame_count": float(cap.get(7) or 0.0),
            "width": float(cap.get(3) or 0.0),
            "height": float(cap.get(4) or 0.0),
        }
    finally:
        cap.release()
