"""Color helpers used by legacy scripts."""

from __future__ import annotations

from typing import Any


def mean_bgr(frame: Any, bbox: tuple[int, int, int, int]) -> tuple[float, float, float] | None:
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    values = crop.reshape(-1, crop.shape[-1]).mean(axis=0)
    return float(values[0]), float(values[1]), float(values[2])


def dominant_team_color(frame: Any, bbox: tuple[int, int, int, int]) -> tuple[float, float, float] | None:
    x1, y1, x2, y2 = bbox
    upper_bbox = (x1, y1, x2, y1 + max(1, (y2 - y1) // 2))
    return mean_bgr(frame, upper_bbox)
