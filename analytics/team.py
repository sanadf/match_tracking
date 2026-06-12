"""Appearance-based team assignment helpers."""

from __future__ import annotations

from math import sqrt
from typing import Any

from .models import FIELD_PERSON_LABELS, Detection


class TeamAssigner:
    """Maintain two coarse color prototypes and assign players to A/B."""

    def __init__(self, distance_threshold: float = 42.0) -> None:
        self.distance_threshold = distance_threshold
        self.prototypes: dict[str, tuple[float, float, float]] = {}

    def update(self, frame: Any, detections: list[Detection]) -> list[Detection]:
        for detection in detections:
            if detection.label not in FIELD_PERSON_LABELS:
                continue
            color = _mean_bgr(frame, detection.bbox)
            if color is None:
                detection.team = detection.team or "unknown"
                continue
            team = self._assign_color(color)
            detection.team = team
            self._update_prototype(team, color)
        return detections

    def _assign_color(self, color: tuple[float, float, float]) -> str:
        if not self.prototypes:
            return "A"
        if len(self.prototypes) == 1:
            team, prototype = next(iter(self.prototypes.items()))
            if _distance(color, prototype) > self.distance_threshold:
                return "B" if team == "A" else "A"
            return team
        return min(self.prototypes, key=lambda name: _distance(color, self.prototypes[name]))

    def _update_prototype(self, team: str, color: tuple[float, float, float]) -> None:
        previous = self.prototypes.get(team)
        if previous is None:
            self.prototypes[team] = color
            return
        self.prototypes[team] = tuple(previous[idx] * 0.9 + color[idx] * 0.1 for idx in range(3))


def assign_teams(frame: Any, detections: list[Detection], assigner: TeamAssigner | None = None) -> list[Detection]:
    assigner = assigner or TeamAssigner()
    return assigner.update(frame, detections)


def _mean_bgr(frame: Any, bbox: tuple[int, int, int, int]) -> tuple[float, float, float] | None:
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Use the upper body region when possible; it is usually the jersey.
    upper = crop[: max(1, crop.shape[0] // 2), :]
    mean = upper.reshape(-1, upper.shape[-1]).mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sqrt(sum((left[idx] - right[idx]) ** 2 for idx in range(3)))
