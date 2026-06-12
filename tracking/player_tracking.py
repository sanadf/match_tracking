"""Compatibility wrapper around the analytics centroid tracker."""

from __future__ import annotations

from analytics.models import Detection
from analytics.tracker import CentroidTracker


class PlayerTracker:
    def __init__(self, max_distance: float = 90.0, max_lost: int = 20) -> None:
        self._tracker = CentroidTracker(max_distance=max_distance, max_lost=max_lost)

    def update(self, detections: list[Detection]) -> list[Detection]:
        return self._tracker.update(detections)


def track_players(detections: list[Detection], tracker: PlayerTracker | None = None) -> list[Detection]:
    tracker = tracker or PlayerTracker()
    return tracker.update(detections)
