"""Lightweight centroid tracking for detections."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .models import TRACKABLE_LABELS, Detection


@dataclass(slots=True)
class TrackState:
    track_id: int
    label: str
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    age: int = 0
    lost: int = 0
    team: str | None = None


class CentroidTracker:
    """Assign stable IDs by nearest-neighbor center matching."""

    def __init__(self, max_distance: float = 90.0, max_lost: int = 20) -> None:
        self.max_distance = max_distance
        self.max_lost = max_lost
        self._next_id = 1
        self.tracks: dict[int, TrackState] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        trackable = [det for det in detections if det.label in TRACKABLE_LABELS]
        unmatched_tracks = set(self.tracks)

        for detection in sorted(trackable, key=lambda det: det.area, reverse=True):
            match_id = self._best_match(detection, unmatched_tracks)
            if match_id is None:
                match_id = self._new_track(detection)
            else:
                unmatched_tracks.discard(match_id)
            state = self.tracks[match_id]
            state.center = detection.center
            state.bbox = detection.bbox
            state.label = detection.label
            state.age += 1
            state.lost = 0
            detection.track_id = match_id
            if state.team:
                detection.team = state.team

        for track_id in list(unmatched_tracks):
            state = self.tracks[track_id]
            state.lost += 1
            if state.lost > self.max_lost:
                del self.tracks[track_id]

        return detections

    def remember_team(self, track_id: int | None, team: str | None) -> None:
        if track_id is None or not team:
            return
        if track_id in self.tracks:
            self.tracks[track_id].team = team

    def _best_match(self, detection: Detection, candidates: set[int]) -> int | None:
        best_id: int | None = None
        best_distance = self.max_distance
        for track_id in candidates:
            state = self.tracks[track_id]
            if not _labels_compatible(state.label, detection.label):
                continue
            distance = hypot(detection.center[0] - state.center[0], detection.center[1] - state.center[1])
            if distance < best_distance:
                best_id = track_id
                best_distance = distance
        return best_id

    def _new_track(self, detection: Detection) -> int:
        track_id = self._next_id
        self._next_id += 1
        self.tracks[track_id] = TrackState(
            track_id=track_id,
            label=detection.label,
            center=detection.center,
            bbox=detection.bbox,
        )
        return track_id


def track_detections(detections: list[Detection], tracker: CentroidTracker | None = None) -> list[Detection]:
    tracker = tracker or CentroidTracker()
    return tracker.update(detections)


def _labels_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    return {left, right} <= {"player", "goalkeeper"}
