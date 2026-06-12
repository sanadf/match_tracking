"""Legacy frame tracking facade."""

from __future__ import annotations

from typing import Any

from analytics.detection import Detector
from analytics.models import Detection
from analytics.team import TeamAssigner
from analytics.tracker import CentroidTracker


def track_frame(
    frame: Any,
    detector: Detector,
    tracker: CentroidTracker | None = None,
    team_assigner: TeamAssigner | None = None,
) -> list[Detection]:
    tracker = tracker or CentroidTracker()
    team_assigner = team_assigner or TeamAssigner()
    detections = detector.detect(frame)
    tracker.update(detections)
    team_assigner.update(frame, detections)
    return detections
