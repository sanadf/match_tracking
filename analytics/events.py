"""Simple event inference from ball ownership changes."""

from __future__ import annotations

from math import hypot

from .models import FIELD_PERSON_LABELS, Detection, MatchEvent


class EventDetector:
    """Infer possession-change events from nearest player to ball."""

    def __init__(self, fps: float, max_ball_owner_distance: float = 95.0) -> None:
        self.fps = fps if fps > 0 else 25.0
        self.max_ball_owner_distance = max_ball_owner_distance
        self.previous_owner: int | None = None

    def update(self, frame_index: int, detections: list[Detection]) -> tuple[int | None, list[MatchEvent]]:
        owner = nearest_ball_owner(detections, self.max_ball_owner_distance)
        events: list[MatchEvent] = []
        if owner is not None and self.previous_owner is not None and owner.track_id != self.previous_owner:
            events.append(
                MatchEvent(
                    frame_index=frame_index,
                    timestamp=frame_index / self.fps,
                    event_type="possession_change",
                    team=owner.team,
                    track_id=owner.track_id,
                    description=f"Ball ownership changed to track {owner.track_id}",
                )
            )
        if owner is not None:
            self.previous_owner = owner.track_id
        return (owner.track_id if owner else None), events


def nearest_ball_owner(detections: list[Detection], max_distance: float = 95.0) -> Detection | None:
    balls = [det for det in detections if det.label == "ball"]
    players = [det for det in detections if det.label in FIELD_PERSON_LABELS and det.track_id is not None]
    if not balls or not players:
        return None

    ball = max(balls, key=lambda det: det.confidence)
    best_player: Detection | None = None
    best_distance = max_distance
    for player in players:
        distance = hypot(ball.center[0] - player.center[0], ball.center[1] - player.center[1])
        if distance < best_distance:
            best_player = player
            best_distance = distance
    return best_player
