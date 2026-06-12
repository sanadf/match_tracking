"""Legacy possession helpers."""

from __future__ import annotations

from collections import Counter

from analytics.events import nearest_ball_owner
from analytics.models import Detection


def estimate_possession(detections: list[Detection]) -> str:
    owner = nearest_ball_owner(detections)
    if owner is None:
        return "unknown"
    return owner.team or "unknown"


def possession_share(team_frames: Counter[str]) -> dict[str, float]:
    total = sum(team_frames.values())
    if total == 0:
        return {}
    return {team: frames / total for team, frames in team_frames.items()}
