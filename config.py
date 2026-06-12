"""Default configuration values for local football analytics runs."""

from __future__ import annotations

from analytics.models import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, REQUIRED_CLASS_NAMES

DEFAULT_INPUT = "demo_vid.mp4"
DEFAULT_OUTPUT = "output/demo_analyzed.mp4"
DEFAULT_STATS_DIR = "output/stats"
DEFAULT_MODEL_PATH = "models/football_detector_best.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_MAX_FRAMES = 300

TEAM_COLORS = {
    "A": (40, 210, 80),
    "B": (255, 160, 40),
    "unknown": (220, 220, 220),
}

__all__ = [
    "CLASS_ID_TO_NAME",
    "CLASS_NAME_TO_ID",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_INPUT",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_OUTPUT",
    "DEFAULT_STATS_DIR",
    "REQUIRED_CLASS_NAMES",
    "TEAM_COLORS",
]
