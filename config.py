"""Default project configuration."""

from pathlib import Path

DEFAULT_INPUT_VIDEO = Path("video/faisaly_match.mp4")
DEFAULT_OUTPUT_VIDEO = Path("output/analyzed_match.mp4")
DEFAULT_STATS_DIR = Path("output/stats")

TEAM_NAMES = ("Al Faisaly", "Hussein Irbid")

# Legacy color hints are kept for manual experiments only. The main pipeline
# uses crop embeddings, motion history, and context instead of these thresholds.
TEAM_COLORS = {
    "Al Faisaly": (173, 216, 230),
    "Hussein Irbid": (0, 255, 255),
}

DEFAULT_FPS = 30
MAX_MISSING_SECONDS = 30.0
