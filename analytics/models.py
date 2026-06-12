"""Shared data models for the football analytics pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_CLASS_NAMES: tuple[str, ...] = ("player", "ball", "goalkeeper", "referee")
CLASS_ID_TO_NAME: dict[int, str] = dict(enumerate(REQUIRED_CLASS_NAMES))
CLASS_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in CLASS_ID_TO_NAME.items()}
FIELD_PERSON_LABELS: frozenset[str] = frozenset({"player", "goalkeeper"})
TRACKABLE_LABELS: frozenset[str] = frozenset({"player", "goalkeeper", "referee"})
BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class Detection:
    """Single detector result in xyxy pixel coordinates."""

    class_id: int
    label: str
    confidence: float
    bbox: BBox
    track_id: int | None = None
    team: str | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def with_track(self, track_id: int) -> "Detection":
        self.track_id = track_id
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox),
            "track_id": self.track_id,
            "team": self.team,
            "source": self.source,
        }


@dataclass(slots=True)
class MatchEvent:
    frame_index: int
    timestamp: float
    event_type: str
    team: str | None = None
    track_id: int | None = None
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "event_type": self.event_type,
            "team": self.team or "",
            "track_id": self.track_id if self.track_id is not None else "",
            "description": self.description,
        }


@dataclass(slots=True)
class FrameAnalysis:
    frame_index: int
    timestamp: float
    detections: list[Detection]
    events: list[MatchEvent] = field(default_factory=list)


@dataclass(slots=True)
class PipelineResult:
    input_path: Path
    output_path: Path
    stats_dir: Path
    processed_frames: int
    status: str
    summary_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "missing_input", "no_frames", "opencv_unavailable"}
