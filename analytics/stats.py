"""Statistics accumulation and export for analyzed matches."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .events import EventDetector
from .models import FIELD_PERSON_LABELS, Detection, MatchEvent


@dataclass
class TrackStats:
    track_id: int
    label: str
    team: str = "unknown"
    frames_seen: int = 0
    possession_frames: int = 0


@dataclass
class StatsAccumulator:
    fps: float
    detection_counts: Counter[str] = field(default_factory=Counter)
    track_stats: dict[int, TrackStats] = field(default_factory=dict)
    team_possession_frames: Counter[str] = field(default_factory=Counter)
    events: list[MatchEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.event_detector = EventDetector(self.fps)

    def update(self, frame_index: int, detections: list[Detection]) -> list[MatchEvent]:
        for detection in detections:
            self.detection_counts[detection.label] += 1
            if detection.track_id is None or detection.label not in FIELD_PERSON_LABELS:
                continue
            stats = self.track_stats.setdefault(
                detection.track_id,
                TrackStats(track_id=detection.track_id, label=detection.label),
            )
            stats.label = detection.label
            stats.team = detection.team or stats.team
            stats.frames_seen += 1

        owner_id, new_events = self.event_detector.update(frame_index, detections)
        if owner_id is not None and owner_id in self.track_stats:
            owner_stats = self.track_stats[owner_id]
            owner_stats.possession_frames += 1
            self.team_possession_frames[owner_stats.team] += 1
        self.events.extend(new_events)
        return new_events

    def possession_snapshot(self) -> dict[str, float]:
        total = sum(self.team_possession_frames.values())
        if total == 0:
            return {}
        return {team: frames / total for team, frames in self.team_possession_frames.items()}

    def export(
        self,
        stats_dir: str | Path,
        *,
        input_path: str | Path,
        output_path: str | Path,
        processed_frames: int,
        status: str,
        warnings: list[str],
    ) -> Path:
        root = Path(stats_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._write_player_stats(root / "player_stats.csv")
        self._write_team_stats(root / "team_stats.csv")
        self._write_events(root / "events.csv")
        self._write_detection_counts(root / "detection_counts.csv")
        summary_path = root / "summary.json"
        summary = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "processed_frames": processed_frames,
            "status": status,
            "warnings": warnings,
            "detection_counts": dict(self.detection_counts),
            "team_possession_frames": dict(self.team_possession_frames),
            "team_possession_share": self.possession_snapshot(),
            "event_count": len(self.events),
            "track_count": len(self.track_stats),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        self._write_excel_report(root / "match_report.xlsx", summary)
        return summary_path

    def _write_player_stats(self, path: Path) -> None:
        rows = sorted(self.track_stats.values(), key=lambda item: item.track_id)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["track_id", "label", "team", "frames_seen", "possession_frames"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "track_id": row.track_id,
                        "label": row.label,
                        "team": row.team,
                        "frames_seen": row.frames_seen,
                        "possession_frames": row.possession_frames,
                    }
                )

    def _write_team_stats(self, path: Path) -> None:
        frames_by_team: defaultdict[str, int] = defaultdict(int)
        for stats in self.track_stats.values():
            frames_by_team[stats.team] += stats.frames_seen
        total_possession = sum(self.team_possession_frames.values())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["team", "player_frames", "possession_frames", "possession_share"],
            )
            writer.writeheader()
            for team in sorted(set(frames_by_team) | set(self.team_possession_frames)):
                possession_frames = self.team_possession_frames[team]
                share = possession_frames / total_possession if total_possession else 0.0
                writer.writerow(
                    {
                        "team": team,
                        "player_frames": frames_by_team[team],
                        "possession_frames": possession_frames,
                        "possession_share": f"{share:.4f}",
                    }
                )

    def _write_events(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["frame_index", "timestamp", "event_type", "team", "track_id", "description"],
            )
            writer.writeheader()
            for event in self.events:
                writer.writerow(event.as_dict())

    def _write_detection_counts(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["label", "count"])
            writer.writeheader()
            for label, count in sorted(self.detection_counts.items()):
                writer.writerow({"label": label, "count": count})

    def _write_excel_report(self, path: Path, summary: dict[str, object]) -> None:
        try:
            import pandas as pd
        except Exception:
            return
        try:
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
                pd.DataFrame([vars(row) for row in self.track_stats.values()]).to_excel(
                    writer,
                    sheet_name="players",
                    index=False,
                )
                pd.DataFrame([event.as_dict() for event in self.events]).to_excel(
                    writer,
                    sheet_name="events",
                    index=False,
                )
        except Exception:
            path.unlink(missing_ok=True)
