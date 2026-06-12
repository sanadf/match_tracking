"""Small appearance-memory helpers for future re-identification upgrades."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AppearanceMemory:
    """Store lightweight per-track descriptors."""

    descriptors: dict[int, tuple[float, ...]] = field(default_factory=dict)

    def update(self, track_id: int | None, descriptor: tuple[float, ...] | None) -> None:
        if track_id is None or descriptor is None:
            return
        previous = self.descriptors.get(track_id)
        if previous is None:
            self.descriptors[track_id] = descriptor
            return
        length = min(len(previous), len(descriptor))
        self.descriptors[track_id] = tuple(previous[idx] * 0.8 + descriptor[idx] * 0.2 for idx in range(length))

    def get(self, track_id: int) -> tuple[float, ...] | None:
        return self.descriptors.get(track_id)
