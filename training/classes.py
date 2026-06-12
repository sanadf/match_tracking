"""Canonical detector classes and label normalization rules."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

FINAL_CLASS_NAMES = ["player", "ball", "goalkeeper", "referee"]
FINAL_CLASS_TO_ID = {name: idx for idx, name in enumerate(FINAL_CLASS_NAMES)}

SYNONYM_MAP = {
    "ball": "ball",
    "ballon": "ball",
    "football": "ball",
    "footballball": "ball",
    "soccerball": "ball",
    "balloon": "ball",
    "player": "player",
    "joueur": "player",
    "person": "player",
    "footballplayer": "player",
    "soccerplayer": "player",
    "athlete": "player",
    "goalkeeper": "goalkeeper",
    "gardien": "goalkeeper",
    "gardiendebut": "goalkeeper",
    "goalie": "goalkeeper",
    "keeper": "goalkeeper",
    "goalkeeperplayer": "goalkeeper",
    "referee": "referee",
    "arbitre": "referee",
    "ref": "referee",
    "matchofficial": "referee",
    "official": "referee",
}


@dataclass(frozen=True)
class ClassMapping:
    raw: str
    normalized: str
    final_name: Optional[str]
    final_id: Optional[int]


def normalize_label_name(label: object) -> str:
    text = str(label).strip().lower()
    text = text.replace("&", "and")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text)


def map_label(label: object) -> ClassMapping:
    normalized = normalize_label_name(label)
    final_name = SYNONYM_MAP.get(normalized)
    final_id = FINAL_CLASS_TO_ID.get(final_name) if final_name is not None else None
    return ClassMapping(str(label), normalized, final_name, final_id)


def yaml_names_block() -> str:
    lines = ["names:"]
    for idx, name in enumerate(FINAL_CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    return "\n".join(lines)
