"""Manual event tagging helper for match videos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag match events while stepping through a video.")
    parser.add_argument("--input", default="video/match.mp4", help="Video to review.")
    parser.add_argument("--output", default="output/stats/manual_events.csv", help="CSV event output path.")
    parser.add_argument("--headless", action="store_true", help="Create an empty event CSV without opening a window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.headless:
        _write_events(output, [])
        return 0

    try:
        import cv2
    except Exception as exc:
        print(f"OpenCV unavailable: {exc}")
        _write_events(output, [])
        return 0

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Could not open {args.input}; writing an empty event CSV.")
        _write_events(output, [])
        return 0

    events: list[dict[str, object]] = []
    frame_index = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print("Keys: p=pass, s=shot, f=foul, c=corner, q=quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Manual event tagger", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        event_type = {ord("p"): "pass", ord("s"): "shot", ord("f"): "foul", ord("c"): "corner"}.get(key)
        if event_type:
            events.append(
                {
                    "frame_index": frame_index,
                    "timestamp": round(frame_index / fps, 3),
                    "event_type": event_type,
                    "description": "manual tag",
                }
            )
        frame_index += 1
    cap.release()
    cv2.destroyAllWindows()
    _write_events(output, events)
    print(f"Wrote {len(events)} events to {output}")
    return 0


def _write_events(path: Path, events: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "timestamp", "event_type", "description"])
        writer.writeheader()
        writer.writerows(events)


if __name__ == "__main__":
    raise SystemExit(main())
