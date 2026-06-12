"""Manual event tagger kept as a backup to automatic event detection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually tag backup football events from a video.")
    parser.add_argument("--input", default="video/faisaly_match.mp4", help="Input match video path.")
    parser.add_argument("--output", default="output/manual_events.csv", help="CSV file for tagged events.")
    parser.add_argument("--team", default="Team 1", help="Team name to write for manual tags.")
    return parser.parse_args()


def log_event(path: Path, timestamp: float, event_type: str, team: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["timestamp", "event_type", "team"])
        writer.writerow([round(timestamp, 2), event_type, team])


def main() -> int:
    import cv2

    args = parse_args()
    output = Path(args.output)
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0
    print("Press P for pass, S for shot, D for dribble, I for interception, Q to quit.")

    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_index / fps
        cv2.imshow("Manual Event Tagger", frame)
        key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
        if key == ord("p"):
            log_event(output, timestamp, "pass", args.team)
        elif key == ord("s"):
            log_event(output, timestamp, "shot", args.team)
        elif key == ord("d"):
            log_event(output, timestamp, "dribble", args.team)
        elif key == ord("i"):
            log_event(output, timestamp, "interception", args.team)
        elif key == ord("q"):
            break
        frame_index += 1

    capture.release()
    cv2.destroyAllWindows()
    print(f"Manual events saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
