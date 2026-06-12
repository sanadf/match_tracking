# Football Match Analytics

Analyze a football match video and export an annotated video plus player, team, possession, and event statistics.

This is no longer a color-threshold-only prototype. The pipeline now uses detector adapters, camera motion compensation, persistent player tracking, crop appearance embeddings, team smoothing, ball context, and automatic event inference. Color hints remain only as legacy/manual fallback context.

## What It Outputs

- Annotated video with stable player IDs, team labels, ball marker, possession state, and recent events.
- `player_stats.csv`
- `team_stats.csv`
- `events.csv`
- `match_report.xlsx` when Excel dependencies are installed.

Default output locations:

- Video: `output/analyzed_match.mp4`
- Stats: `output/stats/`

## Install

Use Python 3.11 if possible.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional YOLO/OCR support:

```bash
python -m pip install -r requirements-ml.txt
```

## Input Video

Put your match video in the `video/` folder, or pass any path with `--input`.

The local sample path used during development is:

```bash
video/faisaly_match.mp4
```

Large videos are ignored by git. Keep them local or use Git LFS if you need to version them.

## Run

OpenCV fallback detector:

```bash
python main.py \
  --input video/faisaly_match.mp4 \
  --output output/analyzed_match.mp4 \
  --stats-dir output/stats \
  --team-one "Al Faisaly" \
  --team-two "Hussein Irbid"
```

Process a short smoke test:

```bash
python main.py --input video/faisaly_match.mp4 --max-frames 300
```

YOLO detector when you have model weights:

```bash
python main.py \
  --detector yolo \
  --model-path models/football_detector_best.pt \
  --input video/faisaly_match.mp4
```

If `models/football_detector_best.pt` exists, `--detector auto` will use it automatically.

Preview while processing:

```bash
python main.py --input video/faisaly_match.mp4 --display
```

## Manual Event Tagging Backup

Automatic event detection is part of the main pipeline. Manual tagging can still be used as a backup:

```bash
python events_tagger.py \
  --input video/faisaly_match.mp4 \
  --output output/manual_events.csv \
  --team "Al Faisaly"
```

Keys:

- `P`: pass
- `S`: shot
- `D`: dribble
- `I`: interception
- `Q`: quit

## Current Analytics Approach

Player tracking:

- Player detections come from YOLO when available, otherwise an OpenCV motion/shape fallback.
- Each player track keeps a persistent ID and remains available for re-identification after temporary loss.
- Matching combines camera-compensated position, motion history, bounding-box overlap, crop appearance embeddings, and size continuity.
- Optional OCR can attach jersey numbers when `pytesseract` is installed and `--enable-jersey-ocr` is used.

Team assignment:

- Two team prototypes are learned from full player crop embeddings.
- Decisions are smoothed per track.
- This is not based mainly on jersey color thresholds.

Camera movement:

- Background features are tracked frame to frame.
- A global affine motion estimate is used to compensate player predictions and distance stats.

Event detection:

- Possession is inferred from player-ball proximity.
- Passes, interceptions, possession changes, dribbles, and shot candidates are inferred from owner transitions, ball speed, direction, and context.

## Project Structure

```text
analytics/              Core pipeline package
  camera.py             Camera motion compensation
  detection.py          YOLO/OpenCV detector adapters
  events.py             Automatic event inference
  reid.py               Player crop embeddings and optional jersey OCR
  stats.py              CSV/XLSX export
  team.py               Team prototype assignment
  tracker.py            Persistent player tracker
  visualization.py      Annotated video overlays
docs/TECHNICAL_PLAN.md  Current structure, weak points, and roadmap
tracking/               Legacy compatibility helpers
utils/                  Legacy drawing/video utilities
main.py                 CLI entrypoint
events_tagger.py        Manual backup tagging tool
```

## Accuracy Notes

The OpenCV fallback makes the project runnable without downloading model weights, but real scouting accuracy needs trained football-specific detection and ReID models. The code is structured so those models can be added through `analytics/detection.py` and `analytics/reid.py` without changing the rest of the pipeline.

## Train A Football Detector

The repo includes a full YOLO-format training pipeline for these final classes:

```text
0 player
1 ball
2 goalkeeper
3 referee
```

It inspects raw labels, normalizes synonyms, reports wrong/extra labels, creates
train/val/test splits when needed, optionally adds motion-blur and lighting
augmentation, trains multiple YOLO candidates, compares metrics, and promotes
the best weights to `models/football_detector_best.pt`.

Install optional ML dependencies:

```bash
python -m pip install -r requirements-ml.txt
```

One-command flow:

```bash
python -m training.full_pipeline \
  --dataset datasets/raw_football \
  --workdir training_runs/football_dataset \
  --augment \
  --models yolo11s.pt yolov8s.pt yolov10s.pt yolo11m.pt yolov8m.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch auto \
  --device 0
```

See [docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md) for step-by-step commands and output details.
