# Football Match Analytics Plan

## Current Structure

- `main.py`: command-line pipeline entry point.
- `analytics/`: detector adapters, tracking, team assignment, event inference, visualization, and exports.
- `training/`: dataset inspection, label cleaning, split creation, augmentation, and YOLO training helpers.
- `tracking/` and `utils/`: compatibility wrappers for older scripts.
- `models/`, `data/`, `runs/`, `output/`, and local videos are ignored because they can be large or generated.

## Final Detector Classes

```text
0 player
1 ball
2 goalkeeper
3 referee
```

The class order is shared by the training utilities and the runtime YOLO
detector. Referees are detected and annotated but excluded from possession
ownership.

## Runtime Pipeline

1. Parse CLI arguments into `PipelineConfig`.
2. Create a detector:
   - YOLO with `models/football_detector_best.pt` when available.
   - OpenCV fallback when weights or optional ML dependencies are absent.
   - No-op detector for explicit smoke tests.
3. Read video frames with OpenCV.
4. Detect players, ball, goalkeepers, and referees.
5. Track field people with centroid matching.
6. Assign coarse teams from jersey color.
7. Estimate ball ownership from nearest tracked player/goalkeeper.
8. Emit possession-change events.
9. Write annotated video and CSV/JSON/XLSX stats.

## Exported Files

- `player_stats.csv`
- `team_stats.csv`
- `events.csv`
- `detection_counts.csv`
- `summary.json`
- `match_report.xlsx` when pandas/openpyxl are installed

## Safety Notes

The runtime path is dependency-tolerant so the repository can be smoke-tested
without committing datasets, videos, generated runs, or model weights. Missing
input video or missing YOLO weights are reported in `summary.json` and on stdout
instead of crashing the command.
