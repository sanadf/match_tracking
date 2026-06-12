# Detector Training Pipeline

The detector training workflow prepares a football dataset, normalizes labels to
the four final classes, trains YOLO variants with transfer learning, compares
metrics, and promotes the best local weights for inference.

## Final Classes

```text
0 player
1 ball
2 goalkeeper
3 referee
```

Known label synonyms are normalized before training:

- `ball`, `football`, `football ball`, `soccer ball`, `balloon` -> `ball`
- `player`, `person`, `football player`, `athlete` -> `player`
- `goalkeeper`, `goalie`, `keeper`, `gardien` -> `goalkeeper`
- `referee`, `ref`, `match official`, `arbitre` -> `referee`

## Expected Dataset Format

```text
dataset/
  data.yaml
  images/
    train/ or unsplit images
  labels/
    train/ or matching unsplit .txt labels
```

If the dataset is not already split, the pipeline can create `train`, `val`, and
`test` folders without modifying the source dataset.

## Inspect Labels

```bash
python -m training.inspect_dataset \
  --dataset datasets/raw_football \
  --report training_runs/raw_dataset_report.json
```

## Clean Labels

```bash
python -m training.clean_labels \
  --dataset datasets/raw_football \
  --output training_runs/football_dataset/cleaned \
  --copy-mode copy
```

Unknown labels are dropped by default and reported. Use `--keep-unknown` to fail
instead of dropping them.

## Split Dataset

```bash
python -m training.split_dataset \
  --dataset training_runs/football_dataset/cleaned \
  --output training_runs/football_dataset/prepared \
  --train 0.8 \
  --val 0.1 \
  --test 0.1
```

## Optional Offline Augmentation

```bash
python -m training.augment_yolo \
  --dataset training_runs/football_dataset/prepared \
  --motion-blur 2 \
  --lighting 1
```

## Train And Compare Models

```bash
python -m pip install -r requirements-ml.txt

python -m training.train_yolo \
  --data training_runs/football_dataset/prepared/data.yaml \
  --models yolo11s.pt yolov8s.pt yolov10s.pt yolo11m.pt yolov8m.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch auto \
  --device 0
```

Generated outputs include per-model run folders, comparison CSV/JSON files, and
promoted local weights at `models/football_detector_best.pt`. These artifacts
are ignored by Git.

## Use The Trained Model

```bash
python main.py \
  --detector yolo \
  --model-path models/football_detector_best.pt \
  --input demo_vid.mp4 \
  --output output/demo_analyzed.mp4 \
  --stats-dir output/stats
```
