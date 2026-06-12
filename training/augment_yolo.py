"""Offline augmentation helper for YOLO train images."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .yolo_dataset import IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add simple blur and lighting variants to a YOLO train split.")
    parser.add_argument("--dataset", required=True, help="Prepared YOLO dataset root.")
    parser.add_argument("--motion-blur", type=int, default=1, help="Blur variants per train image.")
    parser.add_argument("--lighting", type=int, default=1, help="Brightness variants per train image.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    image_dir = dataset / "images" / "train"
    label_dir = dataset / "labels" / "train"
    if not image_dir.exists():
        print(f"No train image directory found at {image_dir}; nothing to augment.")
        return 0
    try:
        import cv2
    except Exception as exc:
        print(f"OpenCV unavailable ({exc}); copying no augmentation variants.")
        return 0

    count = 0
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        label_path = label_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for index in range(args.motion_blur):
            variant = cv2.GaussianBlur(image, (5, 5), 0)
            count += _write_variant(cv2, variant, image_path, label_path, f"blur{index}")
        for index in range(args.lighting):
            alpha = 1.0 + 0.12 * (index + 1)
            beta = 12 * (index + 1)
            variant = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            count += _write_variant(cv2, variant, image_path, label_path, f"light{index}")
    print(f"Wrote {count} augmented images.")
    return 0


def _write_variant(cv2: object, image, image_path: Path, label_path: Path, suffix: str) -> int:
    output_image = image_path.with_name(f"{image_path.stem}_{suffix}{image_path.suffix}")
    output_label = label_path.with_name(f"{label_path.stem}_{suffix}.txt")
    if output_image.exists():
        return 0
    cv2.imwrite(str(output_image), image)
    if label_path.exists():
        shutil.copyfile(label_path, output_label)
    else:
        output_label.write_text("", encoding="utf-8")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
