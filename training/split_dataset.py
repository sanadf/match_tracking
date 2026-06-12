"""Create train/val/test splits for a YOLO dataset."""

from __future__ import annotations

import argparse

from .yolo_dataset import split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a YOLO dataset into train, val, and test folders.")
    parser.add_argument("--dataset", required=True, help="Source dataset root.")
    parser.add_argument("--output", required=True, help="Output dataset root.")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="copy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_dataset(
        args.dataset,
        args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        copy_mode=args.copy_mode,
    )
    print(f"Split dataset written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
