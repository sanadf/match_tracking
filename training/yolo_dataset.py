"""YOLO dataset discovery, parsing, and report helpers."""

from __future__ import annotations

import ast
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from .classes import FINAL_CLASS_NAMES, map_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"valid": "val", "validation": "val"}

__all__ = [
    "DatasetInspection",
    "LabelLine",
    "clean_dataset",
    "discover_images",
    "find_data_yaml",
    "has_split",
    "image_split",
    "inspect_dataset",
    "label_path_for_image",
    "parse_label_file",
    "read_dataset_names",
    "resolve_raw_class_name",
    "save_report",
    "split_dataset",
    "write_data_yaml",
]


@dataclass
class LabelLine:
    class_token: str
    coords: list[str]
    raw: str
    line_number: int


@dataclass
class DatasetInspection:
    root: Path
    data_yaml: Optional[Path]
    names: dict[int, str]
    image_count: int = 0
    label_file_count: int = 0
    labeled_image_count: int = 0
    missing_label_count: int = 0
    empty_label_count: int = 0
    class_token_counts: Counter[str] = field(default_factory=Counter)
    raw_name_counts: Counter[str] = field(default_factory=Counter)
    final_class_counts: Counter[str] = field(default_factory=Counter)
    unknown_labels: Counter[str] = field(default_factory=Counter)
    invalid_lines: list[dict[str, object]] = field(default_factory=list)
    split_image_counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "data_yaml": str(self.data_yaml) if self.data_yaml else None,
            "names": self.names,
            "image_count": self.image_count,
            "label_file_count": self.label_file_count,
            "labeled_image_count": self.labeled_image_count,
            "missing_label_count": self.missing_label_count,
            "empty_label_count": self.empty_label_count,
            "class_token_counts": dict(self.class_token_counts),
            "raw_name_counts": dict(self.raw_name_counts),
            "final_class_counts": dict(self.final_class_counts),
            "unknown_labels": dict(self.unknown_labels),
            "invalid_lines": self.invalid_lines[:100],
            "split_image_counts": self.split_image_counts,
            "examples": dict(self.examples),
        }


def find_data_yaml(root: Path) -> Optional[Path]:
    for candidate in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        path = root / candidate
        if path.exists():
            return path
    matches = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    return matches[0] if matches else None


def read_dataset_names(root: Path, data_yaml: Optional[Path] = None) -> dict[int, str]:
    data_yaml = data_yaml or find_data_yaml(root)
    if data_yaml is None:
        return {}
    try:
        import yaml

        content = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        content = _minimal_yaml_read(data_yaml)

    names = content.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    return {}


def discover_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for_image(image_path: Path, root: Path) -> Path:
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    else:
        parts = ["labels", *parts]
    return root.joinpath(*parts).with_suffix(".txt")


def image_split(image_path: Path) -> Optional[str]:
    parts = image_path.parts
    for split in SPLITS:
        if split in parts:
            return split
    for raw_split, final_split in SPLIT_ALIASES.items():
        if raw_split in parts:
            return final_split
    return None


def parse_label_file(path: Path) -> tuple[list[LabelLine], list[dict[str, object]]]:
    parsed: list[LabelLine] = []
    invalid: list[dict[str, object]] = []
    if not path.exists():
        return parsed, invalid

    for idx, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            invalid.append({"file": str(path), "line": idx, "raw": raw, "reason": "expected class plus 4 bbox values"})
            continue
        class_token = parts[0]
        coords = parts[1:5]
        try:
            [float(value) for value in coords]
        except ValueError:
            invalid.append({"file": str(path), "line": idx, "raw": raw, "reason": "bbox values are not numeric"})
            continue
        parsed.append(LabelLine(class_token, coords, raw, idx))
    return parsed, invalid


def inspect_dataset(root: str | Path) -> DatasetInspection:
    dataset_root = Path(root).resolve()
    data_yaml = find_data_yaml(dataset_root)
    names = read_dataset_names(dataset_root, data_yaml)
    report = DatasetInspection(dataset_root, data_yaml, names)

    images = discover_images(dataset_root)
    report.image_count = len(images)
    report.split_image_counts = dict(Counter(image_split(path) or "unsplit" for path in images))

    label_files_seen: set[Path] = set()
    for image in images:
        label_path = label_path_for_image(image, dataset_root)
        if not label_path.exists():
            report.missing_label_count += 1
            continue
        label_files_seen.add(label_path)
        lines, invalid = parse_label_file(label_path)
        report.invalid_lines.extend(invalid)
        if not lines:
            report.empty_label_count += 1
            continue
        report.labeled_image_count += 1
        for line in lines:
            raw_name = resolve_raw_class_name(line.class_token, names)
            mapped = map_label(raw_name)
            report.class_token_counts[line.class_token] += 1
            report.raw_name_counts[raw_name] += 1
            if mapped.final_name is None:
                report.unknown_labels[raw_name] += 1
                _append_example(report.examples, raw_name, label_path)
            else:
                report.final_class_counts[mapped.final_name] += 1

    extra_label_files = sorted(path for path in dataset_root.rglob("labels/**/*.txt") if path not in label_files_seen)
    report.label_file_count = len(label_files_seen) + len(extra_label_files)
    for label_path in extra_label_files[:100]:
        _append_example(report.examples, "orphan_label_file", label_path)
    return report


def clean_dataset(
    source_root: str | Path,
    output_root: str | Path,
    drop_unknown: bool = True,
    copy_mode: str = "copy",
    progress_every: int = 0,
    logger: Optional[Callable[[str], None]] = None,
) -> DatasetInspection:
    """Copy a YOLO dataset while rewriting labels to the 4 canonical classes.

    This is a public training API used by ``training.clean_labels`` and
    ``training.full_pipeline``. Keep it here so both CLIs share the same label
    normalization path.
    """

    src = Path(source_root).resolve()
    dst = Path(output_root).resolve()
    names = read_dataset_names(src)
    dst.mkdir(parents=True, exist_ok=True)
    _clear_managed_dataset_outputs(dst)
    images = discover_images(src)

    if logger:
        logger(f"Found {len(images)} images to clean.")

    for idx, image in enumerate(images, start=1):
        label_path = label_path_for_image(image, src)
        lines, _ = parse_label_file(label_path)
        split = image_split(image) or "unsplit"
        image_dst = dst / "images" / split / image.name
        label_dst = dst / "labels" / split / f"{image.stem}.txt"
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        _copy_file(image, image_dst, copy_mode)

        cleaned_lines: list[str] = []
        for line in lines:
            raw_name = resolve_raw_class_name(line.class_token, names)
            mapped = map_label(raw_name)
            if mapped.final_id is None:
                if drop_unknown:
                    continue
                raise ValueError(f"Unknown label {raw_name!r} in {label_path}")
            cleaned_lines.append(" ".join([str(mapped.final_id), *line.coords]))
        label_dst.write_text("\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""), encoding="utf-8")
        if logger and progress_every > 0 and idx % progress_every == 0:
            logger(f"Cleaned {idx}/{len(images)} images...")

    write_data_yaml(dst)
    if logger:
        logger(f"Finished cleaning {len(images)} images.")
    return inspect_dataset(dst)


def split_dataset(
    source_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy_mode: str = "copy",
) -> None:
    src = Path(source_root).resolve()
    dst = Path(output_root).resolve()
    images = discover_images(src)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must add up to 1.0")

    rng = random.Random(seed)
    rng.shuffle(images)
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    groups = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split, split_images in groups.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image in split_images:
            label = label_path_for_image(image, src)
            image_dst = dst / "images" / split / image.name
            label_dst = dst / "labels" / split / f"{image.stem}.txt"
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(image, image_dst, copy_mode)
            if label.exists():
                _copy_file(label, label_dst, copy_mode)
            else:
                label_dst.write_text("", encoding="utf-8")
    write_data_yaml(dst)


def has_split(root: str | Path) -> bool:
    dataset_root = Path(root)
    return all((dataset_root / "images" / split).exists() for split in ("train", "val"))


def write_data_yaml(root: str | Path) -> Path:
    dataset_root = Path(root).resolve()
    path = dataset_root / "data.yaml"
    content = [
        f"path: {dataset_root}",
        "train: images/train" if (dataset_root / "images" / "train").exists() else "train: images/unsplit",
        "val: images/val" if (dataset_root / "images" / "val").exists() else "val: images/unsplit",
        "test: images/test" if (dataset_root / "images" / "test").exists() else "test: images/unsplit",
        "nc: 4",
        "names:",
    ]
    for idx, name in enumerate(FINAL_CLASS_NAMES):
        content.append(f"  {idx}: {name}")
    path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return path


def save_report(report: DatasetInspection, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def resolve_raw_class_name(class_token: str, names: dict[int, str]) -> str:
    try:
        class_id = int(float(class_token))
    except ValueError:
        return class_token
    return names.get(class_id, class_token)


def _copy_file(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "symlink":
        try:
            dst.symlink_to(src)
        except OSError:
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copyfile(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copyfile(src, dst)
    elif mode == "copy":
        shutil.copyfile(src, dst)
    else:
        raise ValueError("copy_mode must be copy, symlink, or hardlink")


def _clear_managed_dataset_outputs(dst: Path) -> None:
    for directory in (dst / "images", dst / "labels"):
        if directory.exists():
            shutil.rmtree(directory)
    data_yaml = dst / "data.yaml"
    if data_yaml.exists():
        data_yaml.unlink()


def _append_example(examples: dict[str, list[str]], key: str, path: Path) -> None:
    bucket = examples[key]
    if len(bucket) < 10:
        bucket.append(str(path))


def _minimal_yaml_read(path: Path) -> dict[str, object]:
    content: dict[str, object] = {}
    in_names_block = False
    names: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if in_names_block and line.startswith((" ", "\t")) and ":" in line:
            key, value = line.split(":", 1)
            try:
                names[int(key.strip())] = value.strip().strip("'\"")
            except ValueError:
                pass
            continue
        in_names_block = False
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "names":
            in_names_block = True
            try:
                content[key] = ast.literal_eval(value)
            except Exception:
                content[key] = names
        elif key:
            content[key] = value
    if names:
        content["names"] = names
    return content
