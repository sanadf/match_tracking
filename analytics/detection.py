"""Detector adapters for football objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .models import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, Detection

YOLO_CLASS_NAMES: dict[int, str] = CLASS_ID_TO_NAME.copy()


class Detector(Protocol):
    name: str
    warnings: list[str]

    def detect(self, frame: Any) -> list[Detection]:
        """Return detections for a single BGR frame."""


class NullDetector:
    name = "none"

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def detect(self, frame: Any) -> list[Detection]:
        return []


class YoloDetector:
    """Ultralytics YOLO adapter with the required football class order."""

    name = "yolo"
    class_names = YOLO_CLASS_NAMES

    def __init__(self, model_path: str | Path | None, confidence: float = 0.25) -> None:
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.confidence = confidence
        self.model: Any | None = None
        self.warnings: list[str] = []
        self.available = False
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path is None:
            self.warnings.append("YOLO model path was not provided; using fallback detections.")
            return
        if not self.model_path.exists():
            self.warnings.append(f"YOLO model not found at {self.model_path}; using fallback detections.")
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.warnings.append(f"Ultralytics is unavailable ({exc}); using fallback detections.")
            return
        try:
            self.model = YOLO(str(self.model_path))
            self.available = True
        except Exception as exc:
            self.warnings.append(f"Could not load YOLO model {self.model_path}: {exc}; using fallback detections.")

    def detect(self, frame: Any) -> list[Detection]:
        if not self.available or self.model is None:
            return []
        try:
            results = self.model.predict(frame, conf=self.confidence, verbose=False)
        except Exception as exc:
            self.warnings.append(f"YOLO prediction failed: {exc}")
            return []

        detections: list[Detection] = []
        height, width = frame.shape[:2]
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(_scalar(getattr(box, "cls", 0)))
                label = self.class_names.get(class_id)
                if label is None:
                    continue
                confidence = float(_scalar(getattr(box, "conf", 0.0)))
                if confidence < self.confidence:
                    continue
                xyxy = _to_list(getattr(box, "xyxy", [0, 0, 0, 0]))
                bbox = _clip_box(tuple(int(round(value)) for value in xyxy[:4]), width, height)
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=label,
                        confidence=confidence,
                        bbox=bbox,
                        source="yolo",
                    )
                )
        return detections


class ClassicalDetector:
    """Small OpenCV fallback so the pipeline remains runnable without weights."""

    name = "classical"

    def __init__(self, confidence: float = 0.25) -> None:
        self.confidence = confidence
        self.warnings: list[str] = []

    def detect(self, frame: Any) -> list[Detection]:
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            self.warnings.append(f"OpenCV fallback detector is unavailable: {exc}")
            return []

        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (35, 35, 35), (90, 255, 255))
        saturated = cv2.inRange(hsv, (0, 35, 35), (179, 255, 255))
        foreground = cv2.bitwise_and(saturated, cv2.bitwise_not(green))
        kernel = np.ones((3, 3), dtype=np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        frame_area = float(width * height)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
            area = cv2.contourArea(contour)
            if area < 35:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 1 or h <= 1:
                continue
            aspect = h / max(1, w)
            if area < 700 and 0.55 <= (w / max(1, h)) <= 1.8:
                label = "ball"
            elif h >= 18 and aspect >= 1.05:
                label = "player"
            else:
                continue
            class_id = CLASS_NAME_TO_ID[label]
            score = max(self.confidence, min(0.82, area / frame_area * 160.0))
            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    confidence=score,
                    bbox=_clip_box((x, y, x + w, y + h), width, height),
                    source="classical",
                )
            )
        return detections


class FallbackDetector:
    """Use YOLO when available and OpenCV fallback otherwise."""

    name = "hybrid"

    def __init__(self, primary: YoloDetector, fallback: ClassicalDetector) -> None:
        self.primary = primary
        self.fallback = fallback
        self.warnings = [*primary.warnings, *fallback.warnings]

    def detect(self, frame: Any) -> list[Detection]:
        detections = self.primary.detect(frame)
        if detections:
            return detections
        if not self.primary.available:
            return self.fallback.detect(frame)
        return detections


def create_detector(name: str, model_path: str | Path | None, confidence: float) -> Detector:
    normalized = (name or "auto").strip().lower()
    if normalized == "none":
        return NullDetector()
    if normalized == "classical":
        return ClassicalDetector(confidence)
    if normalized in {"auto", "yolo"}:
        yolo = YoloDetector(model_path, confidence)
        if normalized == "yolo":
            return FallbackDetector(yolo, ClassicalDetector(confidence))
        if yolo.available:
            return yolo
        return ClassicalDetector(confidence)
    raise ValueError(f"Unknown detector {name!r}. Expected auto, yolo, classical, or none.")


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, (list, tuple)):
        return float(value[0])
    try:
        return float(value[0])
    except Exception:
        return float(value)


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        if len(value) == 1 and not isinstance(value[0], (int, float)):
            value = value[0]
    except Exception:
        pass
    return [float(item) for item in list(value)]


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2
