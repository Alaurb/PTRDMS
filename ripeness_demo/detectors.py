from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


CLASS_COLORS = {
    # Paper four-stage taxonomy. These are the counted maturity states used by
    # the deployed classifier trained on the reviewed 815-image corpus.
    "immature": "#2563eb",
    "green_mature": "#22c55e",
    "discoloration": "#f59e0b",
    "mature": "#ef4444",
    # Compatibility colours for the archived Green Gem demonstration only.
    "mature_green": "#22c55e",
    "harvest_ready": "#eab308",
    "overripe_or_defective": "#a855f7",
}

GREEN_GEM_CLASS_ORDER = ("overripe_or_defective", "harvest_ready", "mature_green", "immature")
PAPER_FOUR_STAGE_ORDER = ("mature", "discoloration", "green_mature", "immature")
# ``legacy`` remains readable for immutable archived demonstration output.
LEGACY_CLASS_ORDER = PAPER_FOUR_STAGE_ORDER


def class_order_for(class_names: set[str] | list[str] | tuple[str, ...], taxonomy: str | None = None) -> tuple[str, ...]:
    """Return a display order without silently relabelling archived predictions."""
    if taxonomy is not None:
        if taxonomy not in {"legacy", "paper_four_stage", "green_gem"}:
            raise ValueError(f"Unknown taxonomy: {taxonomy}")
        order = GREEN_GEM_CLASS_ORDER if taxonomy == "green_gem" else PAPER_FOUR_STAGE_ORDER
        if set(class_names) - set(order):
            raise ValueError("Prediction classes conflict with configured taxonomy")
        return order
    # Compatibility for old callers; production uses model metadata.
    if set(class_names) & {"mature_green", "harvest_ready", "overripe_or_defective"}:
        return GREEN_GEM_CLASS_ORDER
    return PAPER_FOUR_STAGE_ORDER

CLASS_ALIASES = {
    "immature": "immature",
    "immature-period": "immature",
    "unripe": "immature",
    "mature_green": "mature_green",
    "mature green": "mature_green",
    "harvest_ready": "harvest_ready",
    "harvest ready": "harvest_ready",
    "overripe_or_defective": "overripe_or_defective",
    "overripe or defective": "overripe_or_defective",
    "green": "green_mature",
    "green mature": "green_mature",
    "green_mature": "green_mature",
    "green-maturity-period": "green_mature",
    "breaker": "discoloration",
    "turning": "discoloration",
    "discoloration": "discoloration",
    "discoloration-period": "discoloration",
    "half-ripe": "discoloration",
    "half_ripe": "discoloration",
    "mature": "mature",
    "maturity": "mature",
    "maturity-period": "mature",
    "ripe": "mature",
    "red": "mature",
}


@dataclass(slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    detector: str
    detector_confidence: float | None = None
    classifier_confidence: float | None = None
    raw_class: str = ""

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class Detector(Protocol):
    name: str

    def detect(self, image: Image.Image) -> list[Detection]: ...


def model_taxonomy(names) -> str:
    """Validate the complete model vocabulary, including classes absent in a run."""
    labels = {str(n).lower().strip() for n in names.values()}
    unknown = labels - set(CLASS_ALIASES) - {"other"}
    if unknown:
        raise ValueError(f"Unmapped model classes: {sorted(unknown)}")
    canonical = {CLASS_ALIASES[n] for n in labels if n != "other"}
    green = canonical & (set(GREEN_GEM_CLASS_ORDER) - {"immature"})
    paper = canonical & (set(PAPER_FOUR_STAGE_ORDER) - {"immature"})
    if green and paper or not (green or paper):
        raise ValueError("Mixed or ambiguous model taxonomy")
    return "green_gem" if green else "paper_four_stage"


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(first_area + second_area - intersection, 1e-9)


def _merge_box_candidates(
    candidates: list[tuple[tuple[float, float, float, float], float]],
    iou_threshold: float = 0.55,
) -> list[tuple[tuple[float, float, float, float], float]]:
    kept: list[tuple[tuple[float, float, float, float], float]] = []
    for bbox, confidence in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(_box_iou(bbox, existing_bbox) < iou_threshold for existing_bbox, _ in kept):
            kept.append((bbox, confidence))
    return kept


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = rgb.astype(np.float32) / 255.0
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-7)
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-7
    r, g, b = values[..., 0], values[..., 1], values[..., 2]
    red = nonzero & (maximum == r)
    green = nonzero & (maximum == g)
    blue = nonzero & (maximum == b)
    hue[red] = ((g[red] - b[red]) / delta[red]) % 6.0
    hue[green] = (b[green] - r[green]) / delta[green] + 2.0
    hue[blue] = (r[blue] - g[blue]) / delta[blue] + 4.0
    return hue / 6.0, saturation, maximum


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask
    for _ in range(iterations):
        padded = np.pad(result, 1, constant_values=False)
        result = np.logical_and.reduce(
            [padded[dy : dy + result.shape[0], dx : dx + result.shape[1]] for dy in range(3) for dx in range(3)]
        )
    return result


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask
    for _ in range(iterations):
        padded = np.pad(result, 1, constant_values=False)
        result = np.logical_or.reduce(
            [padded[dy : dy + result.shape[0], dx : dx + result.shape[1]] for dy in range(3) for dx in range(3)]
        )
    return result


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int, list[tuple[int, int]]]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    output = []
    for y, x in zip(*np.nonzero(mask & ~visited)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []
        min_x = max_x = int(x)
        min_y = max_y = int(y)
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            min_x, max_x = min(min_x, cx), max(max_x, cx)
            min_y, max_y = min(min_y, cy), max(max_y, cy)
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        output.append((min_x, min_y, max_x + 1, max_y + 1, len(pixels), pixels))
    return output


class ColorShapeDetector:
    """Dependency-light Green Gem visual proxy when trained weights are absent.

    It uses HSV color, morphological filtering and component shape. This is an
    engineering fallback, not a replacement for the paper's trained YOLO model.
    """

    name = "color_shape_fallback_green_gem_proxy"
    taxonomy = "green_gem"

    def __init__(self, analysis_size: int = 360, max_detections: int = 12):
        self.analysis_size = analysis_size
        self.max_detections = max_detections

    def detect(self, image: Image.Image) -> list[Detection]:
        original_width, original_height = image.size
        scale = min(1.0, self.analysis_size / max(original_width, original_height))
        work_size = (max(64, round(original_width * scale)), max(64, round(original_height * scale)))
        work = image.convert("RGB").resize(work_size, Image.Resampling.BILINEAR)
        rgb = np.asarray(work)
        hue, saturation, value = _rgb_to_hsv(rgb)

        red = ((hue < 0.045) | (hue > 0.97)) & (saturation > 0.40) & (value > 0.20)
        warm = (hue >= 0.035) & (hue < 0.17) & (saturation > 0.42) & (value > 0.25)
        # Green fruit in the supplied images is generally brighter and smoother
        # than foliage. The higher value floor deliberately favors precision.
        green = (hue >= 0.16) & (hue < 0.45) & (saturation > 0.27) & (value > 0.30) & (value < 0.90)
        # Cube-face bottom mainly contains soil and plastic mulch, which otherwise
        # creates many brown/orange false positives.
        warm[int(work_size[1] * 0.76) :, :] = False
        detections: list[Detection] = []
        # Warm fruit must be labelled independently from adjacent green foliage;
        # otherwise both colors merge into one very large connected component.
        component_masks = (("warm", red | warm), ("green", green))
        for mask_kind, raw_mask in component_masks:
            candidate = _binary_dilate(_binary_erode(raw_mask, iterations=2), iterations=2)
            for x1, y1, x2, y2, area, pixels in _components(candidate):
                box_w, box_h = x2 - x1, y2 - y1
                max_area = 6500 if mask_kind == "warm" else 1900
                max_dimension = 110 if mask_kind == "warm" else 72
                if area < 34 or area > max_area or min(box_w, box_h) < 8 or max(box_w, box_h) > max_dimension:
                    continue
                aspect = box_w / max(box_h, 1)
                fill = area / max(box_w * box_h, 1)
                roundness = area / max(np.pi * (max(box_w, box_h) / 2.0) ** 2, 1.0)
                if not 0.64 <= aspect <= 1.56 or fill < 0.40 or roundness < 0.28:
                    continue

                yy = np.fromiter((p[0] for p in pixels), dtype=np.int32)
                xx = np.fromiter((p[1] for p in pixels), dtype=np.int32)
                local = np.zeros((box_h + 2, box_w + 2), dtype=bool)
                local[yy - y1 + 1, xx - x1 + 1] = True
                perimeter = int(
                    (local & ~np.roll(local, 1, axis=0)).sum()
                    + (local & ~np.roll(local, -1, axis=0)).sum()
                    + (local & ~np.roll(local, 1, axis=1)).sum()
                    + (local & ~np.roll(local, -1, axis=1)).sum()
                )
                circularity = min(1.0, 4.0 * np.pi * area / max(perimeter * perimeter, 1))
                if circularity < 0.36:
                    continue
                red_fraction = float(red[yy, xx].mean())
                warm_fraction = float(warm[yy, xx].mean())
                mean_value = float(value[yy, xx].mean())
                # Green Gem fruit is harvest-ready because of a yellow halo,
                # not a red surface. This is only a runnable UI proxy.
                if mask_kind == "warm" and red_fraction >= 0.42:
                    class_name = "overripe_or_defective"
                elif mask_kind == "warm":
                    class_name = "harvest_ready"
                elif mean_value >= 0.42 and area >= 34:
                    class_name = "mature_green"
                else:
                    class_name = "immature"

                confidence = min(
                    0.92,
                    0.20
                    + 0.22 * min(fill, 1.0)
                    + 0.22 * min(roundness, 1.0)
                    + 0.26 * circularity,
                )
                if confidence < 0.58:
                    continue
                sx, sy = original_width / work_size[0], original_height / work_size[1]
                detections.append(
                    Detection(class_name, confidence, (x1 * sx, y1 * sy, x2 * sx, y2 * sy), self.name)
                )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[: self.max_detections]


class YoloRipenessDetector:
    name = "yolo"

    def __init__(self, weights: str | Path, confidence: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Install optional-requirements.txt first.") from exc
        self.model = YOLO(str(weights))
        self.taxonomy = model_taxonomy(self.model.names)
        self.confidence = confidence

    def detect(self, image: Image.Image) -> list[Detection]:
        bgr = np.asarray(image.convert("RGB"))[..., ::-1].copy()
        result = self.model.predict(bgr, conf=self.confidence, verbose=False)[0]
        detections: list[Detection] = []
        for box in result.boxes:
            raw_name = str(self.model.names[int(box.cls[0])]).lower().strip()
            canonical = CLASS_ALIASES.get(raw_name)
            if canonical is None:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(Detection(canonical, float(box.conf[0]), (x1, y1, x2, y2), self.name))
        return detections


class TwoStageYoloDetector:
    """Tomato detector followed by the paper's four-stage ripeness classifier."""

    name = "yolo_two_stage"

    def __init__(
        self,
        detector_weights: str | Path,
        classifier_weights: str | Path,
        confidence: float = 0.25,
        brightness_gain: float = 1.25,
        detector_imgsz: int = 1280,
        classifier_imgsz: int | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Install optional-requirements.txt first.") from exc
        self.detector = YOLO(str(detector_weights))
        self.classifier = YOLO(str(classifier_weights))
        self.classifier_imgsz = classifier_imgsz or self.classifier.overrides.get("imgsz", 224)
        self.taxonomy = model_taxonomy(self.classifier.names)
        if {str(n).lower().strip() for n in self.detector.names.values()} != {"tomato"}:
            raise ValueError("Two-stage detector must have exactly one class: tomato")
        self.audit = []
        self.confidence = confidence
        self.brightness_gain = max(1.0, float(brightness_gain))
        self.detector_imgsz = max(320, int(detector_imgsz))
        self.name = "yolo_two_stage_brightness_ensemble" if self.brightness_gain > 1.001 else "yolo_two_stage"

    def detect(self, image: Image.Image) -> list[Detection]:
        rgb = np.asarray(image.convert("RGB"))
        self.audit = []
        bgr = rgb[..., ::-1].copy()
        detector_inputs = [bgr]
        if self.brightness_gain > 1.001:
            enhanced = ImageEnhance.Brightness(image.convert("RGB")).enhance(self.brightness_gain)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)
            detector_inputs.append(np.asarray(enhanced)[..., ::-1].copy())
        results = self.detector.predict(
            detector_inputs,
            conf=self.confidence,
            imgsz=self.detector_imgsz,
            verbose=False,
        )
        candidates: list[tuple[tuple[float, float, float, float], float]] = []
        for result in results:
            for box in result.boxes:
                bbox = tuple(float(value) for value in box.xyxy[0].tolist())
                candidates.append((bbox, float(box.conf[0])))
        merged_candidates = _merge_box_candidates(candidates)
        height, width = rgb.shape[:2]
        detections: list[Detection] = []
        for bbox, detector_confidence in merged_candidates:
            x1, y1, x2, y2 = bbox
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            crop = bgr[iy1:iy2, ix1:ix2]
            classification = self.classifier.predict(crop, imgsz=self.classifier_imgsz, verbose=False)[0]
            class_id = int(classification.probs.top1)
            raw_name = str(self.classifier.names[class_id]).lower().strip()
            canonical = CLASS_ALIASES.get(raw_name)
            classifier_confidence = float(classification.probs.top1conf)
            self.audit.append(dict(bbox=bbox, raw_class=raw_name,
                                   detector_confidence=detector_confidence,
                                   classifier_confidence=classifier_confidence,
                                   decision="rejected_other" if raw_name == "other" else "accepted"))
            if canonical is None or raw_name == "other":
                continue
            classifier_confidence = float(classification.probs.top1conf)
            combined_confidence = float(np.sqrt(detector_confidence * classifier_confidence))
            detections.append(
                Detection(canonical, combined_confidence, (x1, y1, x2, y2), self.name,
                          detector_confidence, classifier_confidence, raw_name)
            )
        return detections


def build_detector(
    mode: str = "auto",
    weights: str | Path | None = None,
    confidence: float = 0.25,
    detector_weights: str | Path | None = None,
    classifier_weights: str | Path | None = None,
    brightness_gain: float = 1.25,
    detector_imgsz: int = 1280,
    classifier_imgsz: int | None = None,
) -> Detector:
    mode = mode.lower()
    if mode not in {"auto", "yolo", "two-stage", "color"}:
        raise ValueError(f"Unknown detector mode: {mode}")
    if detector_weights or classifier_weights:
        if not detector_weights or not classifier_weights:
            raise ValueError("Two-stage inference requires both detector and classifier weights")
        detector_path, classifier_path = Path(detector_weights), Path(classifier_weights)
        if not detector_path.exists():
            raise FileNotFoundError(f"Tomato detector weights do not exist: {detector_path}")
        if not classifier_path.exists():
            raise FileNotFoundError(f"Ripeness classifier weights do not exist: {classifier_path}")
        if mode in {"auto", "two-stage"}:
            return TwoStageYoloDetector(
                detector_path,
                classifier_path,
                confidence,
                brightness_gain,
                detector_imgsz,
                classifier_imgsz,
            )
    if weights:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights do not exist: {weights}")
        if mode in {"auto", "yolo"}:
            return YoloRipenessDetector(weights, confidence)
    if mode == "two-stage":
        raise ValueError("--detector two-stage requires --detector-weights and --classifier-weights")
    if mode == "yolo":
        raise ValueError("--detector yolo requires --weights pointing to a four-class model")
    return ColorShapeDetector()


def annotate(image: Image.Image, detections: list[Detection]) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=14)
    image_width, _ = result.size
    for detection in detections:
        color = CLASS_COLORS[detection.class_name]
        x1, y1, x2, y2 = detection.bbox
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f"{detection.class_name} {detection.confidence:.2f}"
        label_box = draw.textbbox((x1, y1), text, font=font, stroke_width=1)
        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 6
        label_top = max(0, y1 - label_height)
        label_left = min(max(0, x1), max(0, image_width - label_width))
        draw.rectangle((label_left, label_top, label_left + label_width, y1), fill=color)
        draw.text((label_left + 4, label_top + 2), text, fill="white", font=font, stroke_width=1, stroke_fill=color)
    return result
