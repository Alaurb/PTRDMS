from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .detectors import Detection


@dataclass
class Pose:
    frame: str
    x: float
    y: float
    z: float
    yaw: float
    map_px: float
    map_py: float
    source: str
    route: int | str


@dataclass
class SpatialDetection:
    frame: str
    side: str
    class_name: str
    confidence: float
    detector: str
    x: float
    y: float
    z: float
    map_px: float
    map_py: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    view_width: int
    view_height: int
    annotated_image: str
    position_source: str = "assumed_row_plane"
    track_id: str = ""
    detector_confidence: float | None = None
    classifier_confidence: float | None = None
    raw_class: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _frame_number(path: str | Path) -> int | None:
    stem = Path(path).stem
    panoramic_match = re.search(r"panoramic(\d+)", stem, flags=re.IGNORECASE)
    if panoramic_match:
        return int(panoramic_match.group(1))
    matches = re.findall(r"\d+", stem)
    return int(matches[-1]) if matches else None


def _contiguous_groups(paths: list[Path]) -> list[list[Path]]:
    groups: list[list[Path]] = []
    for path in paths:
        number = _frame_number(path)
        if not groups:
            groups.append([path])
            continue
        previous = _frame_number(groups[-1][-1])
        if number is not None and previous is not None and number == previous + 1:
            groups[-1].append(path)
        else:
            groups.append([path])
    return groups


def _find_aisles(map_image: Image.Image, count: int) -> list[float]:
    gray = np.asarray(map_image.convert("L"), dtype=np.float32) / 255.0
    height, width = gray.shape
    left, right = max(1, width // 20), min(width - 1, width - width // 20)
    profile = gray[:, left:right].mean(axis=1)
    window = max(5, height // 90)
    if window % 2 == 0:
        window += 1
    smooth = np.convolve(profile, np.ones(window) / window, mode="same")
    threshold = float(np.percentile(smooth[5:-5], 58))
    candidates = [
        y
        for y in range(8, height - 8)
        if smooth[y] >= threshold and smooth[y] >= smooth[y - 1] and smooth[y] > smooth[y + 1]
    ]
    separated: list[int] = []
    minimum_gap = max(9, height // 35)
    for candidate in sorted(candidates, key=lambda y: smooth[y], reverse=True):
        if all(abs(candidate - existing) >= minimum_gap for existing in separated):
            separated.append(candidate)
    separated.sort()
    if len(separated) < count:
        return np.linspace(height * 0.10, height * 0.90, count).tolist()
    indices = np.linspace(0, len(separated) - 1, count).round().astype(int)
    return [float(separated[index]) for index in indices]


def generate_demo_poses(
    image_paths: list[Path], map_image: Image.Image, map_width_m: float = 35.0
) -> dict[str, Pose]:
    """Generate deterministic demonstration poses from contiguous filename sequences.

    These poses place each contiguous acquisition sequence on a free aisle detected
    in the supplied occupancy map. They are explicitly labelled synthetic_map_path.
    """
    if not image_paths:
        return {}
    width_px, height_px = map_image.size
    scale = map_width_m / width_px
    groups = _contiguous_groups(image_paths)
    aisle_rows = _find_aisles(map_image, len(groups))
    poses: dict[str, Pose] = {}

    for route_index, (group, aisle_y_px) in enumerate(zip(groups, aisle_rows), start=1):
        start_px, end_px = width_px * 0.07, width_px * 0.93
        if route_index % 2 == 0:
            start_px, end_px = end_px, start_px
        denominator = max(len(group) - 1, 1)
        for index, path in enumerate(group):
            fraction = index / denominator
            map_px = start_px + (end_px - start_px) * fraction
            yaw = 0.0 if end_px >= start_px else math.pi
            pose = Pose(
                frame=path.name,
                x=map_px * scale,
                y=(height_px - aisle_y_px) * scale,
                z=0.0,
                yaw=yaw,
                map_px=map_px,
                map_py=aisle_y_px,
                source="synthetic_map_path",
                route=route_index,
            )
            poses[path.name] = pose
            poses[path.stem] = pose
    return poses


def load_pose_csv(
    csv_path: str | Path, map_image: Image.Image, map_width_m: float = 35.0
) -> dict[str, Pose]:
    """Load frame,x,y,z,yaw pose rows. yaw is in radians; x/y/z are metres."""
    width_px, height_px = map_image.size
    scale = map_width_m / width_px
    output: dict[str, Pose] = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=2):
            frame = (row.get("frame") or row.get("image") or row.get("filename") or "").strip()
            if not frame:
                raise ValueError(f"Pose CSV row {row_index} has no frame/image/filename value")
            try:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row.get("z") or 0.0)
                yaw = float(row.get("yaw") or 0.0)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid pose values in CSV row {row_index}: {row}") from exc
            if not all(math.isfinite(value) for value in (x, y, z, yaw)):
                raise ValueError(f"Non-finite pose in row {row_index}")
            if frame in output or Path(frame).stem in output:
                raise ValueError(f"Duplicate pose frame: {frame}")
            route_value = row.get("route") or "0"
            pose = Pose(
                frame=frame,
                x=x,
                y=y,
                z=z,
                yaw=yaw,
                map_px=x / scale,
                map_py=height_px - y / scale,
                source="pose_csv",
                route=int(route_value) if route_value.lstrip("-").isdigit() else route_value,
            )
            output[frame] = pose
            output[Path(frame).stem] = pose
    return output


def project_detection(
    detection: Detection,
    side: str,
    pose: Pose,
    view_size: tuple[int, int],
    map_size: tuple[int, int],
    map_width_m: float = 35.0,
    row_offset_m: float = 1.20,
    camera_height_m: float = 1.15,
    fov_degrees: float = 90.0,
    annotated_image: str = "",
    range_m: float | None = None,
) -> SpatialDetection:
    """Project a side-face detection using pose, cubemap bearing and row distance.

    With no depth stream in the supplied sample, row_offset_m supplies the assumed
    lateral crop-row distance. Replace it with measured depth for quantitative use.
    """
    view_width, view_height = view_size
    center_x, center_y = detection.center
    if side not in ("left", "right"):
        raise ValueError("Spatial projection requires a left or right cube face")
    if min(view_size) < 2 or not 0 < fov_degrees < 180:
        raise ValueError("Invalid perspective geometry")
    if not all(math.isfinite(v) for v in (map_width_m, row_offset_m, camera_height_m)) or min(map_width_m, row_offset_m) <= 0:
        raise ValueError("Map width and row offset must be finite and positive")
    # Match the endpoint sampling convention in panorama._face_vectors.
    u = (2 * center_x / (view_width - 1) - 1) * math.tan(math.radians(fov_degrees) / 2)
    v = (1 - 2 * center_y / (view_height - 1)) * math.tan(math.radians(fov_degrees) / 2)
    lateral = 1.0 if side == "left" else -1.0
    forward = u if side == "left" else -u
    if range_m is not None:
        if not math.isfinite(range_m) or range_m <= 0:
            raise ValueError("Measured radial range must be finite and positive")
        factor = range_m / math.sqrt(forward * forward + 1 + v * v)
    else:
        factor = row_offset_m
    x = pose.x + factor * (forward * math.cos(pose.yaw) - lateral * math.sin(pose.yaw))
    y = pose.y + factor * (forward * math.sin(pose.yaw) + lateral * math.cos(pose.yaw))
    # Pose CSV z is the camera optical-centre height; synthetic paths use the configured height.
    z = (pose.z if pose.source == "pose_csv" else camera_height_m) + factor * v

    map_width_px, map_height_px = map_size
    scale = map_width_m / map_width_px
    # Keep out-of-map coordinates: clipping would fabricate positions and bias metrics.
    map_px = x / scale
    map_py = map_height_px - y / scale
    x1, y1, x2, y2 = detection.bbox
    return SpatialDetection(
        frame=pose.frame,
        side=side,
        class_name=detection.class_name,
        confidence=detection.confidence,
        detector=detection.detector,
        x=x,
        y=y,
        z=z,
        map_px=map_px,
        map_py=map_py,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        view_width=view_width,
        view_height=view_height,
        annotated_image=annotated_image,
        position_source="measured_radial_range" if range_m is not None else "assumed_row_plane",
        detector_confidence=detection.detector_confidence,
        classifier_confidence=detection.classifier_confidence,
        raw_class=detection.raw_class,
    )
