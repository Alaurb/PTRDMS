"""Measured-range quality gates and conservative within-pass association."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


class RangeEvidence:
    """Registered radial-range maps in metres; no unregistered RGB-D input."""

    def __init__(self, manifest: str, tolerance_s: float = 0.05):
        path = Path(manifest).resolve()
        self.root = path.parent
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("range_convention") != "radial_metres" or not data.get("calibration_id"):
            raise ValueError("Range manifest requires radial_metres and a calibration_id")
        if not math.isfinite(tolerance_s) or tolerance_s < 0:
            raise ValueError("Synchronization tolerance must be finite and nonnegative")
        self.calibration_id = data["calibration_id"]
        self.records = {}
        self.tolerance = tolerance_s
        for record in data["views"]:
            key = (record["frame"], record["side"])
            if key in self.records or key[1] not in ("left", "right"):
                raise ValueError(f"Duplicate or invalid range view: {key}")
            self.records[key] = record

    def load(self, frame, side, size):
        record = self.records.get((frame, side))
        if record is None:
            raise ValueError(f"Missing synchronized range view: {frame}/{side}")
        times = [float(record[k]) for k in ("image_time_s", "range_time_s", "pose_time_s")]
        if not all(math.isfinite(t) for t in times) or max(times) - min(times) > self.tolerance:
            raise ValueError(f"Timestamp alignment exceeds tolerance: {frame}/{side}")
        ranges = np.load(self.root / record["file"], allow_pickle=False)
        if ranges.shape != (size[1], size[0]) or not np.issubdtype(ranges.dtype, np.number):
            raise ValueError(f"Range map must match projected view resolution: {frame}/{side}")
        return ranges


def foreground_range(ranges, bbox, min_samples=9):
    """Median of the central half-box; missing evidence is explicitly rejected."""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None
    h, w = ranges.shape
    xa, xb = max(0, int(x1 + width * .25)), min(w, math.ceil(x2 - width * .25))
    ya, yb = max(0, int(y1 + height * .25)), min(h, math.ceil(y2 - height * .25))
    region = ranges[ya:yb, xa:xb]
    valid = region[np.isfinite(region) & (region > 0)]
    if valid.size < min_samples or valid.size < region.size * .5:
        return None
    return float(np.median(valid))


def associate(detections, poses, distance_m=.12, max_frame_gap=3):
    """One-to-one greedy spatial candidates per frame, never on synthetic geometry.

    IDs are local to one pass. Class is not a hard gate: maturity may change.
    Association is heuristic and must be evaluated against manually assigned IDs.
    """
    if not math.isfinite(distance_m) or distance_m <= 0 or max_frame_gap < 1:
        raise ValueError("Invalid association thresholds")
    tracks, links = [], []
    eligible = all(p.source == "pose_csv" for p in poses)
    by_frame = {}
    for detection in detections:
        by_frame.setdefault(detection.frame, []).append(detection)
    for index, pose in enumerate(poses):
        current = by_frame.get(pose.frame, [])
        candidates = []
        for di, detection in enumerate(current):
            if not eligible or detection.position_source != "measured_radial_range":
                continue
            xyz = np.array([detection.x, detection.y, detection.z])
            for ti, track in enumerate(tracks):
                if (track["route"], track["side"]) != (pose.route, detection.side):
                    continue
                if index - track["last_index"] > max_frame_gap:
                    continue
                if not track["measured"]:
                    continue
                distance = float(np.linalg.norm(xyz - track["xyz"]))
                if distance <= distance_m:
                    candidates.append((distance, di, ti))
        matched_d, matched_t = set(), set()
        assignments = {}
        for distance, di, ti in sorted(candidates):
            if di not in matched_d and ti not in matched_t:
                assignments[di] = ti
                matched_d.add(di)
                matched_t.add(ti)
        for di, detection in enumerate(current):
            ti = assignments.get(di)
            xyz = np.array([detection.x, detection.y, detection.z])
            if ti is None:
                ti = len(tracks)
                tracks.append(dict(track_id=f"T{ti+1:06d}", route=pose.route,
                                   side=detection.side, xyz=xyz, first_frame=pose.frame,
                                   last_frame=pose.frame, last_index=index, observations=0,
                                   votes=Counter(), measured=eligible and detection.position_source == "measured_radial_range"))
            track = tracks[ti]
            n = track["observations"]
            track["xyz"] = (track["xyz"] * n + xyz) / (n + 1)
            track["observations"] += 1
            track["last_index"], track["last_frame"] = index, pose.frame
            track["votes"][detection.class_name] += detection.confidence
            detection.track_id = track["track_id"]
            links.append(dict(frame=pose.frame, side=detection.side, track_id=detection.track_id,
                              observation_index=di, matched_previous=di in assignments,
                              class_name=detection.class_name, x=detection.x, y=detection.y, z=detection.z))
    rows = []
    for track in tracks:
        rows.append(dict(track_id=track["track_id"], route=track["route"], side=track["side"],
                         x=float(track["xyz"][0]), y=float(track["xyz"][1]), z=float(track["xyz"][2]),
                         first_frame=track["first_frame"], last_frame=track["last_frame"],
                         observations=track["observations"], class_name=track["votes"].most_common(1)[0][0],
                         status="association_candidate" if track["measured"] else "unassociated_observation"))
    return rows, links
