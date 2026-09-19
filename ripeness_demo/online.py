"""ROS-independent building blocks for timestamped online PTRDMS processing."""

from __future__ import annotations

import math
import threading
from dataclasses import replace

from .mapping import Pose


def timestamp_key(seconds: int, nanoseconds: int) -> str:
    """Format a ROS timestamp without losing nanosecond identity to a float."""

    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("Invalid ROS timestamp")
    return f"{seconds}.{nanoseconds:09d}"


class TimedPoseBuffer:
    """Bounded, thread-safe pose buffer for online panorama association.

    Poses are stored by the exact ROS header timestamp. ``frame_id`` remains
    coordinate-frame metadata and is never used as a pose-identity key.
    """

    def __init__(self, map_size: tuple[int, int], map_width_m: float, max_poses: int = 2_000):
        if max_poses < 2 or map_width_m <= 0:
            raise ValueError("max_poses must be at least 2 and map_width_m must be positive")
        self._map_size = map_size
        self._map_width_m = map_width_m
        self._max_poses = max_poses
        self._poses: dict[str, Pose] = {}
        self._lock = threading.Lock()
        self.duplicate_timestamp_count = 0

    def add(
        self,
        *,
        seconds: int,
        nanoseconds: int,
        frame_id: str,
        x: float,
        y: float,
        z: float,
        yaw: float,
        route: str | int = "online",
    ) -> bool:
        """Store one pose; retain the first message when a timestamp is repeated."""

        key = timestamp_key(seconds, nanoseconds)
        values = (x, y, z, yaw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Online pose contains a non-finite value")
        width_px, height_px = self._map_size
        scale = self._map_width_m / width_px
        pose = Pose(
            frame=frame_id,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            map_px=x / scale,
            map_py=height_px - y / scale,
            source="ros_pose",
            route=route,
            timestamp_s=seconds + nanoseconds / 1_000_000_000,
        )
        with self._lock:
            if key in self._poses:
                self.duplicate_timestamp_count += 1
                return False
            self._poses[key] = pose
            if len(self._poses) > self._max_poses:
                oldest = min(self._poses, key=lambda item: self._poses[item].timestamp_s or 0.0)
                del self._poses[oldest]
        return True

    def match(
        self,
        *,
        seconds: int,
        nanoseconds: int,
        frame_name: str,
        tolerance_s: float,
    ) -> Pose | None:
        """Return one unique nearest pose, or ``None`` when no safe match exists."""

        if not math.isfinite(tolerance_s) or tolerance_s <= 0:
            raise ValueError("tolerance_s must be finite and positive")
        frame_time = seconds + nanoseconds / 1_000_000_000
        with self._lock:
            candidates = list(self._poses.values())
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda pose: abs(float(pose.timestamp_s) - frame_time))
        nearest = ranked[0]
        delta_s = abs(float(nearest.timestamp_s) - frame_time)
        if delta_s > tolerance_s:
            return None
        if len(ranked) > 1:
            next_delta_s = abs(float(ranked[1].timestamp_s) - frame_time)
            if math.isclose(next_delta_s, delta_s, abs_tol=1e-9):
                return None
        return replace(
            nearest,
            frame=frame_name,
            frame_timestamp_s=frame_time,
            match_delta_s=delta_s,
        )

    def size(self) -> int:
        with self._lock:
            return len(self._poses)
