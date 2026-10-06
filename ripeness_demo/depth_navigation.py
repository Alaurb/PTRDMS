"""Metric RGB-D traversability and supervisory gait decisions (ROS independent)."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class DepthConfig:
    # Geometry and locomotion limits are supplied by the platform calibration.
    robot_width_m: float
    safety_margin_m: float
    ground_z_m: float
    max_step_up_m: float
    max_step_down_m: float
    max_slope_deg: float
    max_roughness_m: float
    collision_height_m: float
    terrain_trigger_m: float
    normal_speed_mps: float
    terrain_speed_mps: float
    reaction_time_s: float
    braking_accel_mps2: float
    min_depth_m: float = 0.2
    max_depth_m: float = 4.0
    stride: int = 4
    cell_m: float = 0.1
    forward_min_m: float = 0.25
    forward_max_m: float = 2.0
    ground_fit_max_m: float = 0.85
    ground_anchor_band_m: float = 0.08
    plane_tolerance_m: float = 0.025
    min_points_cell: int = 3
    min_coverage: float = 0.65
    enter_frames: int = 3
    exit_frames: int = 5
    stale_s: float = 0.3
    perception_half_width_m: float = 1.2

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name != "ground_z_m" and value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("stride", "min_points_cell", "enter_frames", "exit_frames"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if not 0 < self.min_coverage <= 1 or not 0 < self.max_slope_deg < 90:
            raise ValueError("Invalid coverage or slope limit")
        if not self.min_depth_m < self.max_depth_m or not self.forward_min_m < self.ground_fit_max_m < self.forward_max_m:
            raise ValueError("Invalid sensing or ground-fit range")
        if self.terrain_trigger_m >= min(self.max_step_up_m, self.max_step_down_m):
            raise ValueError("Terrain trigger must lie below the step limits")
        if self.terrain_speed_mps > self.normal_speed_mps:
            raise ValueError("Terrain speed must not exceed normal speed")
        if self.perception_half_width_m < self.robot_width_m / 2 + self.safety_margin_m:
            raise ValueError("Perception width must include the travel corridor")
        if self.collision_height_m <= self.max_step_up_m:
            raise ValueError("Collision height must exceed the step limit")
        if self.forward_max_m - self.forward_min_m < self.cell_m:
            raise ValueError("Look-ahead range must include at least one complete cell")


def depth_to_points(depth, intrinsics, transform, encoding="16UC1", stride=4,
                    min_depth_m=0.2, max_depth_m=4.0):
    """Rectified optical depth -> calibrated, gravity-aligned XYZ.

    intrinsics=(fx, fy, cx, cy); transform maps optical coordinates into the
    gravity-aligned robot frame, with x forward, y left and z up.
    """
    depth = np.asarray(depth)
    k = np.asarray(intrinsics, dtype=float)
    t = np.asarray(transform, dtype=float)
    if depth.ndim != 2 or k.shape != (4,) or not np.isfinite(k).all() or np.any(k[:2] <= 0):
        raise ValueError("Expected a 2D depth image and valid intrinsics")
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError("Expected a finite homogeneous transform")
    if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5):
        raise ValueError("Transform rotation must be orthonormal and right-handed")
    if not isinstance(stride, int) or stride < 1 or not 0 < min_depth_m < max_depth_m:
        raise ValueError("Invalid sampling or depth range")
    if encoding not in ("16UC1", "32FC1"):
        raise ValueError("Depth encoding must be 16UC1 (mm) or 32FC1 (m)")
    if encoding == "16UC1" and depth.dtype != np.uint16:
        raise ValueError("16UC1 requires uint16 depth")
    if encoding == "32FC1" and not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("32FC1 requires floating-point depth")
    z = depth[::stride, ::stride].astype(float) * (0.001 if encoding == "16UC1" else 1.0)
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    fx, fy, cx, cy = k
    optical = np.column_stack(((u[valid] - cx) * z[valid] / fx,
                               (v[valid] - cy) * z[valid] / fy, z[valid]))
    return optical @ t[:3, :3].T + t[:3, 3]


def fit_ground(points, config):
    """Deterministic RANSAC on a calibrated near-ground anchor, not all surfaces."""
    c = config
    half = c.robot_width_m / 2 + c.safety_margin_m
    mask = ((points[:, 0] >= c.forward_min_m) & (points[:, 0] <= c.ground_fit_max_m)
            & (np.abs(points[:, 1]) <= half)
            & (np.abs(points[:, 2] - c.ground_z_m) <= c.ground_anchor_band_m))
    anchor = points[mask]
    if len(anchor) < 12:
        return None
    rng = np.random.default_rng(0)
    best = np.zeros(len(anchor), dtype=bool)
    for _ in range(80):
        sample = anchor[rng.choice(len(anchor), 3, replace=False)]
        n = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n /= norm
        if n[2] < 0:
            n = -n
        if n[2] < math.cos(math.radians(c.max_slope_deg)):
            continue
        inliers = np.abs((anchor - sample[0]) @ n) <= c.plane_tolerance_m
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < max(12, math.ceil(len(anchor) * 0.5)):
        return None
    p = anchor[best]
    if np.linalg.matrix_rank(p[:, :2] - p[:, :2].mean(axis=0)) < 2:
        return None
    a, b, offset = np.linalg.lstsq(np.column_stack((p[:, :2], np.ones(len(p)))), p[:, 2], rcond=None)[0]
    if math.degrees(math.atan(math.hypot(a, b))) > c.max_slope_deg:
        return None
    return np.array([a, b, offset])


def assess_points(points, config):
    """Classify grid cells, retaining unknown space and elevated boundaries.

    terrain cells have a locally flat supporting surface; vertical objects,
    excessive steps, drops and steep surfaces are blocked cells.
    """
    c = config
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 points")
    points = points[np.isfinite(points).all(axis=1)]
    plane = fit_ground(points, c)
    empty = np.empty((0, 3))
    if plane is None:
        return dict(state="STOP", reason="ground_reference_unavailable", coverage=0.,
                    nearest_obstacle_m=None, obstacle_points=empty, plane=None)
    half = c.robot_width_m / 2 + c.safety_margin_m
    nx = math.floor((c.forward_max_m - c.forward_min_m) / c.cell_m + 1e-9)
    ny = math.ceil(2 * c.perception_half_width_m / c.cell_m)
    lateral_max = ny * c.cell_m / 2
    selected = points[(points[:, 0] >= c.forward_min_m) & (points[:, 0] < c.forward_max_m)
                      & (np.abs(points[:, 1]) < lateral_max)]
    keys = np.floor(np.column_stack(((selected[:, 0] - c.forward_min_m) / c.cell_m,
                                    (selected[:, 1] + lateral_max) / c.cell_m))).astype(int)
    cells = {}
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    ordered_keys = keys[order]
    cuts = np.flatnonzero(np.any(np.diff(ordered_keys, axis=0), axis=1)) + 1
    for indexes in np.split(order, cuts):
        if len(indexes) == 0:
            continue
        key = keys[indexes[0]]
        pts = selected[indexes]
        if len(pts) < c.min_points_cell:
            continue
        # Vertical residual is used so thresholds correspond to step height.
        h = pts[:, 2] - (pts[:, :2] @ plane[:2] + plane[2])
        low, high = np.percentile(h, [20, 95])
        surface = pts[h <= np.percentile(h, 40) + c.plane_tolerance_m]
        local_slope = 0.
        roughness = float(np.percentile(h, 80) - np.percentile(h, 20))
        if len(surface) >= 3 and np.linalg.matrix_rank(surface[:, :2] - surface[:, :2].mean(axis=0)) == 2:
            coef = np.linalg.lstsq(np.column_stack((surface[:, :2], np.ones(len(surface)))), surface[:, 2], rcond=None)[0]
            local_slope = math.degrees(math.atan(np.linalg.norm(coef[:2])))
            residual = surface[:, 2] - np.column_stack((surface[:, :2], np.ones(len(surface)))) @ coef
            roughness = float(np.percentile(residual, 90) - np.percentile(residual, 10))
        upright = np.any((h > c.max_step_up_m) & (h <= c.collision_height_m))
        blocked = (low > c.max_step_up_m or low < -c.max_step_down_m or upright
                   or local_slope > c.max_slope_deg or roughness > c.max_roughness_m)
        terrain = abs(low) >= c.terrain_trigger_m or local_slope >= c.max_slope_deg * .5
        center = [c.forward_min_m + (key[0] + .5) * c.cell_m,
                  -lateral_max + (key[1] + .5) * c.cell_m, c.ground_z_m + c.max_step_up_m + .01]
        cells[tuple(key)] = (blocked, terrain, center)
    # Require each forward strip to have sufficient observed ground support.
    corridor_columns = [j for j in range(ny) if abs(-lateral_max + (j + .5) * c.cell_m) <= half + 1e-9]
    coverage = min((sum((i, j) in cells for j in corridor_columns) / len(corridor_columns)
                    for i in range(nx)), default=0.)
    # Elevated side boundaries remain route constraints even when their height
    # could be crossed at a designated transition in the forward corridor.
    obstacles = np.asarray([v[2] for key, v in cells.items()
                            if v[0] or (v[1] and key[1] not in corridor_columns)], dtype=float).reshape(-1, 3)
    corridor_cells = [value for key, value in cells.items() if key[1] in corridor_columns]
    corridor_obstacles = [v[2] for v in corridor_cells if v[0]]
    nearest = float(min(p[0] for p in corridor_obstacles) - c.cell_m / 2) if corridor_obstacles else None
    state, reason = "NORMAL", "regular_ground"
    if coverage < c.min_coverage:
        state, reason = "STOP", "insufficient_ground_support"
    elif corridor_obstacles:
        state, reason = "AVOID", "blocked_corridor"
    elif any(v[1] for v in corridor_cells):
        state, reason = "TERRAIN", "traversable_surface_transition"
    return dict(state=state, reason=reason, coverage=coverage,
                nearest_obstacle_m=nearest, obstacle_points=obstacles, plane=plane)


class DepthSupervisor:
    """Immediate hazard response; persistent gait switching and stale-input stop."""
    def __init__(self, config):
        self.config = config
        self.mode = "normal"
        self.pending = None
        self.count = 0
        self.timestamp = None
        self.last = None

    def update(self, assessment, timestamp_s, speed_mps=0.):
        if not math.isfinite(timestamp_s) or not math.isfinite(speed_mps):
            raise ValueError("Timestamp and speed must be finite")
        if self.timestamp is not None and timestamp_s <= self.timestamp:
            raise ValueError("Depth timestamps must increase")
        c = self.config
        state = assessment["state"]
        nearest = assessment["nearest_obstacle_m"]
        stopping_distance = abs(speed_mps) * c.reaction_time_s + speed_mps ** 2 / (2 * c.braking_accel_mps2) + c.safety_margin_m
        if nearest is not None and nearest <= stopping_distance:
            state = "STOP"
        target = "terrain" if state == "TERRAIN" else "normal" if state == "NORMAL" else None
        if target is not None and target != self.mode:
            self.count = self.count + 1 if self.pending == target else 1
            self.pending = target
            required = c.enter_frames if target == "terrain" else c.exit_frames
            if self.count >= required:
                self.mode, self.pending, self.count = target, None, 0
        else:
            self.pending, self.count = None, 0
        # Pause while waiting for confirmation before entering terrain mode.
        if target == "terrain" and self.mode != "terrain":
            state = "STOP"
        limit = 0. if state == "STOP" else c.terrain_speed_mps if state in ("TERRAIN", "AVOID") or self.mode == "terrain" else c.normal_speed_mps
        self.timestamp = timestamp_s
        self.last = dict(state=state, gait_mode=self.mode, speed_limit_mps=limit,
                         stopping_distance_m=stopping_distance, reason=assessment["reason"],
                         coverage=assessment["coverage"], nearest_obstacle_m=nearest)
        return self.last.copy()

    def command(self, now_s):
        if self.last is None or not math.isfinite(now_s) or now_s < self.timestamp or now_s - self.timestamp > self.config.stale_s:
            return dict(state="STOP", gait_mode=self.mode, speed_limit_mps=0., reason="stale_depth")
        return self.last.copy()


def gate_velocity(vx, vy, omega, decision, measured_speed, config, fresh, acknowledged,
                  max_turn_radps=.3):
    """Apply the depth permit to a planner command without changing its curvature."""
    if (not fresh or not acknowledged or vx <= 0 or abs(vy) > 1e-6
            or not all(math.isfinite(v) for v in (vx, vy, omega, measured_speed))):
        return 0., 0.
    speed = max(abs(vx), abs(measured_speed))
    stop_distance = speed * config.reaction_time_s + speed ** 2 / (2 * config.braking_accel_mps2) + config.safety_margin_m
    nearest = decision.get("nearest_obstacle_m")
    if nearest is not None and nearest <= stop_distance:
        return 0., 0.
    ratio = min(1., decision["speed_limit_mps"] / vx)
    # Scaling both components preserves the local planner's selected curvature.
    if abs(omega) > max_turn_radps:
        ratio = min(ratio, max_turn_radps / abs(omega))
    return vx * ratio, omega * ratio

