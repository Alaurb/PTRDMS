import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
from ripeness_demo.depth_navigation import DepthConfig, DepthSupervisor, depth_to_points, assess_points, gate_velocity
from ripeness_demo.unitree_gait import UnitreeGaitAdapter


def configuration():
    # Synthetic fixture limits, not measured Go2-W specifications.
    return DepthConfig(robot_width_m=.4, safety_margin_m=.1, ground_z_m=-.5,
                       max_step_up_m=.16, max_step_down_m=.12, max_slope_deg=25.,
                       max_roughness_m=.08, collision_height_m=1.2, terrain_trigger_m=.04,
                       normal_speed_mps=.4, terrain_speed_mps=.15, reaction_time_s=.3,
                       braking_accel_mps2=.5, min_points_cell=3)


def scene(config, height=0., obstacle=False, drop=False):
    x, y = np.meshgrid(np.arange(.26, 2., .02), np.arange(-.299, .3, .02))
    z = np.full_like(x, config.ground_z_m)
    z[x >= 1.05] += height
    if drop:
        z[x >= 1.05] -= .3
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    if obstacle:
        elevated = points[(points[:, 0] > 1.05) & (points[:, 0] < 1.25)].copy()
        elevated[:, 2] += .8
        points = np.vstack((points, elevated))
    return points


class DepthNavigationTests(unittest.TestCase):
    def test_depth_units_invalid_pixels_and_optical_transform(self):
        depth = np.array([[1000, 0], [2000, 65535]], dtype=np.uint16)
        transform = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, .5], [0, 0, 0, 1.]])
        mm = depth_to_points(depth, (1, 1, 0, 0), transform, stride=1)
        metres = depth.astype(np.float32) / 1000
        metres[0, 1] = np.nan
        other = depth_to_points(metres, (1, 1, 0, 0), transform, "32FC1", stride=1)
        np.testing.assert_allclose(mm, [[1, 0, .5], [2, 0, -1.5]])
        np.testing.assert_allclose(mm, other)

    def test_flat_step_pedestrian_and_drop_have_distinct_decisions(self):
        c = configuration()
        for kwargs, expected in [({}, "NORMAL"), ({"height": .1}, "TERRAIN"),
                                 ({"height": .3}, "AVOID"), ({"obstacle": True}, "AVOID"),
                                 ({"drop": True}, "AVOID")]:
            with self.subTest(kwargs=kwargs):
                result = assess_points(scene(c, **kwargs), c)
                self.assertEqual(result["state"], expected)
                if expected == "AVOID":
                    self.assertGreater(len(result["obstacle_points"]), 0)

    def test_no_depth_and_missing_ground_do_not_become_free_space(self):
        c = configuration()
        self.assertEqual(assess_points(np.empty((0, 3)), c)["state"], "STOP")
        pts = scene(c)
        pts = pts[~((pts[:, 0] > 1.2) & (pts[:, 0] < 1.5))]
        self.assertEqual(assess_points(pts, c)["state"], "STOP")

    def test_pitch_compensation_preserves_geometric_result(self):
        c = configuration()
        pts = scene(c, height=.1)
        angle = .2
        r = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        body_points = pts @ r.T
        compensated = body_points @ r
        self.assertEqual(assess_points(compensated, c)["state"], "TERRAIN")

    def test_hysteresis_close_obstacle_and_stale_stop(self):
        c = configuration()
        supervisor = DepthSupervisor(c)
        terrain = assess_points(scene(c, height=.1), c)
        self.assertEqual(supervisor.update(terrain, 1.)["speed_limit_mps"], 0)
        self.assertEqual(supervisor.update(terrain, 1.1)["gait_mode"], "normal")
        self.assertEqual(supervisor.update(terrain, 1.2)["gait_mode"], "terrain")
        flat = assess_points(scene(c), c)
        for i in range(4):
            self.assertEqual(supervisor.update(flat, 1.3 + i * .1)["gait_mode"], "terrain")
        self.assertEqual(supervisor.update(flat, 1.7)["gait_mode"], "normal")
        hazard = dict(terrain, state="AVOID", nearest_obstacle_m=.1)
        self.assertEqual(supervisor.update(hazard, 1.8, .4)["state"], "STOP")
        self.assertEqual(supervisor.command(2.2)["state"], "STOP")
        with self.assertRaises(ValueError):
            supervisor.update(flat, 1.8)

    def test_invalid_calibration_and_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(configuration(), braking_accel_mps2=0.)
        with self.assertRaises(ValueError):
            depth_to_points(np.ones((2, 2), dtype=np.uint16), (0, 1, 0, 0), np.eye(4))
        with self.assertRaises(ValueError):
            depth_to_points(np.ones((2, 2), dtype=np.uint16), (1, 1, 0, 0), np.eye(4) * 2)

    def test_side_boundary_is_mapped_without_blocking_centered_route(self):
        c = configuration()
        pts = scene(c)
        side = pts.copy()
        side[:, 1] += .8
        side[:, 2] += .8
        result = assess_points(np.vstack((pts, side)), c)
        self.assertEqual(result["state"], "NORMAL")
        self.assertGreater(len(result["obstacle_points"]), 0)

    def test_velocity_gate_preserves_curvature_and_requires_feedback(self):
        c = configuration()
        decision = dict(speed_limit_mps=.2, nearest_obstacle_m=None)
        self.assertEqual(gate_velocity(.4, 0, .2, decision, 0, c, True, True), (.2, .1))
        for fresh, ack in ((False, True), (True, False)):
            self.assertEqual(gate_velocity(.4, 0, .2, decision, 0, c, fresh, ack), (0, 0))
        self.assertEqual(gate_velocity(-.1, 0, 0, decision, 0, c, True, True), (0, 0))
        self.assertEqual(gate_velocity(.1, .1, 0, decision, 0, c, True, True), (0, 0))
        self.assertEqual(gate_velocity(0, 0, .2, decision, 0, c, True, True), (0, 0))
        decision["nearest_obstacle_m"] = .2
        self.assertEqual(gate_velocity(.4, 0, 0, decision, .4, c, True, True), (0, 0))

    def test_unitree_adapter_retries_failure_and_uses_actual_telemetry(self):
        class Client:
            def __init__(self):
                self.calls = []
                self.code = 1
            def SwitchGait(self, value):
                self.calls.append(value)
                return self.code
        client = Client()
        adapter = UnitreeGaitAdapter(client, 4, 7)
        self.assertFalse(adapter.request("terrain"))
        client.code = 0
        self.assertTrue(adapter.request("terrain"))
        self.assertEqual(client.calls, [7, 7])
        self.assertEqual(adapter.observe(4), "normal")
        self.assertTrue(adapter.request("terrain"))
        with self.assertRaises(ValueError):
            UnitreeGaitAdapter(object(), 0, 1)

    def test_reported_half_metre_limit_is_configurable(self):
        c = replace(configuration(), max_step_up_m=.5)
        self.assertEqual(assess_points(scene(c, height=.45), c)["state"], "TERRAIN")
        self.assertEqual(assess_points(scene(c, height=.55), c)["state"], "AVOID")

    def test_replay_raw_depth_writes_decisions_and_obstacle_arrays(self):
        c = replace(configuration(), stride=1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            v = np.arange(240)[:, None]
            depth = np.broadcast_to(50. / np.maximum(v, 1), (240, 640)).copy().astype(np.float32)
            depth[0] = np.nan
            np.save(root / "depth.npy", depth)
            spec = dict(geometry=c.__dict__, intrinsics=[180., 100., 320., 0.], encoding="32FC1",
                        optical_to_terrain=[[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            (root / "config.json").write_text(json.dumps(spec), encoding="utf-8")
            records = [dict(depth_npy="depth.npy", timestamp_s=1. + i * .1, speed_mps=.2) for i in range(3)]
            (root / "frames.json").write_text(json.dumps(records), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "scripts/replay_depth_navigation.py"
            run = subprocess.run([sys.executable, str(script), "--config", str(root / "config.json"),
                                  "--frames", str(root / "frames.json"), "--output", str(root / "out")],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            decisions = json.loads((root / "out/decisions.json").read_text())
            self.assertEqual([record["state"] for record in decisions], ["NORMAL"] * 3)
            self.assertEqual(len(list((root / "out").glob("obstacles_*.npy"))), 3)


if __name__ == "__main__":
    unittest.main()
