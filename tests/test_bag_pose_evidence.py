import unittest

from scripts.export_rosbag_pose_evidence import build_pose_rows


class BagPoseEvidenceTests(unittest.TestCase):
    def test_builds_image_free_nearest_pose_trace(self):
        rows, rejected = build_pose_rows(
            [(10.01, 7), (10.09, 8)],
            [
                {"bag_time_s": 10.0, "latitude": 31.0, "longitude": 121.0, "height_m": 2.0,
                 "velocity_east_mps": 1.0, "velocity_north_mps": 0.0, "solution_type": 16},
                {"bag_time_s": 10.1, "latitude": 31.000001, "longitude": 121.000001, "height_m": 2.1,
                 "velocity_east_mps": 0.0, "velocity_north_mps": 1.0, "solution_type": 16},
            ],
            .02,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rejected, [])
        self.assertEqual(rows[0]["pose_time_s"], 10.0)
        self.assertIn("x_east_m", rows[1])
        self.assertNotIn("image", rows[0])

    def test_rejects_trigger_outside_time_tolerance(self):
        rows, rejected = build_pose_rows(
            [(11.0, 9)],
            [{"bag_time_s": 10.0, "latitude": 31.0, "longitude": 121.0, "height_m": 2.0,
              "velocity_east_mps": 1.0, "velocity_north_mps": 0.0, "solution_type": 16}],
            .05,
        )
        self.assertEqual(rows, [])
        self.assertEqual(rejected[0]["reason"], "nearest_rtk_exceeds_tolerance")


if __name__ == "__main__":
    unittest.main()
