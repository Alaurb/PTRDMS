import unittest
import tempfile
from dataclasses import replace
from pathlib import Path

from ripeness_demo.detectors import Detection
from ripeness_demo.mapping import Pose, load_frame_times_csv, match_poses_by_timestamp, project_detection
from ripeness_demo.evidence import associate


class SoftSynchronizationTests(unittest.TestCase):
    def test_closest_image_wins_pose_conflict_independent_of_filename_order(self):
        pose = Pose("pose", 1, 2, 1.15, 0, 10, 10, "pose_csv", "rowA", 10.0)
        frames = [Path("a.jpg"), Path("z.jpg")]
        for order in (frames, frames[::-1]):
            rejected = []
            pairs = match_poses_by_timestamp(order, {"pose": pose},
                                             {"a.jpg": 10.04, "z.jpg": 10.01}, .05, rejected=rejected)
            self.assertIn("z.jpg", pairs)
            self.assertNotIn("a.jpg", pairs)
            self.assertEqual(rejected[0]["reason"], "pose_already_matched")

    def test_missing_timestamp_is_skipped_and_tolerance_boundary_is_accepted(self):
        pose = Pose("pose", 1, 2, 1.15, 0, 10, 10, "pose_csv", 1, 10.0)
        rejected = []
        pairs = match_poses_by_timestamp([Path("missing.jpg"), Path("edge.jpg")],
                                        {"pose": pose}, {"edge.jpg": 10.05}, .05, rejected=rejected)
        self.assertIn("edge.jpg", pairs)
        self.assertEqual(rejected[0]["reason"], "missing_frame_timestamp")

    def test_invalid_matching_configuration_is_not_silently_skipped(self):
        pose = Pose("pose", 1, 2, 1.15, 0, 10, 10, "pose_csv", 1)
        with self.assertRaises(ValueError):
            match_poses_by_timestamp([], {"pose": pose}, {}, .05)
        with self.assertRaises(ValueError):
            match_poses_by_timestamp([], {"pose": pose}, {}, -1)

    def test_empty_image_timestamp_row_does_not_interrupt_other_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "times.csv"
            csv_path.write_text("frame,timestamp_s\nempty.jpg,\nvalid.jpg,10.01\n", encoding="utf-8")
            times = load_frame_times_csv(csv_path)
            pose = Pose("pose", 1, 2, 1.15, 0, 10, 10, "pose_csv", 1, 10.0)
            rejected = []
            pairs = match_poses_by_timestamp([Path("empty.jpg"), Path("valid.jpg")],
                                            {"pose": pose}, times, .05, rejected=rejected)
            self.assertIn("valid.jpg", pairs)
            self.assertEqual(rejected[0]["reason"], "missing_frame_timestamp")


class SpatialCellTests(unittest.TestCase):
    def fixtures(self):
        p1 = Pose("a.jpg", 1, 2, 1.15, 0, 10, 10, "pose_csv", "rowA")
        p2 = replace(p1, frame="b.jpg")
        d1 = project_detection(Detection("mature", .9, (45, 45, 55, 55), "fixture"),
                               "left", p1, (101, 101), (100, 100))
        d1.x, d1.y, d1.z = .01, .01, 1.21
        return p1, p2, d1, replace(d1, frame="b.jpg", x=.09)

    def test_projected_observations_share_cells_and_keep_one_to_one_frame_links(self):
        p1, p2, d1, d2 = self.fixtures()
        tracks, links = associate([d1, d2], [p1, p2], cell_size_m=.12, require_measured=False)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["observations"], 2)
        self.assertEqual(tracks[0]["status"], "projected_association")
        self.assertTrue(links[1]["matched_previous"])
        tracks, _ = associate([d1, replace(d1), d2], [p1, p2], cell_size_m=.12, require_measured=False)
        self.assertEqual(len(tracks), 2)

    def test_nearby_detections_in_different_cells_remain_separate(self):
        p1, p2, d1, d2 = self.fixtures()
        d1.x, d2.x = .119, .121
        tracks, _ = associate([d1, d2], [p1, p2], cell_size_m=.12, require_measured=False)
        self.assertEqual(len(tracks), 2)

    def test_synthetic_paths_are_not_merged_and_measured_mode_remains_available(self):
        p1, p2, d1, d2 = self.fixtures()
        tracks, _ = associate([d1, d2], [replace(p1, source="synthetic_map_path"), p2],
                              cell_size_m=.12, require_measured=False)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(associate([d1, d2], [p1, p2])[0]), 2)
        with self.assertRaises(ValueError):
            associate([d1], [p1], cell_size_m=0)


if __name__ == "__main__":
    unittest.main()
