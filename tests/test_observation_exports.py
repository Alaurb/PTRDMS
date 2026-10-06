import csv
import json
import tempfile
import unittest
from pathlib import Path

from ripeness_demo.artifacts import build_evidence_manifest
from ripeness_demo.detectors import Detection
from ripeness_demo.mapping import Pose, project_detection
from ripeness_demo.report import write_csv_files, write_summary


class ObservationExportTests(unittest.TestCase):
    def test_repeated_location_retains_frame_side_observations(self):
        poses = [
            Pose(frame, 5, 5, 1.15, 0, 100, 100, "pose_csv", "rowA")
            for frame in ("day1-panoramic1.jpg", "day2-panoramic1.jpg")
        ]
        detections = [
            project_detection(
                Detection("mature", .9, (45, 45, 55, 55), "test_fixture"),
                "left", pose, (101, 101), (400, 200),
            )
            for pose in poses
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_csv_files(output, poses, detections, taxonomy="paper_four_stage")
            with (output / "observations.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({row["observation_id"] for row in rows}), 4)
            self.assertEqual(sum(int(row["detection_count"]) for row in rows), 2)
            self.assertEqual([int(row["detection_count"]) for row in rows], [1, 0, 1, 0])
            self.assertTrue(all(row["assignment_source"] == "frame_side_proxy" for row in rows))
            # Existing consumers still receive the legacy file and its schema.
            with (output / "plants.csv").open(encoding="utf-8-sig", newline="") as stream:
                legacy = list(csv.DictReader(stream))
            self.assertEqual(len(legacy), len(rows))
            self.assertEqual([int(row["total"]) for row in legacy], [1, 0, 1, 0])
            summary = write_summary(output, poses, detections, "test_fixture", "pose_csv", 2,
                                    taxonomy="paper_four_stage")
            saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["count_unit"], "detection_observation")
            self.assertEqual(saved["grouping_unit"], "frame_side")
            self.assertEqual(summary["detections"], 2)
            self.assertFalse(saved["unique_fruit_count_validated"])
            self.assertFalse(saved["unique_plant_assignment_validated"])

    def test_empty_export_has_headers_and_no_invented_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_csv_files(output, [], [], taxonomy="paper_four_stage")
            with (output / "observations.csv").open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertIn("observation_id", reader.fieldnames)
                self.assertIn("detection_count", reader.fieldnames)
                self.assertEqual(list(reader), [])

    def test_measured_ranges_do_not_validate_plant_or_fruit_counts(self):
        pose = Pose("frame.jpg", 5, 5, 1.15, 0, 100, 100, "pose_csv", "rowA")
        manifest = build_evidence_manifest(
            {}, [pose], processing_mode="offline_batch", range_source="registered_measured_range"
        )
        self.assertEqual(manifest["status"], "measured_geometry_candidate")
        self.assertIn("unique-plant identity or per-plant fruit counts", manifest["not_established"])
        self.assertIn("unique-fruit count or biological truss identity", manifest["not_established"])


if __name__ == "__main__":
    unittest.main()
