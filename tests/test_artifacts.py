import json
import tempfile
import unittest
from pathlib import Path

from ripeness_demo.artifacts import SCHEMA, build_evidence_manifest, write_evidence_manifest
from ripeness_demo.mapping import Pose


class ArtifactTests(unittest.TestCase):
    def test_replay_manifest_disallows_current_model_claim(self):
        pose = Pose("frame.jpg", 0, 0, 0, 0, 0, 0, "synthetic_map_path", 1)
        manifest = build_evidence_manifest(
            {"processed_frames": 1, "detections": 2}, [pose],
            processing_mode="saved_prediction_replay", range_source="assumed_row_plane",
        )
        self.assertEqual(manifest["schema"], SCHEMA)
        self.assertEqual(manifest["status"], "archived_legacy_replay")
        self.assertIn("current-model performance", manifest["not_established"])

    def test_measured_manifest_requires_pose_and_range(self):
        pose = Pose("frame.jpg", 0, 0, 0, 0, 0, 0, "pose_csv", "rowA")
        manifest = build_evidence_manifest(
            {"processed_frames": 1, "detections": 2}, [pose],
            processing_mode="offline_batch", range_source="registered_measured_range",
        )
        self.assertEqual(manifest["status"], "measured_geometry_candidate")
        with tempfile.TemporaryDirectory() as directory:
            write_evidence_manifest(Path(directory), manifest)
            self.assertEqual(json.loads((Path(directory) / "evidence.json").read_text())["schema"], SCHEMA)


if __name__ == "__main__":
    unittest.main()
