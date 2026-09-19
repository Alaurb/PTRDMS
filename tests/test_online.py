import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ripeness_demo.online import TimedPoseBuffer, timestamp_key


class TimedPoseBufferTests(unittest.TestCase):
    def test_exact_timestamp_keys_prevent_frame_id_overwrite(self):
        buffer = TimedPoseBuffer((100, 50), 10.0)
        self.assertTrue(buffer.add(seconds=42, nanoseconds=7, frame_id="map", x=1, y=2, z=1.15, yaw=0))
        self.assertTrue(buffer.add(seconds=42, nanoseconds=8, frame_id="map", x=3, y=2, z=1.15, yaw=0))
        pose = buffer.match(seconds=42, nanoseconds=8, frame_name="frame.jpg", tolerance_s=.01)
        self.assertEqual(timestamp_key(42, 8), "42.000000008")
        self.assertIsNotNone(pose)
        self.assertEqual(pose.frame, "frame.jpg")
        self.assertEqual(pose.x, 3)

    def test_duplicate_stamp_is_retained_and_ambiguous_match_is_rejected(self):
        buffer = TimedPoseBuffer((100, 50), 10.0)
        self.assertTrue(buffer.add(seconds=10, nanoseconds=0, frame_id="map", x=1, y=2, z=1.15, yaw=0))
        self.assertFalse(buffer.add(seconds=10, nanoseconds=0, frame_id="map", x=99, y=2, z=1.15, yaw=0))
        self.assertEqual(buffer.duplicate_timestamp_count, 1)
        self.assertTrue(buffer.add(seconds=10, nanoseconds=20_000_000, frame_id="map", x=2, y=2, z=1.15, yaw=0))
        self.assertIsNone(buffer.match(seconds=10, nanoseconds=10_000_000, frame_name="ambiguous.jpg", tolerance_s=.02))
        pose = buffer.match(seconds=10, nanoseconds=1_000_000, frame_name="near.jpg", tolerance_s=.02)
        self.assertIsNotNone(pose)
        self.assertEqual(pose.x, 1)


if __name__ == "__main__":
    unittest.main()
