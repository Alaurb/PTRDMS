import sys
import unittest
import queue
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

from app import ROOT, build_command, PrmsApp


class AppTests(unittest.TestCase):
    def test_explicit_classifier_size_is_forwarded(self):
        command = build_command(sys.executable, 'images', 'map', 'out', 'det', 'cls', '.25', '0', classifier_imgsz='224')
        self.assertEqual(command[command.index('--classifier-imgsz')+1], '224')

    def test_camera_yaw_offset_is_forwarded(self):
        command = build_command(sys.executable, 'images', 'map', 'out', 'det', 'cls', '.25', '0', camera_yaw_offset_deg='-90')
        self.assertEqual(command[command.index('--camera-yaw-offset-deg')+1], '-90')

    def test_timestamp_pose_mode_is_forwarded(self):
        command = build_command(
            sys.executable, "images", "map", "out", "det", "cls", ".25", "0",
            pose_csv="poses.csv", pose_match_mode="timestamp", frame_times_csv="frame_times.csv",
            pose_timestamp_tolerance_s="0.12",
        )
        self.assertEqual(command[command.index("--pose-match-mode") + 1], "timestamp")
        self.assertEqual(command[command.index("--frame-times-csv") + 1], "frame_times.csv")
        self.assertEqual(command[command.index("--pose-timestamp-tolerance-s") + 1], "0.12")

    def test_start_is_guarded_before_worker_launch(self):
        app = PrmsApp.__new__(PrmsApp)
        app.busy = False
        app.status = Mock(); app.run_button = Mock(); app._append = Mock()
        with patch('app.threading.Thread') as thread:
            app._start(['python'], 'starting')
            app._start(['python'], 'starting again')
            self.assertTrue(app.busy)
            self.assertEqual(thread.call_count, 1)

    def test_failed_process_does_not_replace_successful_result(self):
        app = PrmsApp.__new__(PrmsApp)
        app.messages = queue.Queue(); app.messages.put(('__DONE__',1))
        app.busy = True; app.process = object()
        app.pending_result = Path('failed/index.html')
        app.last_result = Path('successful/index.html')
        app.status = Mock(); app.run_button = Mock(); app.root = SimpleNamespace(after=Mock())
        app._drain_messages()
        self.assertEqual(app.last_result,Path('successful/index.html'))
        self.assertIsNone(app.pending_result)
        self.assertFalse(app.busy)

    def test_build_command_includes_requested_inputs(self):
        command = build_command(
            sys.executable, "panoramas", "map.jpg", "out", "detector.pt", "classifier.pt",
            "0.25", "0", "poses.csv", "ranges.json", True,
        )
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(Path(command[1]), ROOT / "demo.py")
        self.assertIn("--pose-csv", command)
        self.assertIn("poses.csv", command)
        self.assertIn("--range-manifest", command)
        self.assertIn("--export-six-faces", command)

    def test_build_command_omits_optional_inputs(self):
        command = build_command(sys.executable, "panoramas", "map.jpg", "out", "detector.pt", "classifier.pt", "0.25", "1")
        self.assertNotIn("--pose-csv", command)
        self.assertNotIn("--range-manifest", command)


if __name__ == "__main__":
    unittest.main()
