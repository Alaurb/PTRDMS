import math
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ripeness_demo.detectors import ColorShapeDetector, Detection, _merge_box_candidates, class_order_for
from ripeness_demo.mapping import Pose, SpatialDetection, generate_demo_poses, project_detection
from ripeness_demo.panorama import extract_cube_face, extract_side_views
from ripeness_demo.report import HTML_TEMPLATE, _plant_columns


class CoreTests(unittest.TestCase):
    def test_panorama_extracts_square_side_views(self):
        panorama = Image.fromarray(np.tile(np.arange(240, dtype=np.uint8), (120, 1)))
        panorama = panorama.convert("RGB")
        sides = extract_side_views(panorama, 96)
        self.assertEqual(sides["left"].size, (96, 96))
        self.assertEqual(sides["right"].size, (96, 96))
        self.assertFalse(np.array_equal(np.asarray(sides["left"]), np.asarray(sides["right"])))

    def test_camera_yaw_offset_aligns_logical_left_with_native_front(self):
        panorama = Image.fromarray(np.tile(np.arange(240, dtype=np.uint8), (120, 1))).convert("RGB")
        native_front = extract_cube_face(panorama, "front", 96)
        robot_left = extract_cube_face(panorama, "left", 96, camera_yaw_offset_rad=-math.pi / 2)
        self.assertTrue(np.array_equal(np.asarray(native_front), np.asarray(robot_left)))

    def test_color_detector_marks_yellow_round_region_harvest_ready(self):
        image = Image.new("RGB", (240, 240), "#193b24")
        draw = ImageDraw.Draw(image)
        draw.ellipse((75, 70, 150, 145), fill="#dfbb2b")
        detections = ColorShapeDetector(analysis_size=240).detect(image)
        self.assertTrue(any(item.class_name == "harvest_ready" for item in detections))

    def test_brightness_ensemble_merges_overlapping_detector_boxes(self):
        candidates = [((10.0, 10.0, 50.0, 50.0), 0.70), ((12.0, 11.0, 51.0, 49.0), 0.82)]
        merged = _merge_box_candidates(candidates)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][1], 0.82)

    def test_paper_four_stage_classes_select_paper_report_schema(self):
        self.assertEqual(
            class_order_for(["immature", "mature"]),
            ("mature", "discoloration", "green_mature", "immature"),
        )

    def test_projection_places_left_detection_to_left_of_heading(self):
        pose = Pose("frame.jpg", 5.0, 5.0, 0.0, 0.0, 100, 100, "test", 1)
        detection = Detection("mature", 0.9, (45, 45, 55, 55), "test")
        point = project_detection(detection, "left", pose, (101, 101), (400, 200), map_width_m=20)
        self.assertAlmostEqual(point.x, 5.0, places=4)
        self.assertAlmostEqual(point.y, 6.2, places=4)
        self.assertAlmostEqual(point.z, 1.15, places=4)

    def test_decimal_panorama_names_remain_one_contiguous_route(self):
        paths = [Path(f"11.29-panoramic{number}.0.jpg") for number in range(702, 715)]
        map_image = Image.new("RGB", (700, 432), "white")
        poses = generate_demo_poses(paths, map_image)
        routes = {poses[path.name].route for path in paths}
        self.assertEqual(routes, {1})

    def test_plant_columns_use_one_cube_per_detection(self):
        pose = Pose("11.29-panoramic702.0.jpg", 5.0, 8.0, 0.0, 0.0, 100, 100, "test", 1)
        detections = [
            SpatialDetection(
                pose.frame, "left", class_name, 0.9, "test", 5.0, 9.15, 1.2,
                100, 100, 10, 10, 20, 20, 100, 100, "annotated.jpg",
            )
            for class_name in ("immature", "mature", "green_mature")
        ]
        plants = _plant_columns([pose], detections)
        self.assertEqual(len(plants), 2)
        self.assertEqual(plants[0]["plant_id"], "P702-L")
        self.assertEqual(plants[0]["total"], 3)
        self.assertEqual(
            [cube["class_name"] for cube in plants[0]["cubes"]],
            ["mature", "green_mature", "immature"],
        )
        self.assertEqual(plants[1]["total"], 0)

    def test_left_and_right_map_markers_are_on_opposite_sides_of_route(self):
        pose = Pose("11.29-panoramic702.0.jpg", 5.0, 8.0, 0.0, 0.0, 100, 100, "test", 1)
        plants = _plant_columns([pose], [], map_size=(400, 200), map_width_m=20.0)
        self.assertLess(plants[0]["map_py"], pose.map_py)
        self.assertGreater(plants[1]["map_py"], pose.map_py)
        self.assertGreater(plants[1]["map_py"] - plants[0]["map_py"], 40)

    def test_html_report_is_english_only(self):
        self.assertIn('<html lang="en">', HTML_TEMPLATE)
        self.assertIn("3D tomato-bunch view", HTML_TEMPLATE)
        self.assertIn("let coordinateMode=summary.range_source==='registered_measured_range'", HTML_TEMPLATE)
        self.assertIn("Estimated coordinates", HTML_TEMPLATE)
        self.assertIn("not reconstructed stems", HTML_TEMPLATE)
        self.assertIsNone(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", HTML_TEMPLATE))


if __name__ == "__main__":
    unittest.main()
