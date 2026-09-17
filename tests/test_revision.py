import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

from ripeness_demo.detectors import Detection
from ripeness_demo.mapping import Pose, load_pose_csv, project_detection
from ripeness_demo.panorama import extract_all_faces
from ripeness_demo.evidence import RangeEvidence, foreground_range, associate
from scripts.evaluate_spatial import evaluate


class RevisionTests(unittest.TestCase):
    def test_pipeline_measured_range_outputs_and_rejections(self):
        import demo
        class FixtureDetector:
            name = "test_fixture_not_yolo"
            def detect(self, image):
                return [Detection("mature",.9,(40,40,60,60),self.name),
                        Detection("immature",.8,(75,40,95,60),self.name)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"input").mkdir()
            Image.new("RGB",(202,101),"green").save(root/"input/a.jpg")
            Image.new("RGB",(100,100),"white").save(root/"map.png")
            (root/"poses.csv").write_text("frame,x,y,z,yaw,route\na.jpg,5,5,1.2,0,rowA\n")
            ranges=np.full((101,101),1.15)
            ranges[:,75:]=3.0
            np.save(root/"range.npy",ranges)
            views=[dict(frame="a.jpg",side=s,file="range.npy",image_time_s=1,range_time_s=1,pose_time_s=1) for s in ("left","right")]
            (root/"range.json").write_text(json.dumps(dict(calibration_id="fixture",range_convention="radial_metres",views=views)))
            argv=["demo.py","--input",str(root/"input"),"--map",str(root/"map.png"),"--output",str(root/"out"),"--pose-csv",str(root/"poses.csv"),"--range-manifest",str(root/"range.json"),"--face-size","101"]
            with patch("sys.argv",argv), patch("demo.build_detector",return_value=FixtureDetector()):
                demo.run(demo.parse_args())
            summary=json.loads((root/"out/summary.json").read_text())
            self.assertEqual(summary["detections"],2)
            self.assertEqual(summary["rejected_observations"],2)
            self.assertEqual(summary["association_status"],"spatial_candidates")
            self.assertTrue((root/"out/runtime.json").exists())
            html=(root/"out/index.html").read_text()
            self.assertIn("3D tomato-bunch view",html)

    def point(self, frame="a.jpg", x=0, source="pose_csv", range_m=1):
        pose = Pose(frame,x,0,1.2,0,0,0,source,"rowA")
        detection = Detection("mature",.9,(45,45,55,55),"test")
        return pose, project_detection(detection,"left",pose,(101,101),(100,100),range_m=range_m)

    def test_projection_rays_and_height(self):
        pose, point = self.point()
        self.assertAlmostEqual(point.x,0)
        self.assertAlmostEqual(point.y,1)
        self.assertAlmostEqual(point.z,1.2)
        upper = project_detection(Detection("mature",.9,(70,20,80,30),"test"),"left",pose,(101,101),(100,100),row_offset_m=2)
        self.assertAlmostEqual(upper.x,1)
        self.assertAlmostEqual(upper.y,2)
        self.assertAlmostEqual(upper.z,2.2)

    def test_outside_map_is_not_clipped(self):
        pose, _ = self.point(x=-10)
        p = project_detection(Detection("mature",.9,(45,45,55,55),"test"),"left",pose,(101,101),(100,100))
        self.assertLess(p.x,0)
        self.assertLess(p.map_px,0)

    def test_pose_string_route_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/"pose.csv"
            file.write_text("frame,x,y,z,yaw,route\na.jpg,0,0,1.2,0,rowA\n")
            self.assertEqual(load_pose_csv(file,Image.new("RGB",(100,100)))["a.jpg"].route,"rowA")
            file.write_text(file.read_text()+"a.jpg,0,0,1.2,0,rowA\n")
            with self.assertRaises(ValueError): load_pose_csv(file,Image.new("RGB",(100,100)))

    def test_six_faces(self):
        faces=extract_all_faces(Image.new("RGB",(256,128),"red"),64)
        self.assertEqual(set(faces),{"left","right","front","back","top","bottom"})
        self.assertTrue(all(image.size==(64,64) for image in faces.values()))

    def test_range_and_sync_fail_closed(self):
        self.assertEqual(foreground_range(np.full((20,20),2.),(0,0,20,20)),2)
        self.assertIsNone(foreground_range(np.full((20,20),np.nan),(0,0,20,20)))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            np.save(root/"range.npy",np.ones((20,20)))
            record=dict(frame="a.jpg",side="left",file="range.npy",image_time_s=1,pose_time_s=1,range_time_s=2)
            (root/"manifest.json").write_text(json.dumps(dict(range_convention="radial_metres",calibration_id="test",views=[record])))
            evidence=RangeEvidence(root/"manifest.json")
            with self.assertRaises(ValueError): evidence.load("a.jpg","left",(20,20))

    def test_tracking_one_to_one_and_no_synthetic_merge(self):
        p1,d1=self.point("a.jpg")
        p2,d2=self.point("b.jpg",x=.02)
        _,d3=self.point("b.jpg",x=.03)
        tracks,links=associate([d1,d2,d3],[p1,p2])
        self.assertEqual(len(tracks),2)
        self.assertEqual(d1.track_id,d2.track_id)
        self.assertNotEqual(d2.track_id,d3.track_id)
        p1.source=p2.source="synthetic_map_path"
        self.assertEqual(len(associate([d1,d2],[p1,p2])[0]),2)

    def test_evaluation_has_known_error_and_undefined_pair_score(self):
        pred=dict(frame="a",side="left",observation_index=0,track_id="T1",class_name="mature",x=1,y=0,z=0)
        truth=dict(pred,target_id="A",x=0)
        report=evaluate([pred],[truth])
        self.assertEqual(report["localization_rmse_m"],1)
        self.assertIsNone(report["association_pair_recall"])


if __name__ == "__main__": unittest.main()
