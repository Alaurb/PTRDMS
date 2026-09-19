# Paper evidence coverage

| Requirement | Repository support | Remaining evidence |
|---|---|---|
| Six-face panorama reprojection; bilateral inference | **Implemented:** `ripeness_demo/panorama.py:extract_all_faces()` and `extract_side_views()`, called by `demo.py:run()` | inspect faces and source correspondence |
| Tomato detection and ripeness classification | **Implemented:** `ripeness_demo/detectors.py:build_detector()` selects the supplied two-stage weights; `demo.py:run()` applies it to left/right faces | independent labels, checkpoint/figure provenance, held-out evaluation |
| Batch and online sequence processing | **Implemented:** `demo.py:run()` provides offline batch processing; `live_ros.py` consumes ROS1 panoramas and timestamped poses, and `ripeness_demo/online.py:TimedPoseBuffer` performs online pose association | the manuscript evaluation remains the offline batch workflow; online deployment uses the configured ROS topics |
| Pixel vertical information | **Implemented:** `ripeness_demo/mapping.py:SpatialDetection` and `ripeness_demo/report.py:write_csv_files()` retain bbox y coordinates and view dimensions | metric height needs verified camera pose/range |
| Spatial display | **Implemented:** `ripeness_demo/mapping.py:project_detection()` and `ripeness_demo/report.py:write_html_report()` support assumed-row-plane or registered-range coordinates | sample coordinates are synthetic/estimated unless measured inputs and independent evaluation are supplied |
| Nearest row and duplicate observations | **Implemented conditionally:** `demo.py:run()` applies the row-band gate only with registered ranges; `ripeness_demo/evidence.py:associate()` makes conservative within-pass associations only with CSV poses and measured radial ranges | independently verified row/fruit IDs and spatial truth; outputs are candidates, not validated unique-fruit counts |
| External localization data | **Implemented:** `ripeness_demo/mapping.py:load_pose_csv()` and `ripeness_demo/evidence.py:RangeEvidence`; ROS schemas are retained in `interfaces/ros_msgs/` | upstream synchronization/extrinsics, full 6-DoF extension |
| Navigation and gait performance | **Intentionally excluded:** no navigation or gait-control entry point is shipped; see `README.md` and `docs/interfaces.md` | requires external field evidence |

The repository satisfies the image-processing and observation-visualization scope, not every empirical claim of the original preprint. The default example cannot establish unique-fruit counts, physical height accuracy, persistent plant mapping, or independent recognition accuracy. Historical metrics are retained as records, not freshly reproduced results.
