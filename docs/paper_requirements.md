# Manuscript feature coverage

| Manuscript feature | Implementation | Inputs and outputs |
|---|---|---|
| Panorama reprojection | `ripeness_demo/panorama.py` | Six perspective faces; left/right inference views |
| Tomato detection and ripeness classification | `ripeness_demo/detectors.py` | Two-stage model weights; boxes, classes, confidence |
| Sequence processing | `demo.py:run()` | Panorama folder and image-matched localization records |
| Detection statistics | `ripeness_demo/report.py:write_csv_files()` | Frame-side groups in `observations.csv`; legacy `plants.csv` |
| Vertical image information | `SpatialDetection` and CSV exports | Bounding-box coordinates and view dimensions |
| Spatial display | `project_detection()` and `write_html_report()` | Pose-based row-plane or registered-range projection |
| Row filtering | Range-projection branch in `demo.py` | Registered ranges and configured lateral row band |
| Cross-frame association | `ripeness_demo/evidence.py:associate()` | CSV poses, spatial cells or measured-distance matching, route/side and frame gap |
| Localization import | `load_pose_csv()` | Filename matching or timestamp soft synchronization; trajectory and matching-log exports |

Evaluation protocols and recognition results are documented in
[detector evaluation](detector_evaluation.md) and
[four-stage classification evaluation](four_stage_maturity_evaluation.md).
