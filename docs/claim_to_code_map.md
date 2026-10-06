# Feature-to-code map

| Feature | Code location | Usage |
|---|---|---|
| Panorama reprojection | `ripeness_demo/panorama.py:extract_all_faces()` and `extract_side_views()` | Export six faces with `--export-six-faces`; inference uses left/right views |
| Detection and four-stage classification | `ripeness_demo/detectors.py:TwoStageYoloDetector` | Use `--detector two-stage` with both model weights |
| Foreground range extraction | `ripeness_demo/evidence.py:RangeEvidence` | Supply registered `radial_metres` arrays with calibration and timestamps |
| Row-band filtering | Range-projection loop in `demo.py:run()` | Supply registered ranges; configure `row_offset_m` and `row_tolerance_m` |
| Cross-frame association | `ripeness_demo/evidence.py:associate()` | Supply CSV poses; select spatial cells or measured-range distance matching; configure frame-gap and cell size |
| Spatial projection | `ripeness_demo/mapping.py:project_detection()` | Pose plus row plane or registered radial range; recorded-coordinate view displays exported x/y/z |
| Processing metadata | `ripeness_demo/artifacts.py` | `evidence.json` records processing mode and input provenance |
| Batch processing | `demo.py:run()` | Input panorama folder; annotated images, CSV/JSON files and local report |
| Pose matching | `load_pose_csv()` and `match_poses_by_timestamp()` | Filename matching or nearest-timestamp soft synchronization with skipped-frame log |
| Display trajectory | `generate_demo_poses()` | Provides an ordered viewing path when poses are omitted |
| ROS adapter | `live_ros.py` and `ripeness_demo/online.py:TimedPoseBuffer` | Compressed panoramas and timestamped pose messages |
| ROS bag pose export | `scripts/export_rosbag_pose_evidence.py` | Export trigger timestamps and RTK path coordinates |

See [interfaces](interfaces.md) for schemas and coordinate conventions,
and the [README](../README.md) for the batch workflow. The reviewed crop dataset,
annotations and split manifests are available on
[Zenodo](https://doi.org/10.5281/zenodo.22827683).
