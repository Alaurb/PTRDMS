# Claim-to-code map

This map identifies the public implementation that supports each repository claim and
the conditions under which it operates. It is not a performance-validation table.
Where measured geometry or independent ground truth is absent, the corresponding
outputs remain observations or candidates rather than validated field quantities.

| Capability | Code location | Enablement condition | Validation status |
|---|---|---|---|
| Six perspective reprojections and left/right inference selection | `ripeness_demo/panorama.py:extract_all_faces()`; `extract_side_views()`; `demo.py:run()` | Every input panorama is geometrically reprojected. All six faces are exported only with `--export-six-faces`; inference consumes left/right faces. | Projection and interface tests are public; no claim that faces are camera-supplied images. |
| One-class tomato detection and four-stage classification | `ripeness_demo/detectors.py:TwoStageYoloDetector`; `build_detector()`; `demo.py:run()` | Invoke `demo.py --detector two-stage` with both released weights. | Detector and crop-classifier evaluations are separate. Four-stage scores are crop-level, not end-to-end panorama-system accuracy. |
| Registered radial-range foreground statistic | `ripeness_demo/evidence.py:RangeEvidence.load()`; `foreground_range()` | `--range-manifest` plus matched `--pose-csv`; `radial_metres`, a calibration ID, image-size agreement, and timestamp tolerance are required. | The central half-box median rejects inadequate/invalid samples; leaf occlusion and calibration quality require independent evaluation. |
| Nearest-row band gate | `demo.py:run()` (range-projection loop, `if ranges is not None`) | Only with registered radial ranges. Detections outside `row_offset_m ± row_tolerance_m` are written to `rejected_observations.json` with `outside_nearest_row_band`. | A geometric filter only; no semantic guarantee of nearest-row identity. Without range data it is disabled and reported as `not_verified_no_depth`. |
| Conservative within-pass cross-frame association | `ripeness_demo/evidence.py:associate()` | Every pose must have `source == "pose_csv"`, and each associated detection must have `position_source == "measured_radial_range"`; route/side, maximum frame gap, and 3D-distance thresholds also apply. | Track IDs are heuristic association candidates, local to one pass; manually assigned fruit IDs are required for validation. |
| Per-fruit estimated `(x, y, z)` projection | `ripeness_demo/mapping.py:project_detection()`; `ripeness_demo/report.py:fruitPosition()` | Coordinates use a pose plus either registered radial range or the assumed row plane. The local report’s **View: recorded coordinates** mode uses `cube.x/y/z` directly. | Assumed-row-plane output is illustrative only. Registered-range output is still a candidate until independently evaluated. |
| Evidence-boundary artifact and visible report banner | `ripeness_demo/artifacts.py:build_evidence_manifest()`; `write_evidence_manifest()`; `ripeness_demo/report.py:HTML_TEMPLATE` | Every `demo.py` run writes `evidence.json`; the report displays its headline. | Status distinguishes `unvalidated_observation_layout`, `measured_geometry_candidate`, and archived replay; no status establishes unique-fruit or spatial accuracy. |
| Offline batch processing | `demo.py:run()` (`processing_mode="offline_batch"`); `ripeness_demo/report.py:write_html_report()` | Run `demo.py` over a user-supplied panorama folder and map. | Online acquisition and real-time latency are not implemented or claimed. |
| Pose provenance and synthetic fallback | `ripeness_demo/mapping.py:load_pose_csv()`; `generate_demo_poses()`; `ripeness_demo/report.py:write_summary()` | `--pose-csv` uses supplied camera-centre poses; otherwise a filename-order display path is generated. | Synthetic paths are labelled `synthetic_map_path` and cannot support counting, coverage, or spatial-accuracy claims. |

The runnable workflow and its input/output contract are documented in `README.md` and
`docs/interfaces.md`. The four-stage crop corpus, annotations, five fixed splits, and
exclusion list are available separately at [Zenodo concept DOI 10.5281/zenodo.22822983](https://doi.org/10.5281/zenodo.22822983).
