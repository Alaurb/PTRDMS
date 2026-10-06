# External data interfaces

PTRDMS binds panoramas to map poses using image-matched localization records. The batch workflow runs without ROS.

## Camera poses

By default, pass `--pose-csv poses.csv` with `frame,x,y,z,yaw,route`; `frame` is the image filename and the loader binds it by full filename with a filename-stem fallback. For timestamp binding, pass `--pose-match-mode timestamp --pose-csv poses.csv --frame-times-csv frame_times.csv`: pose rows require `x,y,z,yaw,timestamp_s[,route]` (a frame field is optional), frame-time rows require `frame,timestamp_s`, and aliases `timestamp`, `time_s`, and `time` are accepted. Each panorama receives one unique nearest pose within `--pose-timestamp-tolerance-s` (default 0.05 s); frames missing a timestamp, with an ambiguous nearest pose or outside the tolerance are skipped. When several images share a nearest pose, the smallest-residual pair is retained (ties use acquisition time and filename). Accepted pairs are processed in acquisition-time order, and `synchronization.json` records all matching decisions. This is nearest-neighbour association, not interpolation. `trajectory.csv` records each timestamp-bound frame's frame time, pose time, and absolute match delta. Coordinates are metres in a shared Cartesian map frame, yaw is radians counterclockwise about +z, +x is camera-forward at zero yaw and +y left. z is the camera optical-centre height, not robot base height. roll and pitch are assumed zero. `route` is an optional session/row identifier. Upstream processing remains responsible for camera extrinsics and a common time base. Map image coordinates use a bottom-left metric origin, with image v pointing downward; supply a compatible map image and `--map-width-m` scale.

## Panorama yaw extrinsic

Pass `--camera-yaw-offset-deg` to align the native equirectangular forward axis with the robot-forward direction used by each pose. The offset is subtracted during panorama sampling, so the selected logical `left` and `right` faces stay robot-relative. A `-90` degree setting, for example, selects the native panorama's front/back views as robot-relative left/right views. This is a camera-mount calibration value, not a detector setting; it must be recorded with the run and applied consistently when registered range maps are generated.

When poses are omitted, the viewer generates an ordered display trajectory tagged `synthetic_map_path`.

## Registered radial ranges

Pass `--range-manifest ranges.json` together with poses. Example manifest:

```json
{"calibration_id":"YOUR_VERIFIED_CALIBRATION","range_convention":"radial_metres","views":[{"frame":"frame.jpg","side":"left","file":"left.npy","image_time_s":1.0,"range_time_s":1.0,"pose_time_s":1.0},{"frame":"frame.jpg","side":"right","file":"right.npy","image_time_s":1.0,"range_time_s":1.0,"pose_time_s":1.0}]}
```

NPY paths are relative to the manifest. Each array must match its projected view height and width and contain radial distance along the pixel ray, not unconverted camera z-depth. Arrays must already be calibrated and registered to the cube-face convention. Metadata timestamps are checked against `--sync-tolerance-s`; each entry links a projected view to its synchronized range array.

The central bounding-box region supplies the foreground range statistic. Row
filtering applies the configured lateral band when registered ranges and CSV
poses are supplied. Cross-frame association uses route, side and frame gap. The default
`--association-method spatial-cell` groups observations in map-aligned 3D cells
(default side length 0.12 m, maximum gap three frames), using projected locations
and supplied CSV poses. Matching is one-to-one per frame.
`--association-method measured-distance` uses measured radial ranges and a 3D
distance threshold. Row filtering uses measured ranges; its status and the
association mode are recorded in `summary.json`.

## Outputs and ROS compatibility

`detections.csv` retains pixel bounding boxes, view dimensions, predicted class/confidence, x/y/z, position_source and track_id. Pixel vertical centre can be calculated as `(bbox_y1+bbox_y2)/2`. `trajectory.csv` preserves pose provenance. `summary.json` states pose/range assumptions and association status. `tracks.json` and `association_links.json` describe candidate associations. `observations.csv` groups detections by frame and side, with `observation_id` (`frame:side`), `detection_count`, class counts, and `assignment_source=frame_side_proxy`. Its x/y values locate the observation anchor. `plants.csv` exports the same groups in the legacy format. `summary.json` records `count_unit=detection_observation` and `grouping_unit=frame_side`.

`live_ros.py` provides the optional online ROS1 entry point. It subscribes to
`sensor_msgs/CompressedImage` panoramas and `geometry_msgs/PoseStamped` poses,
stores poses by exact `header.stamp`, and accepts one unique nearest pose within
the configured tolerance for each incoming panorama. The rolling result folder
uses the same annotated-image, CSV, JSON, and map-report formats as batch mode.
Each accepted frame is additionally published as a JSON `std_msgs/String`
message on `/ptrdms/observations` by default; the topic is configurable with
`--result-topic`.

The MaturityDetection and MaturityObservationArray schemas in
`interfaces/ros_msgs/` remain available as adapter references. An integration
adapter should preserve acquisition timestamps and coordinate provenance when it
publishes these messages.
