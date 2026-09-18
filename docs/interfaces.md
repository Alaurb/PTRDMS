# External data interfaces

No navigation or ROS installation is required. Localization and robot control remain external.

## Camera poses

Pass `--pose-csv poses.csv`. Header: `frame,x,y,z,yaw,route`. One unique row per input image; frame is the image filename. Coordinates are metres in a shared Cartesian map frame, yaw is radians counterclockwise about +z, +x is camera-forward at zero yaw and +y left. z is the camera optical-centre height, not robot base height. roll and pitch are assumed zero. An upstream adapter must apply camera extrinsics and match each pose to image acquisition time. This loader does not interpolate raw odometry timestamps. `route` is an optional session/row identifier. Map image coordinates use a bottom-left metric origin, with image v pointing downward; supply a compatible map image and `--map-width-m` scale.

Missing poses generate a clearly tagged `synthetic_map_path` for display only. Supplying a CSV does not prove its measurements are accurate.

## Registered radial ranges

Pass `--range-manifest ranges.json` together with poses. Example manifest:

```json
{"calibration_id":"YOUR_VERIFIED_CALIBRATION","range_convention":"radial_metres","views":[{"frame":"frame.jpg","side":"left","file":"left.npy","image_time_s":1.0,"range_time_s":1.0,"pose_time_s":1.0},{"frame":"frame.jpg","side":"right","file":"right.npy","image_time_s":1.0,"range_time_s":1.0,"pose_time_s":1.0}]}
```

NPY paths are relative to the manifest. Each array must match its projected view height and width and contain radial distance along the pixel ray, not unconverted camera z-depth. Arrays must already be calibrated and registered to the cube-face convention. Metadata timestamps are checked against `--sync-tolerance-s`; this checks supplied metadata consistency, not calibration accuracy or image/pose provenance. This example is a schema, not supplied field data.

The central bounding-box region supplies a range statistic. Leaf occlusion can still corrupt it. The lateral nearest-row band is a geometric gate, not a semantic guarantee, and it is applied only when `--range-manifest` is supplied (the guarded branch is `demo.py:run()`, `if ranges is not None`, around the projection loop). It requires calibrated, registered radial ranges together with `--pose-csv`; without range input the gate is not applied, `summary.json` records `nearest_row_filter="not_verified_no_depth"`, and association status is `disabled_no_measured_geometry`. Candidate associations additionally require every processed pose to be sourced from the CSV and every retained detection to use a measured radial range; they remain unverified without ground truth.

## Outputs and ROS compatibility

`detections.csv` retains pixel bounding boxes, view dimensions, predicted class/confidence, x/y/z, position_source and track_id. Pixel vertical centre can be calculated as `(bbox_y1+bbox_y2)/2`. `trajectory.csv` preserves pose provenance. `summary.json` states pose/range assumptions and association status. `tracks.json` and `association_links.json` describe candidate associations. `plants.csv` groups observations by frame and side; it does not count unique plants.

The previous MaturityDetection and MaturityObservationArray schemas are preserved in `interfaces/ros_msgs/` as adapter references. The former ROS CSV publisher and TF-origin placeholder fusion are retired. An adapter must explicitly map CSV xyxy boxes to ROS centre/size fields, preserve acquisition timestamps, and set coordinate provenance. No navigation node or live ROS publisher is included.
