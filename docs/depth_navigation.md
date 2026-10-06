# D435i depth navigation and terrain gait selection

The module reconstructs rectified depth, estimates a constrained ground reference,
classifies local grid cells and produces normal, terrain, avoid or stop decisions.
Side boundaries are inserted into the local obstacle map. Traversable forward
surface transitions request terrain mode after persistent observations.

## Inputs and coordinates

- Raw depth: `16UC1` millimetres or `32FC1` metres; use rectified depth and its own
  `CameraInfo`, not the intrinsics of a different RGB image.
- Gravity-aligned robot coordinates: x forward, y left, z up. `base_stabilized`
  must have a dynamic TF connection to the optical frame at the image timestamp.
  Its origin follows the robot; pitch and roll come from the localization or
  fused attitude estimate. Raw D435i accelerometer/gyro streams alone are not
  an orientation quaternion.
- `geometry.ground_z_m`: ground height relative to this frame, from calibration.
- Robot footprint, collision height, slopes, down-step and braking limits live
  in `geometry`. `max_step_up_m=0.5` follows the author's approximate platform
  limit. Other values in the example JSON illustrate configuration; replace
  them with the mounted-camera and platform measurements.

The nominal height limit does not by itself authorize crossing: supporting
ground, local slope, roughness and collision-envelope checks also apply.
Missing depth stays unknown. The nearest-ground anchor avoids fitting a raised
bed or wall as the reference ground.

## Offline replay

Save raw depth frames as NumPy arrays and create a frame manifest:

```json
[
  {"depth_npy": "depth_000.npy", "timestamp_s": 1.0, "speed_mps": 0.2},
  {"depth_npy": "depth_001.npy", "timestamp_s": 1.1, "speed_mps": 0.2}
]
```

Each record may supply its own `optical_to_terrain` transform for attitude
compensation. Intrinsics, encoding and the geometry configuration are in
`config/depth_navigation.example.json`; frame paths are relative to the manifest.

```bash
python scripts/replay_depth_navigation.py --config calibration.json \
  --frames frames.json --output outputs/depth_navigation
```

Outputs are `decisions.json` and one obstacle-cell array per depth frame. Decisions
record timestamps, state, gait request, clearance, ground coverage and speed limit.

## ROS Noetic integration

Run from the repository root in a sourced ROS environment with NumPy, cv_bridge,
TF2 and the existing robot navigation stack. The RealSense ROS1 driver supplies
`/camera/depth/image_rect_raw` and `/camera/depth/camera_info`.

```bash
rosparam load config/depth_navigation.example.json /ptrdms_depth_navigation
python3 depth_ros.py _terrain_frame:=base_stabilized
```

Only `geometry` is read from the JSON by the ROS node. The other example keys are
used for offline replay; online calibration comes from `CameraInfo` and timestamped TF.
Remap move_base's velocity output to `/move_base/cmd_vel`, so that the depth node
receives the planner command and publishes the gated command on `/cmd_vel`.
The existing robot velocity driver consumes `/cmd_vel`.

| Interface | Meaning |
|---|---|
| `/ptrdms_depth_navigation/obstacles` | Blocking cells and elevated side boundaries, PointCloud2 |
| `/ptrdms_depth_navigation/clearing` | Valid depth returns for free-space ray tracing, PointCloud2 |
| `/ptrdms_depth_navigation/state` | JSON state, coverage, stopping distance and speed limit |
| `/ptrdms_depth_navigation/gait_request` | String: `normal` or `terrain` |
| `/robot/gait_state` | String from actual robot feedback, refreshed continuously |
| `/odom` | Timestamped measured velocity |

Add `{name: depth_obstacles, type: "costmap_2d::ObstacleLayer"}` to the existing
local costmap plugin list, alongside the LiDAR obstacle and inflation layers.
Merge `config/depth_costmap.example.yaml` under `local_costmap`. The two depth
sources share this layer; clearing runs before marking, so current obstacles
remain marked. Set `sensor_frame` to the optical frame to preserve the physical
ray origin even though points are expressed in the gravity-aligned frame.
Inflation and footprint configuration remain part of the existing local planner.

The forward camera gate permits forward curved trajectories within the planner's
footprint checks. Reverse travel, lateral travel and rotation in place require
the existing surrounding-obstacle sensing and a separate command policy; this
entry point returns zero for those commands. Stale depth, odometry, planner
commands or gait feedback also return zero. Separate command topics avoid loops.

## Unitree gait connection

The standalone `unitree_gait_ros.py` bridge calls `SportClient.SwitchGait` on a
firmware-matched SDK, and forwards the actual `gait_type` from
`rt/lf/sportmodestate` as the normal/terrain feedback. It sends changes only when
needed, retries rejected requests and does not treat request acceptance as
proof that the gait has changed. Supply the installed firmware's two IDs:

```bash
python3 unitree_gait_ros.py _network_interface:=eth0 \
  _normal_gait_id:=$NORMAL_GAIT_ID _terrain_gait_id:=$TERRAIN_GAIT_ID
```

Official Go2-W examples reference `SwitchGait`, while newer public Go2 SDK
versions remove that method. Therefore the bridge checks for the method at
startup; an existing firmware-specific robot driver can alternatively consume
`gait_request` and publish `gait_state`. Mode IDs are configured rather than
borrowing the gait enumeration of a different robot.

Sources: [RealSense ROS1 driver](https://github.com/realsenseai/realsense-ros/tree/ros1-legacy),
[ROS depth encoding](https://github.com/ros-infrastructure/rep/blob/master/rep-0118.rst),
[ROS obstacle layer](https://github.com/ros-planning/navigation/blob/noetic-devel/costmap_2d/plugins/obstacle_layer.cpp),
[Unitree Go2-W example](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2w/go2w_sport_client.cpp),
[Unitree interface change log](https://github.com/unitreerobotics/unitree_ros2/blob/master/CHANGELOG.md).
