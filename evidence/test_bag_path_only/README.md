# Path-only ROS bag timing fixture

This public fixture was exported from the authorized `test.bag` acquisition
record. It contains no panorama image files and the exporter did not read an
image topic. Opaque `shot_NNNN.jpg` strings are timing-test identifiers only;
they do not name or include a source panorama.

`trigger_pose_trace.csv` records each camera-trigger bag timestamp, its unique
nearest RTK bag timestamp, their absolute time difference, local ENU path
coordinates, and a velocity-derived yaw estimate. `ptrdms_frame_times.csv` and
`ptrdms_pose_samples.csv` are directly compatible with PTRDMS timestamp-mode
interface testing. `rejected_triggers.csv` is intentionally header-only: all
1,121 triggers matched within the recorded 60 ms upper bound (observed maximum
56.991 ms; see `metadata.json`).

`path_preview.png` visualizes the exported trigger/RTK route. Its line colour
progresses from early to late acquisition, and hollow markers occur every 50th
trigger. It is generated without image data by `scripts/render_path_evidence.py`.

The common clock is **ROS bag record time**, not the RTK message-header time;
the latter uses a distinct device clock in this acquisition. The coordinates
are an RTK-derived local ENU approximation and yaw is inferred from horizontal
velocity. This is timing/path provenance evidence, not an accuracy validation
of fruit positions, camera extrinsics, row gating, or unique-fruit counting.
