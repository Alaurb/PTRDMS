# Paper evidence coverage

| Requirement | Repository support | Remaining evidence |
|---|---|---|
| Six-face panorama reprojection; bilateral inference | projection code, raw images, six-face export, tests | inspect faces and source correspondence |
| Tomato detection and ripeness classification | two-stage inference, supplied weights and archived metrics | independent labels, checkpoint/figure provenance, held-out evaluation |
| Continuous sequence analysis | offline batch processing and saved replay | online acquisition and real-time latency are not claimed |
| Pixel vertical information | bbox_y1/y2 and view dimensions retained | metric height needs verified camera pose/range |
| Spatial display | offline 3D viewer, assumed row-plane or registered-range projection | sample coordinates are synthetic/estimated |
| Nearest row and duplicate observations | range gate and conservative association candidates | independently verified row/fruit IDs and spatial truth |
| External localization data | pose CSV, radial-range manifest, preserved ROS message schemas | upstream synchronization/extrinsics, full 6-DoF extension |
| Navigation and gait performance | intentionally excluded | requires external field evidence |

The repository satisfies the image-processing and observation-visualization scope, not every empirical claim of the original preprint. The default example cannot establish unique-fruit counts, physical height accuracy, persistent plant mapping, or independent recognition accuracy. Historical metrics are retained as records, not freshly reproduced results.
