# PTRDMS: Panoramic Tomato Ripeness Detection and Mapping System

PTRDMS supports offline batch and online ROS1 workflows for tomato-ripeness
monitoring from equirectangular panoramic images. It reprojects a panorama into
perspective views, detects tomatoes from robot-relative crop-row views, assigns
one of four visual ripeness stages, and exports image-linked mapping results for
review.

The repository focuses on panoramic perception and observation mapping. It
accepts image-matched localization records produced by an upstream platform
when location-aware mapping is required.

## Processing workflow

```text
Panoramic image → geometric perspective reprojection → tomato detection
                → four-stage ripeness classification → mapped observations
                → annotated images, tables, and local visual report
```

Each panorama is geometrically reprojected into six perspective faces. The
standard inference workflow uses the robot-relative left and right crop-row
views; all six faces may also be exported for inspection.

## Maturity classes and model

PTRDMS uses a two-stage model: a one-class YOLO tomato detector followed by a
YOLOv8n classifier with four classes:

- `immature-period`
- `green-maturity-period`
- `discoloration-period`
- `maturity-period`

Class terminology is documented in [docs/four_stage_class_mapping.md](docs/four_stage_class_mapping.md).
The reviewed crop dataset, annotations, reproducible splits, and exclusion list
are available at [Zenodo: 10.5281/zenodo.22822983](https://doi.org/10.5281/zenodo.22822983).
The supplied classifier SHA-256 is
`934d2f956117e02e45bdb1e03922c85a51820c007c990497b2df4b68a295563c`.

## Offline batch workflow

Create the tested environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r optional-requirements.txt
```

For the desktop window, run `python app.py`. For batch operation:

```powershell
python demo.py `
  --input <panorama-folder> `
  --map <map-image> `
  --output outputs/inference `
  --detector two-stage `
  --detector-weights models/tomato_detector.pt `
  --classifier-weights models/tomato_ripeness_classifier.pt `
  --confidence 0.25 `
  --export-six-faces `
  --max-frames 0
```

The run creates a local visual report (`index.html`), perspective views,
annotated detections, `detections.csv`, `trajectory.csv`, spatial-map images,
and JSON records for downstream analysis.

## Online ROS1 operation

PTRDMS can also process a live ROS1 stream. The online entry point subscribes
to an equirectangular JPEG topic (`sensor_msgs/CompressedImage`) and a
localization topic (`geometry_msgs/PoseStamped`), matches each panorama to its
nearest timestamped pose, and continuously refreshes annotated views, tables,
and the local map report. Raw panoramic frames are processed in memory.

In a ROS1 environment, source the robot workspace and run:

```bash
python live_ros.py \
  --map <map-image> \
  --image-topic /camera/image/compressed \
  --pose-topic /robot_pose \
  --output outputs/online \
  --detector two-stage \
  --detector-weights models/tomato_detector.pt \
  --classifier-weights models/tomato_ripeness_classifier.pt
```

Use `--pose-tolerance-s` to configure the accepted image-to-pose timestamp
window and `--report-every` to set the report-refresh interval.

## Localization-aware mapping

For location-aware mapping, provide an image/pose table from the upstream
localization workflow:

```powershell
python demo.py `
  --input <panorama-folder> `
  --map <map-image> `
  --pose-csv <image_matched_poses.csv> `
  --range-manifest <registered_ranges.json> `
  --output outputs/mapping
```

The pose interface supports filename-based matching and explicit timestamp
matching. Use `--pose-match-mode timestamp --frame-times-csv <frame_times.csv>`
when the acquisition pipeline provides image and pose timestamps. A registered
radial-range manifest can additionally be supplied through `--range-manifest`
for range-aware row filtering and within-pass observation association.

`tracks.json`, `association_links.json`, and `rejected_observations.json`
provide traceable records for these mapping operations. If a pose table is not
provided, the viewer renders an ordered demonstration trajectory so that image
observations can still be inspected.

See [docs/interfaces.md](docs/interfaces.md) for CSV and range-manifest
schemas, coordinate conventions, and integration notes.

## Data availability

The public repository contains code, model artifacts, interfaces, tests, and
the released crop dataset reference. Field panoramas and site maps are managed
separately by their data owner and are available from the authors subject to
authorization.

## Verification

```powershell
python scripts/verify_dataset.py
python -m unittest discover -s tests -v
```

`verify_dataset.py` checks the SHA-256 manifest for the distributed model artifacts without loading model weights. The tests validate projection, inference interfaces, evidence boundaries, and output handling.

## Repository layout

```text
app.py                     Desktop operation window
demo.py                    Batch processing entry point
models/                    Tomato detector and four-stage classifier
ripeness_demo/             Projection, inference, mapping, evidence checks, reports
scripts/                   Data preparation, evaluation, and verification helpers
docs/                      Interfaces, provenance, class mapping, and evaluations
tests/                     Processing and desktop-command tests
```

## License

Project code is licensed under [LICENSE](LICENSE). Data assets and imported code retain the terms recorded in [LICENSE_DATA_AND_IMPORTED_CODE](LICENSE_DATA_AND_IMPORTED_CODE). Third-party dependencies retain their own licenses.
