# PTRDMS: Panoramic Tomato Ripeness Detection and Mapping System

PTRDMS is the reproducibility package for the manuscript *Ripeness Monitoring
System for Open Facility Environments Based on a Quadruped Robot and Panoramic
Vision*. It implements the panoramic perception and observation-mapping layers
described there: a panorama is reprojected into perspective views, tomatoes are
detected in robot-relative crop-row views, one of four visual ripeness stages is
assigned, and image-linked mapping results are exported for review.

The repository covers perception and observation mapping only. Navigation, SLAM,
gait control, and real-time robot control are outside its scope.

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
are available as [Zenodo Version v2: 10.5281/zenodo.22827683](https://doi.org/10.5281/zenodo.22827683).
The supplied classifier SHA-256 is
`934d2f956117e02e45bdb1e03922c85a51820c007c990497b2df4b68a295563c`.
Detector and classifier results are evaluated separately; the four-stage
classification results are crop-level measurements, not end-to-end system
accuracy for panoramas.

## Scope and claim boundaries

The repository implements and documents a processing pipeline. It does not
establish end-to-end system accuracy, and the boundaries below apply to every
output it produces.

- **The repository does not navigate the robot.** It consumes image-matched
  localization records produced upstream. Navigation, gait control, and
  localization are external to this package.
- **Reported recognition scores are crop-level, not panorama-system scores.** The
  four-stage classifier is evaluated on reviewed tomato crops. Those numbers are
  not end-to-end accuracy for the full panorama pipeline.
- **The detector and the maturity classifier are evaluated separately.** Neither
  evaluation measures whole-system accuracy.
- **The deployed weights are full-corpus fits and have no held-out evaluation
  split.** After model selection, the released classifier was trained on all 815
  reviewed crops for 11 fixed epochs. Its training-fit metrics are not reported as
  performance.
- **Fruit positions are observations, not validated ground truth.** Coordinates
  come from a camera pose combined with a projected fruit position. Where
  registered radial ranges are unavailable, the assumed-row-plane output is
  illustrative only. No output establishes unique-fruit counts, physical fruit
  height, or spatial accuracy.
- **The manuscript reports offline batch results.** An online ROS1 entry point is
  provided for completeness, but the reported figures come from the offline batch
  workflow.

A per-capability view of what is implemented, under what condition, and with what
validation status is maintained in [docs/claim_to_code_map.md](docs/claim_to_code_map.md).

## Citation

If you use this code or the released model weights, please cite the manuscript.
If you use the reviewed crop dataset, please cite the dataset record as well.

```bibtex
@misc{lyu2026dataset,
  author    = {Lyu, Jinru},
  title     = {Four-Stage Tomato Ripeness Crop Dataset for PTRDMS},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22827683},
  note      = {Version v2}
}
```

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

> **Scope note.** The online entry point is provided so the same pipeline can be
> driven by a live stream. **It has not been validated in the field.** The
> manuscript's reported results come from the offline batch workflow, and no
> online latency, throughput, or real-time on-board-inference claim is made here
> or in the manuscript.

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
  --result-topic /ptrdms/observations \
  --output outputs/online \
  --detector two-stage \
  --detector-weights models/tomato_detector.pt \
  --classifier-weights models/tomato_ripeness_classifier.pt
```

Use `--pose-tolerance-s` to configure the accepted image-to-pose timestamp
window and `--report-every` to set the report-refresh interval. Each accepted
frame is also published as a JSON `std_msgs/String` message on `--result-topic`.

### Public online-processing showcase

The repository includes an image-free timing and path fixture exported from a
ROS bag. It demonstrates the upstream trigger/pose inputs that PTRDMS accepts
for timestamped observation binding; no panorama bytes, tomato detections, or
site map are included in this public artifact.

![Image-free trigger and RTK path preview](evidence/test_bag_path_only/path_preview.png)

Run the renderer again with:

```bash
python scripts/render_path_evidence.py \
  --input evidence/test_bag_path_only/trigger_pose_trace.csv \
  --output evidence/test_bag_path_only/path_preview.png
```

See [evidence/test_bag_path_only/README.md](evidence/test_bag_path_only/README.md)
for its inputs and scope. The online node emits a compact JSON observation
message for each accepted frame; [docs/online_showcase.md](docs/online_showcase.md)
documents the public-facing flow and message fields without exposing field imagery.

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

**Released with this repository.** Source code, the two deployed model weights,
interface schemas, provenance records, and the evaluation records under `docs/`.

**Published separately.** The human-reviewed four-stage tomato crop images, their
annotations, the five fixed source-group-disjoint splits, and the exclusion list
for unclassifiable records are published as the *Four-Stage Tomato Ripeness Crop
Dataset for PTRDMS*: [Zenodo, version DOI 10.5281/zenodo.22827683](https://doi.org/10.5281/zenodo.22827683)
(CC BY 4.0). The crop corpus is not redistributed inside this repository.

**Not redistributed.** Raw panoramas, facility maps, and derived archived
demonstrations originate from the facility operator. They are managed separately
by their data owner and may be requested from the authors subject to
authorization, including for peer-review purposes.

Running the released workflow requires a user-supplied panorama directory and
facility map.

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

## Reproducibility of the reported results

The classifier figures in the manuscript come from five repeated,
source-group-disjoint 70/15/15 holdouts generated with seeds 20260915–20260919,
each containing 572 training, 122 validation, and 121 test crops. The five
repetitions are not five-fold cross-validation: their test sets may overlap, and
no single repetition was selected for reporting.

The seeds are date-formatted integers used as framework RNG seeds; they are not
acquisition dates. The corpus was collected between October and December 2025,
and the held-out sets are drawn from the reviewed corpus rather than partitioned
by collection date.

The per-repetition split manifests and test reports are published with the
dataset and summarized in
[docs/four_stage_maturity_evaluation.md](docs/four_stage_maturity_evaluation.md),
which also lists the ResNet-18 baseline and the two predeclared alternatives
retained as negative results.

## License

Project code is licensed under [LICENSE](LICENSE). Data assets and imported code retain the terms recorded in [LICENSE_DATA_AND_IMPORTED_CODE](LICENSE_DATA_AND_IMPORTED_CODE). Third-party dependencies retain their own licenses.
