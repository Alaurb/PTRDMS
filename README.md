# PTRDMS: Panoramic Tomato Ripeness Detection and Mapping System

PTRDMS is the reproducibility package for the manuscript *Ripeness Monitoring
System for Open Facility Environments Based on a Quadruped Robot and Panoramic
Vision*. It implements the panoramic perception and observation-mapping layers
described there: a panorama is reprojected into perspective views, tomatoes are
detected in robot-relative crop-row views, one of four visual ripeness stages is
assigned, and image-linked mapping results are exported for review.

The processing workflow accepts acquired panoramas and image-matched localization
records from the robot or another acquisition system.

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
Detection and crop classification use separate evaluation protocols, summarized
in the recognition-results table below.

## Features

- Six-face panorama reprojection and left/right crop-row inference.
- Tomato detection and four-stage ripeness classification.
- Image-matched pose import and spatial observation mapping.
- Range-aware row filtering and cross-frame observation association.
- Annotated images, CSV/JSON exports, and an interactive local report.

See [implementation details](docs/claim_to_code_map.md) for code locations and
input requirements. The released classifier was fitted on all 815 reviewed
crops for 11 epochs after model selection.

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
annotated detections, `detections.csv`, `observations.csv`, `trajectory.csv`,
spatial-map images, and JSON records for downstream analysis.

`observations.csv` groups detections by source frame and crop-row side. Its
`observation_id` identifies a frame-side observation, and `detection_count` is
the number of retained detections in that frame-side group. `plants.csv`
provides the same groups in the legacy export format. Cross-frame associations
are exported separately in `tracks.json`.

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
for range-aware row filtering and within-pass observation association. These
operations require calibrated, registered radial ranges and measured camera
poses. Without range data, nearest-row filtering and cross-frame association
are disabled; mapping uses the configured row-plane projection.

`summary.json` records the processing mode, count/grouping units, input sources,
row-filter status, and association status. `evidence.json` records processing
mode and input provenance. `tracks.json`, `association_links.json`, and
`rejected_observations.json` provide traceable records for mapping operations.
If a pose table is not provided, the viewer renders an ordered demonstration trajectory so that image
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

`verify_dataset.py` checks the SHA-256 manifest for the distributed model artifacts without loading model weights. The tests validate projection, inference interfaces, input provenance, and output handling.

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

## Reported recognition results

The table summarizes detection and four-stage crop-classification results.

| Component | Evaluation | Reported results |
|---|---|---|
| One-class tomato detector | Fixed validation, 36 projected views; used for checkpoint and candidate selection | Precision 63.1%; recall 70.9%; mAP@0.5 70.1%; mAP@0.5:0.95 32.4% |
| Four-stage crop classifier | Five repeated source-image-group-disjoint holdouts; 121 test crops per repetition | Accuracy 62.5% ± 2.2 percentage points; macro-F1 64.0% ± 1.9 percentage points |

Classifier values are mean ± sample standard deviation over five repeated
holdouts. See [detector evaluation](docs/detector_evaluation.md) and
[four-stage classifier evaluation](docs/four_stage_maturity_evaluation.md) for
protocols, split manifests, and model provenance.

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
