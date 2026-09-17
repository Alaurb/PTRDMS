# PTRDMS: Panoramic Tomato Ripeness Detection and Mapping System

This repository supports the manuscript *Ripeness Monitoring System for Open Facility Environments Based on a Quadruped Robot and Panoramic Vision*. It provides an **offline** workflow that reprojects equirectangular panoramas into perspective views, detects tomatoes on the left and right crop-row views, assigns a four-stage visual ripeness class, and writes annotated observations and a local result viewer.

The repository contains the perception and observation-mapping components only. Navigation, SLAM, gait control, collision avoidance, and real-time robot control are external to this package.

## Processing workflow

```text
Equirectangular panorama
        |
        v
Six perspective cube faces
        |
        +--> left and right faces --> one-class tomato detector --> four-stage classifier
                                                               |
                                                               v
Camera pose + optional registered range ----------> observation-level outputs
                                                               |
                                                               v
Annotated views + CSV/JSON + local result page
```

The panorama is geometrically reprojected into six faces; only the left and right views are selected for tomato inference. The software does not crop a panorama into six arbitrary image tiles.

## Maturity classes and model

The supplied two-stage model uses a one-class YOLO tomato detector followed by a YOLOv8n classification model. The classifier has exactly four output labels: `immature-period`, `green-maturity-period`, `discoloration-period`, and `maturity-period`. Their correspondence to the manuscript terminology is documented in [the class mapping](docs/four_stage_class_mapping.md).

The shipped classifier is trained from all 815 manually reviewed tomato crops after model selection. Its SHA-256 is `934d2f956117e02e45bdb1e03922c85a51820c007c990497b2df4b68a295563c`. The reviewed crop images and annotations are not redistributed here; they are available from the authors upon reasonable request. Model-selection results, split safeguards, and limitations are reported in [the four-stage evaluation](docs/four_stage_maturity_evaluation.md).

## Run the workflow

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

The user supplies the panorama folder and map image. A run produces `index.html`, projected views, annotated left/right views, `detections.csv`, `trajectory.csv`, `plants.csv`, `tracks.json`, `summary.json`, `run_config.json`, and `evidence.json`.

## Data and evidence boundaries

Raw panoramas, derived archived outputs, and site maps are not redistributed in this public package. They remain under the data owner's control and may be requested from the authors subject to authorization. They are not ground truth and must not be used to calculate detection accuracy, unique-fruit counting accuracy, spatial accuracy, or coverage.

The default example uses a synthetic path generated from frame order and an assumed row plane when measured pose/range inputs are absent. Consequently:

- an image observation is not automatically a unique tomato;
- left/right view selection alone does not prove exclusion of background-row fruit;
- pixel vertical position is retained, but metric fruit height requires calibrated geometry, measured pose, and registered range;
- range filtering and cross-frame association are only enabled when the required measured inputs are supplied and still require independent validation;
- the workflow is offline batch processing, not a real-time end-to-end deployment claim.

External pose and range interfaces are described in [docs/interfaces.md](docs/interfaces.md). The one-class detector evaluation is in [docs/detector_evaluation.md](docs/detector_evaluation.md), and source/data provenance is in [docs/provenance.md](docs/provenance.md).

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
