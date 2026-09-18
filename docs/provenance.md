# Public-package provenance

PTRDMS publishes the source code, four-stage model weights, evaluation records, and interfaces required to inspect the processing workflow.

Raw greenhouse panoramas, site maps, and derived archived demonstrations are not redistributed in this public package. They remain under the data owner's control and may be requested from the authors subject to authorization. The reviewed crop images, annotations, fixed splits, and exclusion list are instead published separately as the [Four-Stage Tomato Ripeness Crop Dataset for PTRDMS](https://doi.org/10.5281/zenodo.22822984). The absence of the raw panorama and map materials does not change the stated claim boundary: panorama observations are not ground truth, and metric mapping requires independently validated geometry.

Model classes, corpus counts, weight hashes, and public-data exclusions are recorded in `metadata/source_audit.json`. The SHA-256 manifest in `metadata/dataset_manifest.json` covers the distributed model artifacts.
