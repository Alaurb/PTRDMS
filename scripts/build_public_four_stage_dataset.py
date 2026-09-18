"""Build the frozen public v1.0 four-stage crop dataset from reviewed links.

The input tree remains untouched.  Only the formal ``review_v1`` snapshot and
the five paper-reported group-disjoint split manifests are read.  Crop pixels
are materialized into a new directory, so the release contains no symbolic
links or absolute source paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


SEEDS = tuple(range(20260915, 20260920))
CLASSES = (
    "immature-period",
    "green-maturity-period",
    "discoloration-period",
    "maturity-period",
)
MODEL_INDEX = {
    "discoloration-period": 0,
    "green-maturity-period": 1,
    "immature-period": 2,
    "maturity-period": 3,
}
CC_BY_4_LEGALCODE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_group_from_target(path: Path) -> str:
    """Match the formal split generator's source-image group rule."""
    name = re.sub(r"^batch\d+__", "", path.resolve().stem)
    return re.sub(r"(?:_tomato_|__)\d+$", "", name)


def records_for_manifest(path: Path) -> list[dict[str, str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if "records" in document:
        return [
            {
                "name": Path(record["source_link"]).name,
                "label": record["label"],
                "source_group": record["source_group"],
                "split": record["split"],
            }
            for record in document["records"]
        ]
    records: list[dict[str, str]] = []
    for split, block in document["splits"].items():
        records.extend(
            {"name": record["name"], "label": record["class"], "source_group": record["source_group"], "split": split}
            for record in block["files"]
        )
    return records


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    """Write stable LF text on Python versions used by the data workstation."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def copy_without_metadata(source: Path, target: Path) -> None:
    """Re-encode PNG pixels without carrying EXIF or other source metadata."""
    with Image.open(source) as image:
        image.load()
        image.convert("RGBA" if image.mode == "RGBA" else "RGB").save(target, format="PNG")
    with Image.open(target) as image:
        if image.getexif():
            raise ValueError(f"EXIF remained after materialization: {target.name}")


def release_readme() -> str:
    return """# Tomato Ripeness Four-Stage Crops v1.0

This dataset accompanies *Ripeness Monitoring System for Open Facility Environments Based on a Quadruped Robot and Panoramic Vision*. It contains 815 manually reviewed, fruit-level tomato crop images for four visual maturity stages. It is not an end-to-end panorama benchmark and cannot support claims about tomato detection accuracy, unique-fruit counting, spatial accuracy, internal maturity, firmness, soluble solids, or harvest readiness.

## Contents

`crops/` contains 815 materialized PNG crop images. `annotations.csv` gives the reviewed class and an anonymized source-group identifier for each crop. `splits/` contains the five source-group-disjoint repeated 70/15/15 holdouts reported in the associated study. `excluded_unclassifiable.csv` records the 46 reviewed crops excluded before every split; they are not a model class and their image pixels are not distributed. `MANIFEST.csv` records SHA-256 and byte size for every package file except itself.

The classes are defined in `class_mapping.md`. Crop identifiers and source-group identifiers are release-local, stable identifiers; they do not disclose source filenames, capture dates, locations, or farm names.

## License and source-material boundary

This dataset is released under CC BY 4.0. Cite the associated DOI: **[10.5281/zenodo.22822984](https://doi.org/10.5281/zenodo.22822984)**. The original panoramas, site maps, derived archived demonstrations, model weights, and other raw acquisition data are not included.

## Reproducibility boundary

Each split CSV has exactly 815 rows. Within a seed, every crop from one anonymized source group is assigned to exactly one of `train`, `val`, or `test`. The reported classification results are crop-level results from repeated source-group-disjoint holdouts; they are not a real-time or full-system result.
"""


def class_mapping() -> str:
    return """# Four-stage class mapping

| Dataset label | Model output index | Manuscript term | Chinese term |
|---|---:|---|---|
| `immature-period` | 2 | immature period | 未熟期 |
| `green-maturity-period` | 1 | green-maturity period | 绿熟期 |
| `discoloration-period` | 0 | discoloration period | 转色期 |
| `maturity-period` | 3 | maturity period | 成熟期 |

`unclassifiable` is a reviewed exclusion disposition, not a deployed model class.
"""


def license_text() -> str:
    """Fetch the complete official CC BY 4.0 legal code for the release."""
    request = Request(CC_BY_4_LEGALCODE_URL, headers={"User-Agent": "PTRDMS-dataset-builder/1.0"})
    with urlopen(request, timeout=30) as response:  # nosec B310: fixed HTTPS legal-code URL
        text = response.read().decode("utf-8")
    if not text.startswith("Attribution 4.0 International"):
        raise RuntimeError("Unexpected response while retrieving the CC BY 4.0 legal code")
    return text


def changelog() -> str:
    return """# Changelog

## v1.0

Frozen release corresponding to the associated manuscript's `v1.0-submission` repository snapshot. It contains the formal `review_v1` 815-crop corpus, five fixed repeated source-group-disjoint split definitions, and the 46-crop exclusion list. Expansion and new-date exploratory branches are intentionally excluded.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output must not already exist: {output}")

    base = project / "data/training/paper_four_stage"
    snapshot = base / "manifests/dataset_snapshot_v1_20260915.csv"
    review = base / "review_v1/tomato_images"
    with snapshot.open(encoding="utf-8-sig", newline="") as handle:
        snapshot_rows = list(csv.DictReader(handle))
    expected_fields = {"label", "new_name", "sha256"}
    if not snapshot_rows or not expected_fields.issubset(snapshot_rows[0]):
        raise ValueError("Unexpected snapshot schema")
    accepted = [row for row in snapshot_rows if row["label"] in CLASSES]
    excluded = [row for row in snapshot_rows if row["label"] == "unclassifiable"]
    if len(accepted) != 815 or len(excluded) != 46 or len(accepted) + len(excluded) != len(snapshot_rows):
        raise ValueError("Snapshot does not match frozen 815+46 release contract")
    if Counter(row["label"] for row in accepted) != Counter({
        "immature-period": 222, "green-maturity-period": 223,
        "discoloration-period": 228, "maturity-period": 142,
    }):
        raise ValueError("Unexpected formal class counts")

    per_seed: dict[int, list[dict[str, str]]] = {}
    all_groups: set[str] = set()
    for seed in SEEDS:
        records = records_for_manifest(base / f"yolov8cls_review_v1_seed{seed}/split_manifest.json")
        if len(records) != 815 or Counter(record["split"] for record in records) != Counter(train=572, val=122, test=121):
            raise ValueError(f"Invalid frozen split counts for seed {seed}")
        group_split: dict[str, str] = {}
        for record in records:
            prior = group_split.setdefault(record["source_group"], record["split"])
            if prior != record["split"]:
                raise ValueError(f"Source-group leakage in seed {seed}: {record['source_group']}")
        per_seed[seed] = records
        all_groups.update(group_split)
    if len(all_groups) != 210:
        raise ValueError(f"Expected 210 formal source groups, got {len(all_groups)}")
    group_id = {group: f"G{index:04d}" for index, group in enumerate(sorted(all_groups), start=1)}

    accepted_by_name = {row["new_name"]: row for row in accepted}
    if len(accepted_by_name) != 815:
        raise ValueError("Crop names are not unique")
    for records in per_seed.values():
        if {record["name"] for record in records} != set(accepted_by_name):
            raise ValueError("Split manifest crop set differs from frozen snapshot")

    output.mkdir(parents=True)
    crops = output / "crops"
    splits = output / "splits"
    crops.mkdir(); splits.mkdir()
    crop_id: dict[str, str] = {}
    annotation_rows: list[dict[str, object]] = []
    for index, row in enumerate(accepted, start=1):
        name, label = row["new_name"], row["label"]
        source = review / label / name
        if not source.is_symlink() or not source.exists():
            raise ValueError(f"Expected live reviewed symlink: {source}")
        if sha256(source) != row["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {name}")
        identifier = f"C{index:04d}"
        target = crops / f"{identifier}.png"
        copy_without_metadata(source, target)
        crop_id[name] = identifier
        annotation_rows.append({
            "crop_id": identifier,
            "source_group_id": "",  # populated from the canonical first seed below
            "label": label,
            "label_index": MODEL_INDEX[label],
            "review_status": "reviewed",
        })

    canonical_group = {record["name"]: group_id[record["source_group"]] for record in per_seed[SEEDS[0]]}
    for row in annotation_rows:
        name = next(name for name, identifier in crop_id.items() if identifier == row["crop_id"])
        row["source_group_id"] = canonical_group[name]
    write_csv(output / "annotations.csv", ["crop_id", "source_group_id", "label", "label_index", "review_status"], annotation_rows)

    for seed, records in per_seed.items():
        split_rows = [
            {"crop_id": crop_id[record["name"]], "label": record["label"],
             "source_group_id": group_id[record["source_group"]], "split": record["split"]}
            for record in sorted(records, key=lambda item: crop_id[item["name"]])
        ]
        write_csv(splits / f"split_seed{seed}.csv", ["crop_id", "label", "source_group_id", "split"], split_rows)

    excluded_groups: dict[str, str] = {}
    excluded_rows: list[dict[str, str]] = []
    for index, row in enumerate(excluded, start=1):
        source = review / "unclassifiable" / row["new_name"]
        if not source.is_symlink() or not source.exists() or sha256(source) != row["sha256"]:
            raise ValueError(f"Invalid excluded reviewed link: {row['new_name']}")
        raw_group = source_group_from_target(source)
        excluded_groups.setdefault(raw_group, f"X{len(excluded_groups) + 1:04d}")
        excluded_rows.append({"crop_id": f"X{index:04d}", "source_group_id": excluded_groups[raw_group],
                              "exclusion_reason": "unclassifiable_after_manual_review"})
    write_csv(output / "excluded_unclassifiable.csv", ["crop_id", "source_group_id", "exclusion_reason"], excluded_rows)

    write_text(output / "README.md", release_readme())
    write_text(output / "class_mapping.md", class_mapping())
    write_text(output / "LICENSE", license_text())
    write_text(output / "CHANGELOG.md", changelog())

    manifest_rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "MANIFEST.csv"):
        manifest_rows.append({"relative_path": path.relative_to(output).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(output / "MANIFEST.csv", ["relative_path", "sha256", "bytes"], manifest_rows)
    print(json.dumps({"output": str(output), "crops": len(annotation_rows), "groups": len(set(canonical_group.values())),
                      "excluded": len(excluded_rows), "manifest_files": len(manifest_rows)}, indent=2))


if __name__ == "__main__":
    main()
