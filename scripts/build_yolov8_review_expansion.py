#!/usr/bin/env python3
"""Build a reproducible YOLOv8 dataset with a fixed validation split.

The original detector split remains untouched.  Approved reviewed images are
added only to the training side, unless their panorama capture group belongs
to the frozen validation split, in which case they are explicitly excluded to
prevent leakage.  Images are linked; labels are copied and normalized to the
five-column YOLO detection format to make this dataset a data freeze.
"""

import argparse
import csv
import hashlib
import math
import re
import shutil
from collections import Counter
from pathlib import Path


SIDE_SUFFIX = re.compile(r"(?:_left|_right|-left|-right)$", re.IGNORECASE)


def capture_group(stem):
    return SIDE_SUFFIX.sub("", stem)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_label(source):
    """Read a class-zero five/six-column review label and return YOLO5 text."""
    output = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) not in (5, 6):
            raise ValueError("{}:{} has {} columns".format(source, line_number, len(fields)))
        try:
            values = [float(value) for value in fields]
        except ValueError:
            raise ValueError("{}:{} is non-numeric".format(source, line_number))
        class_id, x, y, width, height = values[:5]
        if class_id != int(class_id) or int(class_id) != 0:
            raise ValueError("{}:{} is not class zero".format(source, line_number))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("{}:{} has non-finite values".format(source, line_number))
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError("{}:{} has invalid geometry".format(source, line_number))
        output.append("0 {:.8f} {:.8f} {:.8f} {:.8f}".format(x, y, width, height))
    return "\n".join(output) + ("\n" if output else "")


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def write_label(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(normalized_label(source), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True, help="frozen tomato_detector_v1 directory")
    parser.add_argument("--review-root", type=Path, required=True, help="review queue containing an approved freeze")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; must not already exist")
    args = parser.parse_args()

    base = args.base_dataset.resolve()
    review = args.review_root.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("refusing to overwrite {}".format(output))
    base_manifest = base / "manifest.csv"
    review_manifest = review / "review_freeze_manifest.csv"
    if not base_manifest.is_file() or not review_manifest.is_file():
        parser.error("base manifest or frozen review manifest is missing")

    with base_manifest.open(newline="", encoding="utf-8-sig") as handle:
        base_rows = list(csv.DictReader(handle))
    with review_manifest.open(newline="", encoding="utf-8-sig") as handle:
        review_rows = list(csv.DictReader(handle))
    if not base_rows or not review_rows:
        parser.error("input manifest is empty")
    if any(row.get("audit_status") != "approved" for row in review_rows):
        parser.error("every frozen review row must be approved")

    base_val_groups = {row["capture_group"] for row in base_rows if row["split"] == "val"}
    base_names = {Path(row["image"]).name for row in base_rows}
    review_images = review / "images"
    review_labels = review / "labels_to_review"
    included, excluded = [], []
    for row in review_rows:
        image = review_images / row["image"]
        label = review_labels / (Path(row["image"]).stem + ".txt")
        group = capture_group(image.stem)
        if not image.is_file() or not label.is_file():
            parser.error("missing reviewed image or label for {}".format(row["image"]))
        if image.name in base_names:
            parser.error("review image duplicates base dataset: {}".format(image.name))
        item = {"image": image, "label": label, "capture_group": group, "row": row}
        if group in base_val_groups:
            excluded.append(item)
        else:
            included.append(item)

    output.mkdir(parents=True)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    manifest_rows = []
    for row in base_rows:
        split = row["split"]
        image, label = Path(row["image"]), Path(row["label"])
        link(image, output / "images" / split / image.name)
        write_label(label, output / "labels" / split / label.name)
        manifest_rows.append({
            "split": split, "source_kind": "base_v1", "capture_group": row["capture_group"],
            "image": str(image.resolve()), "label": str(label.resolve()),
            "image_sha256": sha256_file(image), "label_sha256": sha256_file(output / "labels" / split / label.name),
        })
    for item in included:
        image, label = item["image"], item["label"]
        link(image, output / "images" / "train" / image.name)
        write_label(label, output / "labels" / "train" / (image.stem + ".txt"))
        manifest_rows.append({
            "split": "train", "source_kind": "review_v1", "capture_group": item["capture_group"],
            "image": str(image.resolve()), "label": str(label.resolve()),
            "image_sha256": sha256_file(image), "label_sha256": sha256_file(output / "labels" / "train" / (image.stem + ".txt")),
        })

    (output / "data.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: tomato\n".format(output), encoding="utf-8"
    )
    fields = ["split", "source_kind", "capture_group", "image", "label", "image_sha256", "label_sha256"]
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output / "excluded_review_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "capture_group", "reason"])
        writer.writeheader()
        for item in excluded:
            writer.writerow({"image": item["image"].name, "capture_group": item["capture_group"], "reason": "capture_group_is_in_fixed_validation_split"})
    counts = Counter(row["split"] for row in manifest_rows)
    (output / "DATA_FREEZE.md").write_text(
        "# YOLOv8 tomato detector expansion v1\n\n"
        "- Frozen validation images: {} (identical to `tomato_detector_v1`)\n"
        "- Base training images: {}\n- Included reviewed training images: {}\n"
        "- Training images after expansion: {}\n- Reviewed images excluded to protect validation groups: {}\n"
        "- Base manifest SHA256: `{}`\n- Review freeze SHA256: `{}`\n- Output manifest SHA256: `{}`\n"
        "- Images are symbolic links to source files; labels are copied, normalized five-column YOLO files.\n".format(
            counts["val"], sum(row["split"] == "train" and row["source_kind"] == "base_v1" for row in manifest_rows),
            len(included), counts["train"], len(excluded), sha256_file(base_manifest),
            sha256_file(review_manifest), sha256_file(output / "manifest.csv")
        ), encoding="utf-8"
    )
    print("Built {}".format(output))
    print("train={} (base={}, reviewed={}); val={} fixed; excluded_review={}".format(
        counts["train"], counts["train"] - len(included), len(included), counts["val"], len(excluded)
    ))


if __name__ == "__main__":
    main()
