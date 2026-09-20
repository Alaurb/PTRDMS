#!/usr/bin/env python3
"""Build a reproducible YOLOv8 dataset with fixed validation and test splits.

The original detector split remains untouched.  Approved reviewed images are
added only to the training side, unless their panorama capture group belongs
to the frozen validation split, in which case they are explicitly excluded to
prevent leakage.  An optional independently reviewed holdout can be added as
the test split, but it must not share a capture group with any base or
training-review image. Images are linked; labels are copied and normalized to
the five-column YOLO detection format to make this dataset a data freeze.
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


def load_frozen_review(root, role):
    """Return a frozen, operator-approved review package and its rows."""
    root = root.resolve()
    manifest = root / "review_freeze_manifest.csv"
    if not manifest.is_file():
        raise ValueError("{} review freeze manifest is missing: {}".format(role, manifest))
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("{} review manifest is empty".format(role))
    if any(row.get("audit_status") != "approved" for row in rows):
        raise ValueError("every {} review row must be approved".format(role))
    return root, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True, help="frozen tomato_detector_v1 directory")
    parser.add_argument("--review-root", type=Path, required=True, help="review queue containing an approved freeze")
    parser.add_argument("--test-review-root", type=Path, help="independent approved review queue for the held-out test split")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; must not already exist")
    args = parser.parse_args()

    base = args.base_dataset.resolve()
    review, review_rows = load_frozen_review(args.review_root, "training")
    test_review = test_rows = None
    if args.test_review_root:
        test_review, test_rows = load_frozen_review(args.test_review_root, "test")
    output = args.output.resolve()
    if output.exists():
        parser.error("refusing to overwrite {}".format(output))
    base_manifest = base / "manifest.csv"
    review_manifest = review / "review_freeze_manifest.csv"
    if not base_manifest.is_file():
        parser.error("base manifest is missing")

    with base_manifest.open(newline="", encoding="utf-8-sig") as handle:
        base_rows = list(csv.DictReader(handle))
    if not base_rows:
        parser.error("base manifest is empty")

    base_val_groups = {row["capture_group"] for row in base_rows if row["split"] == "val"}
    base_groups = {row["capture_group"] for row in base_rows}
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

    heldout = []
    if test_review:
        test_images = test_review / "images"
        test_labels = test_review / "labels_to_review"
        included_groups = {item["capture_group"] for item in included}
        seen_names = set(base_names) | {item["image"].name for item in included}
        for row in test_rows:
            image = test_images / row["image"]
            label = test_labels / (Path(row["image"]).stem + ".txt")
            group = capture_group(image.stem)
            if not image.is_file() or not label.is_file():
                parser.error("missing held-out image or label for {}".format(row["image"]))
            if group in base_groups or group in included_groups:
                parser.error("held-out capture group overlaps training or validation: {}".format(group))
            if image.name in seen_names:
                parser.error("held-out image duplicates another split: {}".format(image.name))
            seen_names.add(image.name)
            heldout.append({"image": image, "label": label, "capture_group": group, "row": row})

    output.mkdir(parents=True)
    for split in ("train", "val", "test"):
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
    for item in heldout:
        image, label = item["image"], item["label"]
        link(image, output / "images" / "test" / image.name)
        write_label(label, output / "labels" / "test" / (image.stem + ".txt"))
        manifest_rows.append({
            "split": "test", "source_kind": "review_holdout_test", "capture_group": item["capture_group"],
            "image": str(image.resolve()), "label": str(label.resolve()),
            "image_sha256": sha256_file(image), "label_sha256": sha256_file(output / "labels" / "test" / (image.stem + ".txt")),
        })

    (output / "data.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: tomato\n".format(output), encoding="utf-8"
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
        "- Independent held-out test images: {}\n"
        "- Base training images: {}\n- Included reviewed training images: {}\n"
        "- Training images after expansion: {}\n- Reviewed images excluded to protect validation groups: {}\n"
        "- Base manifest SHA256: `{}`\n- Training-review freeze SHA256: `{}`\n"
        "- Test-review freeze SHA256: `{}`\n- Output manifest SHA256: `{}`\n"
        "- Images are symbolic links to source files; labels are copied, normalized five-column YOLO files.\n".format(
            counts["val"], counts["test"], sum(row["split"] == "train" and row["source_kind"] == "base_v1" for row in manifest_rows),
            len(included), counts["train"], len(excluded), sha256_file(base_manifest),
            sha256_file(review_manifest),
            sha256_file(test_review / "review_freeze_manifest.csv") if test_review else "not_provided",
            sha256_file(output / "manifest.csv")
        ), encoding="utf-8"
    )
    print("Built {}".format(output))
    print("train={} (base={}, reviewed={}); val={} fixed; test={} held-out; excluded_review={}".format(
        counts["train"], counts["train"] - len(included), len(included), counts["val"], counts["test"], len(excluded)
    ))


if __name__ == "__main__":
    main()
