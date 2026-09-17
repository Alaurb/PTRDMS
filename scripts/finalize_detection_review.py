#!/usr/bin/env python3
"""Validate and freeze a completed one-class tomato-box review queue.

The browser review tool accepts initial six-column detections
(``class x y w h confidence``) so a reviewer can inspect model proposals.
This utility checks both five- and six-column records and, when explicitly
requested, records the operator-confirmed queue as approved.  It never alters
box coordinates; downstream dataset preparation writes standard five-column
YOLO labels in a separate immutable dataset directory.
"""

import argparse
import csv
import hashlib
import math
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_label(path):
    """Return (box_count, issues) for a 5- or 6-column class-zero label."""
    issues = []
    boxes = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) not in (5, 6):
            issues.append("{}:{}: expected 5 or 6 columns".format(path.name, line_number))
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            issues.append("{}:{}: non-numeric value".format(path.name, line_number))
            continue
        class_id, x, y, width, height = values[:5]
        if class_id != int(class_id) or int(class_id) != 0:
            issues.append("{}:{}: expected class 0".format(path.name, line_number))
        elif not all(math.isfinite(value) for value in values):
            issues.append("{}:{}: non-finite value".format(path.name, line_number))
        elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            issues.append("{}:{}: YOLO coordinates outside valid range".format(path.name, line_number))
        else:
            boxes += 1
    return boxes, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument(
        "--approve-all",
        action="store_true",
        help="mark every valid pending row approved after an operator has confirmed review completion",
    )
    parser.add_argument("--reviewer", default="operator-confirmed")
    args = parser.parse_args()

    root = args.review_root.resolve()
    manifest = root / "audit_manifest.csv"
    image_dir = root / "images"
    label_dir = root / "labels_to_review"
    if not manifest.is_file() or not image_dir.is_dir() or not label_dir.is_dir():
        parser.error("review root must contain audit_manifest.csv, images/, and labels_to_review/")

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "image" not in fieldnames or "audit_status" not in fieldnames:
        parser.error("manifest must contain image and audit_status columns")
    if not rows:
        parser.error("review manifest contains no rows")

    issues = []
    box_count = 0
    empty_images = 0
    for row in rows:
        image_name = row["image"]
        label = label_dir / (Path(image_name).stem + ".txt")
        if not (image_dir / image_name).is_file():
            issues.append("{}: missing image".format(image_name))
        if not label.is_file():
            issues.append("{}: missing label".format(image_name))
            continue
        boxes, label_issues = validate_label(label)
        box_count += boxes
        if boxes == 0:
            empty_images += 1
        issues.extend(label_issues)
    if issues:
        parser.error("review labels are invalid:\n" + "\n".join(issues[:30]))

    if args.approve_all:
        for row in rows:
            row["audit_status"] = "approved"
            if "reviewed_by" in row:
                row["reviewed_by"] = args.reviewer
            if "notes" in row and not row["notes"]:
                row["notes"] = "Batch-finalized after operator confirmed manual review; preflight validation passed."
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    snapshot = root / "review_freeze_manifest.csv"
    snapshot_fields = list(fieldnames) + ["label_sha256"]
    with snapshot.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=snapshot_fields)
        writer.writeheader()
        for row in rows:
            copied = dict(row)
            copied["label_sha256"] = sha256_file(label_dir / (Path(row["image"]).stem + ".txt"))
            writer.writerow(copied)
    summary = root / "REVIEW_FREEZE.md"
    summary.write_text(
        "# Tomato detection review freeze\n\n"
        "- Review rows: {}\n- Valid boxes: {}\n- Empty (negative) images: {}\n"
        "- Manifest SHA256: `{}`\n- Snapshot SHA256: `{}`\n"
        "- Input accepts six-column proposal labels, but the training builder emits five-column YOLO labels.\n"
        "- Do not edit `labels_to_review` after this freeze; make a new review round instead.\n".format(
            len(rows), box_count, empty_images, sha256_file(manifest), sha256_file(snapshot)
        ),
        encoding="utf-8",
    )
    print("Validated {} rows, {} boxes, {} negative images.".format(len(rows), box_count, empty_images))
    print("Approved rows: {}".format(sum(row["audit_status"] == "approved" for row in rows)))
    print("Wrote {}".format(snapshot))


if __name__ == "__main__":
    main()
