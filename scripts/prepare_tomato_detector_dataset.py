#!/usr/bin/env python3
"""Audit and prepare a leakage-aware YOLO tomato detection dataset.

The historical ``yolo-v8/tomato_seg`` directory contains standard five-column
YOLO *detection* labels (class, x_center, y_center, width, height), rather
than polygon labels.  This utility deliberately prepares a detector dataset;
it does not relabel boxes as segmentation masks.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIDE_SUFFIX = re.compile(r"(?:_left|_right|-left|-right)$", re.IGNORECASE)


def source_family(stem: str) -> str:
    """Return a coarse acquisition family so each appears in train and val."""
    if stem.startswith("enhanced_11.29-"):
        return "enhanced_11.29"
    if stem.startswith("11.15-panoramic"):
        return "11.15-panoramic"
    return stem.split("_", 1)[0].split("-", 1)[0]


def capture_group(stem: str) -> str:
    """Keep left/right views of one panorama in the same split."""
    return SIDE_SUFFIX.sub("", stem)


def validate_label(path: Path) -> tuple[int, list[str]]:
    """Validate standard one-class YOLO detection annotations."""
    issues: list[str] = []
    boxes = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            issues.append(f"{path.name}:{line_no}: expected 5 columns")
            continue
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError:
            issues.append(f"{path.name}:{line_no}: non-numeric value")
            continue
        if class_id != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            issues.append(f"{path.name}:{line_no}: values outside YOLO detection range")
            continue
        boxes += 1
    return boxes, issues


def choose_validation(groups: dict[str, list[tuple[Path, Path]]], fraction: float, seed: int) -> set[str]:
    """Sample complete capture groups independently in each acquisition family."""
    by_family: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        by_family[source_family(group)].append(group)

    selected: set[str] = set()
    rng = random.Random(seed)
    for family, family_groups in sorted(by_family.items()):
        rng.shuffle(family_groups)
        target = max(1, round(sum(len(groups[group]) for group in family_groups) * fraction))
        current = 0
        for group in family_groups:
            group_size = len(groups[group])
            if current < target:
                selected.add(group)
                current += group_size
    return selected


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="YOLO directory containing images/train and labels/train")
    parser.add_argument("--output", type=Path, required=True, help="new prepared dataset directory")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 0.5:
        parser.error("--val-fraction must be between 0 and 0.5")

    source = args.source.resolve()
    image_dir, label_dir = source / "images" / "train", source / "labels" / "train"
    if not image_dir.is_dir() or not label_dir.is_dir():
        parser.error("--source must contain images/train and labels/train")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")

    images = {path.stem: path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    labels = {path.stem: path for path in label_dir.glob("*.txt") if path.name != "classes.txt"}
    paired_names = sorted(images.keys() & labels.keys())
    issues: list[str] = []
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    box_count = 0
    for stem in paired_names:
        boxes, label_issues = validate_label(labels[stem])
        issues.extend(label_issues)
        box_count += boxes
        groups[capture_group(stem)].append((images[stem], labels[stem]))
    if issues:
        parser.error("invalid labels:\n" + "\n".join(issues[:20]))
    if not paired_names:
        parser.error("no matched image/label pairs")

    validation_groups = choose_validation(groups, args.val_fraction, args.seed)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    rows: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    family_counts: Counter[tuple[str, str]] = Counter()
    for group, pairs in sorted(groups.items()):
        split = "val" if group in validation_groups else "train"
        for image, label in pairs:
            link(image, output / "images" / split / image.name)
            link(label, output / "labels" / split / label.name)
            split_counts[split] += 1
            family_counts[(split, source_family(image.stem))] += 1
            rows.append({"split": split, "capture_group": group, "family": source_family(image.stem), "image": str(image.resolve()), "label": str(label.resolve())})

    (output / "data.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: images/val\nnames:\n  0: tomato\n",
        encoding="utf-8",
    )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "capture_group", "family", "image", "label"])
        writer.writeheader()
        writer.writerows(rows)
    report = output / "AUDIT.md"
    report.write_text(
        "# Tomato detector dataset audit\n\n"
        "- Label format: one-class YOLO detection (5 columns), not segmentation polygons.\n"
        f"- Valid matched image/label pairs: {len(paired_names)}\n"
        f"- Valid bounding boxes: {box_count}\n"
        f"- Images without labels (excluded): {len(images) - len(paired_names)}\n"
        f"- Labels without images (excluded): {len(labels) - len(paired_names)}\n"
        f"- Capture groups: {len(groups)}; validation groups: {len(validation_groups)}\n"
        f"- Split images: train {split_counts['train']}, val {split_counts['val']}\n\n"
        "## Acquisition-family split\n\n"
        + "\n".join(f"- {split}/{family}: {count}" for (split, family), count in sorted(family_counts.items()))
        + "\n\nThe files in this dataset are symbolic links to the historical source data.\n",
        encoding="utf-8",
    )
    print(f"Prepared {output}")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
