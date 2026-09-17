"""Create classifier crops only after validating all labels and split provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ripeness_demo.splits import validate_splits

CLASSES = ("immature-period", "green-maturity-period", "discoloration-period", "maturity-period")
REQUIRED = {"image", "x1", "y1", "x2", "y2", "label", "split",
            "panorama_id", "group_id", "capture_date", "route"}


def prepare(annotations, images, output, padding=0.0):
    """Preflight before any output write; retain source and crop hashes."""
    if not math.isfinite(padding) or not 0 <= padding <= .5:
        raise ValueError("Padding must be between 0 and 0.5")
    image_root, output = Path(images).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be new or empty: {output}")
    with Path(annotations).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(REQUIRED - set(reader.fieldnames or []))))
        rows = [{k: v.strip() if isinstance(v, str) else v for k, v in row.items()} for row in reader]
    hashes = validate_splits(rows, image_root)
    if {r["split"] for r in rows} != {"train", "val", "test"}:
        raise ValueError("Provide separate train, val and test groups")
    counts = Counter()
    planned = []
    for index, row in enumerate(rows, start=2):
        if row["label"] not in CLASSES:
            raise ValueError(f"Row {index}: unknown class {row['label']}")
        if row.get("readability", "").lower() in {"ungradable", "unresolved"}:
            raise ValueError(f"Row {index}: resolve ungradable fruit before training")
        source = image_root / row["image"]
        with Image.open(source) as image:
            image.load()
            width, height = image.size
        try:
            x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {index}: invalid box") from exc
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Row {index}: box must be finite, ordered and inside the image")
        pad_x, pad_y = (x2-x1)*padding, (y2-y1)*padding
        box = (max(0,int(x1-pad_x)), max(0,int(y1-pad_y)),
               min(width,math.ceil(x2+pad_x)), min(height,math.ceil(y2+pad_y)))
        if box[2]-box[0] < 8 or box[3]-box[1] < 8:
            raise ValueError(f"Row {index}: crop is too small")
        target = f"{row['split']}/{row['label']}/crop_{index:06d}.jpg"
        planned.append((source, box, target, index))
        counts[(row["split"], row["label"])] += 1
    missing = [f"{split}/{label}" for split in ("train","val") for label in CLASSES if not counts[(split,label)]]
    if missing:
        raise ValueError("Every class needs train and val examples; missing: " + ", ".join(missing))
    # No files or directories have been written before this point.
    for split in ("train", "val", "test"):
        for label in CLASSES:
            (output / split / label).mkdir(parents=True, exist_ok=True)
    crops = []
    for source, box, relative, index in planned:
        target = output / relative
        with Image.open(source) as image:
            image.convert("RGB").crop(box).save(target, quality=95)
        crops.append(dict(path=relative, row=index, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    manifest = dict(schema="prms.classifier_dataset.v1", rows=rows, image_sha256=hashes,
                    padding=padding, crops=crops)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--padding", type=float, default=0.0)
    args = parser.parse_args()
    counts = prepare(args.annotations, args.images, args.output, args.padding)
    for split in ("train", "val", "test"):
        print(split + ": " + ", ".join(f"{label}={counts[(split,label)]}" for label in CLASSES))


if __name__ == "__main__":
    main()
