"""Create natural-background crops for reviewed four-stage SAM2 samples.

The review folders contain symlinks to SAM2 masked crops.  Their candidate
manifests retain the original image and detection box.  This script resolves
those links, recreates an expanded original-image crop, and writes a manifest
that keeps every generated image traceable to its reviewed SAM2 sample.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image


CLASSES = ("immature-period", "green-maturity-period", "discoloration-period", "maturity-period")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--padding", type=float, default=.08)
    return parser.parse_args()


def manifest_for_crop(crop: Path, project_root: Path) -> Path:
    """Find the candidate manifest belonging to a crop target."""
    crop = crop.resolve()
    for parent in crop.parents:
        candidate = parent / "manifest.jsonl"
        if candidate.is_file():
            try:
                candidate.relative_to(project_root)
            except ValueError:
                continue
            return candidate
    raise FileNotFoundError(f"No candidate manifest above {crop}")


def load_manifest(path: Path) -> dict[str, dict]:
    entries = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            crop_name = Path(record.get("crop", f"crops/{record['id']}.png")).name
            if crop_name in entries:
                raise ValueError(f"Duplicate crop entry in {path}:{line_number}")
            entries[crop_name] = record
    return entries


def expanded_box(box, width: int, height: int, padding: float):
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid box: {box}") from exc
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or not (x1 < x2 and y1 < y2):
        raise ValueError(f"Box outside source image: {box}, image={width}x{height}")
    # Grounding DINO boxes can overshoot an image edge by a small floating-point
    # amount (for example -0.01); clamp that representation error before crop.
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"Box has no in-image area: {box}, image={width}x{height}")
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    return (max(0, math.floor(x1 - pad_x)), max(0, math.floor(y1 - pad_y)),
            min(width, math.ceil(x2 + pad_x)), min(height, math.ceil(y2 + pad_y)))


def main() -> None:
    args = parse_args()
    if not 0 <= args.padding <= .5:
        raise ValueError("--padding must be between 0 and 0.5")
    root, review, output = args.project_root.resolve(), args.review_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output must be new: {output}")
    manifest_cache: dict[Path, dict[str, dict]] = {}
    plan = []
    for label in CLASSES:
        folder = review / label
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for link in sorted(path for path in folder.iterdir() if path.is_file()):
            target = link.resolve()
            manifest_path = manifest_for_crop(target, root)
            records = manifest_cache.setdefault(manifest_path, load_manifest(manifest_path))
            record = records.get(target.name)
            if record is None:
                raise KeyError(f"No manifest entry for {target}")
            source = (root / record["source_image"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = Path(label) / f"{link.stem}.jpg"
            plan.append((label, link, target, source, record["box_xyxy"], destination, manifest_path, record["id"]))
    # Validate all sources and boxes before creating any output.
    for _, _, _, source, box, _, _, _ in plan:
        with Image.open(source) as image:
            expanded_box(box, *image.size, args.padding)
    for label in CLASSES:
        (output / label).mkdir(parents=True)
    rows = []
    for label, link, target, source, box, destination, manifest_path, candidate_id in plan:
        with Image.open(source) as image:
            crop = image.convert("RGB").crop(expanded_box(box, *image.size, args.padding))
            crop.save(output / destination, quality=95)
        rows.append({"label": label, "review_link": str(link.relative_to(review)),
                     "sam2_crop": str(target.relative_to(root)), "source_image": str(source.relative_to(root)),
                     "box_xyxy": json.dumps(box), "candidate_id": candidate_id,
                     "candidate_manifest": str(manifest_path.relative_to(root)), "output": str(destination)})
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"crops": len(rows), "output": str(output), "padding": args.padding}, ensure_ascii=False))


if __name__ == "__main__":
    main()
