#!/usr/bin/env python3
"""Create a standalone, local review package from equirectangular panoramas.

The resulting package is deliberately separate from a frozen detector split.
It contains projected left/right views, empty YOLO label files, and an audit
manifest for ``serve_detection_review.py``.  Approval in the browser is only
an annotation decision; a later capture-group leakage check is still required
before any image may be added to training.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

from PIL import Image

# Let the helper run directly from ``scripts/`` as documented, without
# requiring users to set PYTHONPATH manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ripeness_demo.panorama import extract_side_views, list_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panorama-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--package-role",
        choices=("candidate-training", "heldout-test"),
        default="candidate-training",
        help="intended post-review role; never changes the current frozen split",
    )
    parser.add_argument("--face-size", type=int, default=1440)
    parser.add_argument(
        "--camera-yaw-offset-deg",
        type=float,
        default=0.0,
        help="calibrated native panorama forward-axis offset from robot forward",
    )
    args = parser.parse_args()
    if args.face_size < 64:
        parser.error("--face-size must be at least 64")

    panoramas = list_images(args.panorama_dir)
    if not panoramas:
        parser.error("no panorama images were found")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")

    image_dir = output / "images"
    label_dir = output / "labels_to_review"
    panorama_dir = output / "panoramas"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    panorama_dir.mkdir()
    rows: list[dict[str, str]] = []
    offset_rad = math.radians(args.camera_yaw_offset_deg)
    for panorama_path in panoramas:
        # Retain the exact source panorama next to its projected review views,
        # so future leakage checks and projection audits remain reproducible.
        shutil.copy2(panorama_path, panorama_dir / panorama_path.name)
        with Image.open(panorama_path) as panorama:
            views = extract_side_views(
                panorama,
                args.face_size,
                camera_yaw_offset_rad=offset_rad,
            )
        for side, view in views.items():
            image_name = f"{panorama_path.stem}_{side}.jpg"
            view.save(image_dir / image_name, quality=95)
            (label_dir / f"{Path(image_name).stem}.txt").touch()
            rows.append(
                {
                    "image": image_name,
                    "source_panorama": panorama_path.name,
                    "capture_group": panorama_path.stem,
                    "side": side,
                    "intended_split": args.package_role,
                    "initial_box_count": "0",
                    "audit_status": "pending",
                }
            )

    manifest = output / "audit_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "README.md").write_text(
        "# Local tomato-detector annotation review\n\n"
        f"- Source panoramas: {len(panoramas)}\n"
        f"- Projected review views: {len(rows)}\n"
        "- Original panoramas: retained in `panoramas/` with source timestamps.\n"
        f"- Intended post-review role: `{args.package_role}`.\n"
        "- This package is not part of the current frozen detector split.\n"
        "- Keep both sides of every `capture_group` in the same eventual split.\n",
        encoding="utf-8",
    )
    print(f"Created {output} with {len(rows)} projected review views.")


if __name__ == "__main__":
    main()
