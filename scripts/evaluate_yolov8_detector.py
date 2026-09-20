#!/usr/bin/env python3
"""Evaluate a fixed YOLOv8 detector weight on one declared dataset split."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="trained .pt weight")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--project", required=True, help="directory for evaluation artifacts")
    parser.add_argument("--name", required=True, help="unique evaluation artifact directory name")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    metrics = YOLO(args.model).val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=False,
        plots=True,
    )
    print(json.dumps(metrics.results_dict, sort_keys=True))


if __name__ == "__main__":
    main()
