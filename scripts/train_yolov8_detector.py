#!/usr/bin/env python3
"""Run one reproducible YOLOv8 one-class detector training experiment."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--project", required=True, help="Ultralytics experiment project directory")
    parser.add_argument("--name", required=True, help="unique run name within --project")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-amp", action="store_true", help="disable AMP checks and mixed-precision training")
    args = parser.parse_args()

    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("epochs, imgsz, and batch must be positive")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        amp=not args.no_amp,
        project=args.project,
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
