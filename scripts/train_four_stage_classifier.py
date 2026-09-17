"""Train a four-stage tomato-ripeness classifier using Ultralytics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FOUR_STAGE_CLASSES = ("immature-period", "green-maturity-period", "discoloration-period", "maturity-period")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a four-stage tomato-ripeness model")
    parser.add_argument("--task", required=True, choices=("classify",))
    parser.add_argument("--data", required=True, help="YOLO detect YAML or classification dataset root")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Pretrained Ultralytics YOLOv8 classification model")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=-1, help="-1 selects automatic batch size")
    parser.add_argument("--project", default="runs/four_stage_maturity")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def _check_classifier_layout(root: Path) -> None:
    root = root.resolve()
    missing = [str(root / split / label) for split in ("train", "val", "test") for label in FOUR_STAGE_CLASSES if not (root / split / label).is_dir()]
    if missing:
        raise ValueError("Classifier dataset must contain all four-stage class folders; missing: " + ", ".join(missing))
    manifest_path = root / "split_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Prepare data with prepare_four_stage_classifier.py to create a split manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "prms.classifier_dataset.v1" or not manifest.get("crops"):
        raise ValueError("Unsupported or empty classifier split manifest")
    expected = set()
    for record in manifest["crops"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or path in expected or not path.is_file():
            raise ValueError("Invalid or duplicate crop in split manifest")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Crop changed after preparation: {record['path']}")
        expected.add(path)
    actual = set()
    for split in ("train", "val", "test"):
        folders = {p.name for p in (root / split).iterdir() if p.is_dir()}
        if folders != set(FOUR_STAGE_CLASSES):
            raise ValueError(f"Unexpected class folders in {split}")
        for label in FOUR_STAGE_CLASSES:
            files = {p.resolve() for p in (root / split / label).rglob("*") if p.is_file()}
            if split in ("train", "val") and not files:
                raise ValueError(f"Empty class: {split}/{label}")
            actual.update(files)
    if actual != expected:
        raise ValueError("Dataset files differ from the recorded split manifest")


def main() -> None:
    args = parse_args()
    data = Path(args.data)
    if not data.is_dir():
        raise FileNotFoundError("--data must be a classification dataset directory")
    _check_classifier_layout(data)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install optional-requirements.txt before training") from exc
    model = YOLO(args.model)
    if model.task != "classify":
        raise ValueError(f"Requested classify, but checkpoint task is {model.task}")
    result = model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name)
    print(f"Training complete. Results: {result.save_dir}")
    print("Before deployment, validate on a route/date held out from all training images.")


if __name__ == "__main__":
    main()
