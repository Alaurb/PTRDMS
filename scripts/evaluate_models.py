from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"


def portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    for base in (PROJECT_ROOT, DATA_ROOT):
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return resolved.name


def plain_metrics(result: object) -> dict[str, float]:
    values = getattr(result, "results_dict", {})
    return {str(key): float(value) for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector and classifier with fixed settings")
    parser.add_argument("--detector", default=str(PROJECT_ROOT / "models/tomato_detector.pt"))
    parser.add_argument("--classifier", default=str(PROJECT_ROOT / "models/tomato_ripeness_classifier.pt"))
    parser.add_argument("--detector-data", default=str(DATA_ROOT / "training/detector/dataset.yaml"))
    parser.add_argument("--classifier-data", default=str(DATA_ROOT / "training/classifier"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs/evaluation/metrics.json"))
    parser.add_argument("--name", default="evaluation")
    parser.add_argument("--device", default="0")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--classifier-imgsz", type=int, default=224)
    args = parser.parse_args()

    # Fail instead of allowing validators to fall back to a different split.
    if not (Path(args.classifier_data) / args.split).is_dir():
        raise FileNotFoundError(f"Classifier split does not exist: {args.split}")
    import yaml
    detector_config = yaml.safe_load(Path(args.detector_data).read_text(encoding="utf-8"))
    if not detector_config.get(args.split):
        raise ValueError(f"Detector YAML must explicitly declare {args.split}")

    output = Path(args.output).resolve()
    run_root = output.parent / "validation_runs"
    output.parent.mkdir(parents=True, exist_ok=True)

    detector_result = YOLO(args.detector).val(
        data=str(Path(args.detector_data).resolve()),
        split=args.split,
        imgsz=1280,
        batch=4,
        device=args.device,
        plots=args.plots,
        project=str(run_root),
        name=f"{args.name}_detector",
    )
    classifier_result = YOLO(args.classifier).val(
        data=str(Path(args.classifier_data).resolve()),
        split=args.split,
        imgsz=args.classifier_imgsz,
        batch=32,
        device=args.device,
        plots=args.plots,
        project=str(run_root),
        name=f"{args.name}_classifier",
    )
    payload = {
        "name": args.name,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "models": {"detector": portable_path(args.detector), "classifier": portable_path(args.classifier)},
        "settings": {"detector_imgsz": 1280, "classifier_imgsz": args.classifier_imgsz, "split": args.split},
        "scope": "component evaluation; not end-to-end pipeline performance",
        "detector": plain_metrics(detector_result),
        "classifier": plain_metrics(classifier_result),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
