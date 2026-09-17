#!/usr/bin/env python3
"""Register a reproducible Ultralytics YOLO detection/segmentation experiment.

The registry deliberately stores metrics from the same epoch that maximises
box mAP50-95, plus immutable hashes for the split, run configuration and best
checkpoint.  It is intended for paper-facing comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


METRIC_COLUMNS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
TABLE_COLUMNS = (
    "experiment_id",
    "status",
    "model",
    "task",
    "best_epoch",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "split_manifest_sha256",
    "data_yaml_sha256",
    "args_yaml_sha256",
    "best_weight_sha256",
    "run_dir",
    "notes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_best_metrics(results_csv: Path) -> dict[str, str]:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(column not in rows[0] for column in METRIC_COLUMNS):
        raise ValueError(f"{results_csv} does not contain YOLO box metrics")
    return max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))


def write_markdown(rows: list[dict[str, str]], target: Path) -> None:
    headings = ("Experiment", "Status", "Model", "Epoch", "P", "R", "mAP50", "mAP50-95", "Notes")
    lines = ["# YOLO experiment comparison", "", "All rows use the split hashes recorded below. A trial is accepted only if its mAP50-95 improves and P/R/mAP50 do not decrease versus the current accepted baseline.", "", "| " + " | ".join(headings) + " |", "|" + "|".join(["---"] * len(headings)) + "|"]
    for row in rows:
        values = (
            row["experiment_id"], row["status"], row["model"], row["best_epoch"],
            f"{float(row['precision']):.4f}", f"{float(row['recall']):.4f}",
            f"{float(row['map50']):.4f}", f"{float(row['map50_95']):.4f}", row["notes"],
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Reproducibility records")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['experiment_id']}`: split manifest `{row['split_manifest_sha256']}`, data YAML `{row['data_yaml_sha256']}`, args `{row['args_yaml_sha256']}`, best weights `{row['best_weight_sha256']}`.")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--status", choices=("accepted_baseline", "accepted", "rejected", "exploratory"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    results_csv, args_yaml, weights = run_dir / "results.csv", run_dir / "args.yaml", run_dir / "weights" / "best.pt"
    for path in (results_csv, args_yaml, weights, args.split_manifest, args.data_yaml):
        if not path.is_file():
            parser.error(f"required file is missing: {path}")
    best = load_best_metrics(results_csv)
    registry = args.registry_dir.resolve()
    (registry / "configs").mkdir(parents=True, exist_ok=True)
    copied_args = registry / "configs" / f"{args.experiment_id}.args.yaml"
    shutil.copy2(args_yaml, copied_args)
    row = {
        "experiment_id": args.experiment_id,
        "status": args.status,
        "model": args.model,
        "task": args.task,
        "best_epoch": best["epoch"],
        "precision": best["metrics/precision(B)"],
        "recall": best["metrics/recall(B)"],
        "map50": best["metrics/mAP50(B)"],
        "map50_95": best["metrics/mAP50-95(B)"],
        "split_manifest_sha256": sha256(args.split_manifest),
        "data_yaml_sha256": sha256(args.data_yaml),
        "args_yaml_sha256": sha256(copied_args),
        "best_weight_sha256": sha256(weights),
        "run_dir": str(run_dir),
        "notes": args.notes,
    }
    table = registry / "experiment_table.csv"
    previous: list[dict[str, str]] = []
    if table.exists():
        with table.open(newline="", encoding="utf-8") as handle:
            previous = [old for old in csv.DictReader(handle) if old["experiment_id"] != args.experiment_id]
    accepted = [old for old in previous if old["status"] in {"accepted_baseline", "accepted"}]
    if args.status == "accepted_baseline" and accepted:
        parser.error("an accepted baseline already exists; register later improvements as accepted or rejected")
    if args.status == "accepted":
        if not accepted:
            parser.error("an accepted experiment requires an existing accepted baseline")
        reference = accepted[-1]
        candidate = {
            "precision": float(best["metrics/precision(B)"]),
            "recall": float(best["metrics/recall(B)"]),
            "map50": float(best["metrics/mAP50(B)"]),
            "map50_95": float(best["metrics/mAP50-95(B)"]),
        }
        reference_metrics = {name: float(reference[name]) for name in candidate}
        if not (
            candidate["map50_95"] > reference_metrics["map50_95"]
            and candidate["precision"] >= reference_metrics["precision"]
            and candidate["recall"] >= reference_metrics["recall"]
            and candidate["map50"] >= reference_metrics["map50"]
        ):
            parser.error(
                "candidate fails acceptance rule against " + reference["experiment_id"]
                + ": mAP50-95 must improve, while precision, recall and mAP50 cannot decrease"
            )
    previous.append(row)
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(previous)
    write_markdown(previous, registry / "experiment_table.md")
    print(f"registered {args.experiment_id} in {table}")


if __name__ == "__main__":
    main()
