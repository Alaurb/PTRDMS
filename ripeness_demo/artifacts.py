"""Evidence contracts shared by inference, replay, and the local report.

The report must never make a stronger spatial or maturity claim than the input
data supports.  This module turns that rule into a small versioned artifact.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


SCHEMA = "prms.evidence.v1"


def build_evidence_manifest(
    summary: dict[str, object],
    poses: Iterable[object],
    *,
    processing_mode: str,
    range_source: str,
) -> dict[str, object]:
    """Return an explicit claim boundary for one result directory."""
    pose_sources = sorted({str(getattr(pose, "source", "unknown")) for pose in poses})
    is_replay = processing_mode == "saved_prediction_replay"
    has_measured_geometry = pose_sources == ["pose_csv"] and range_source == "registered_measured_range"
    if is_replay:
        status = "archived_legacy_replay"
        headline = "Archived observations replayed; no model inference was performed."
    elif has_measured_geometry:
        status = "measured_geometry_candidate"
        headline = "Measured pose and registered range were supplied; associations remain candidates."
    else:
        status = "unvalidated_observation_layout"
        headline = "Predictions are arranged as observations; geometry is not validated fruit location."

    established = ["frame-side image observations", "model-predicted class and confidence", "source-image traceability"]
    if has_measured_geometry:
        established.append("range-backed projection candidates")
    not_established = [
        "field-verified four-stage maturity accuracy",
        "unique-fruit count or biological truss identity",
        "validated metric fruit location without independent geometry evaluation",
    ]
    if is_replay:
        not_established.insert(0, "current-model performance")
    return {
        "schema": SCHEMA,
        "taxonomy": summary.get("taxonomy", "unspecified"),
        "detector": summary.get("detector", "unspecified"),
        "status": status,
        "headline": headline,
        "processing_mode": processing_mode,
        "pose_sources": pose_sources,
        "range_source": range_source,
        "established": established,
        "not_established": not_established,
        "counts": {
            "frames": summary.get("processed_frames", 0),
            "observations": summary.get("detections", 0),
        },
    }


def write_evidence_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    (output_dir / "evidence.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
