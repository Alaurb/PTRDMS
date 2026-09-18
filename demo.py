from __future__ import annotations

import argparse
import json
import hashlib
import importlib.metadata
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ripeness_demo.detectors import annotate, build_detector
from ripeness_demo.evidence import RangeEvidence, foreground_range, associate
from ripeness_demo.artifacts import build_evidence_manifest, write_evidence_manifest
from ripeness_demo.mapping import generate_demo_poses, load_pose_csv, project_detection
from ripeness_demo.panorama import extract_side_views, extract_all_faces, list_images
from ripeness_demo.report import draw_map_overlay, write_csv_files, write_html_report, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panoramic tomato maturity detection and spatial mapping")
    parser.add_argument("--input", required=True, help="Directory containing panorama images")
    parser.add_argument("--map", required=True, dest="map_path", help="2D occupancy or farm map image")
    parser.add_argument("--output", default="demo_output", help="Output directory")
    parser.add_argument("--pose-csv", help="Measured poses in frame,x,y,z,yaw[,route] CSV format")
    parser.add_argument("--weights", help="Optional four-class YOLO weights")
    parser.add_argument("--detector-weights", help="Optional tomato detector weights")
    parser.add_argument("--classifier-weights", help="Optional ripeness classifier weights")
    parser.add_argument("--detector", choices=["auto", "yolo", "two-stage", "color"], default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--brightness-gain", type=float, default=1.25, help="Brightness gain for the enhanced detection branch; 1.0 disables enhancement")
    parser.add_argument("--detector-imgsz", type=int, default=1280, help="YOLO detector input size")
    parser.add_argument("--classifier-imgsz", type=int, help="Classifier input size; defaults to checkpoint setting")
    parser.add_argument("--face-size", type=int, default=1440)
    parser.add_argument("--max-frames", type=int, default=24, help="Maximum frames to process; 0 processes every image")
    parser.add_argument("--map-width-m", type=float, default=35.0)
    parser.add_argument("--row-offset-m", type=float, default=1.15, help="Assumed camera-to-row distance when depth is unavailable")
    parser.add_argument("--camera-height-m", type=float, default=1.15)
    parser.add_argument("--camera-yaw-offset-deg", type=float, default=0.0,
                        help="Calibrated native panorama forward-axis offset from robot forward, in degrees")
    parser.add_argument("--export-six-faces", action="store_true")
    parser.add_argument("--range-manifest", help="Registered radial-range NPY maps and synchronization metadata")
    parser.add_argument("--sync-tolerance-s", type=float, default=.05)
    parser.add_argument("--row-tolerance-m", type=float, default=.35)
    parser.add_argument("--association-distance-m", type=float, default=.12)
    parser.add_argument("--association-gap", type=int, default=3)
    return parser.parse_args()


def _sample(paths: list[Path], maximum: int) -> list[Path]:
    if maximum <= 0 or maximum >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, maximum).round().astype(int)
    return [paths[index] for index in dict.fromkeys(indices.tolist())]


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", path.stem)


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    if args.row_tolerance_m <= 0 or args.map_width_m <= 0 or args.row_offset_m <= 0:
        raise ValueError("Map, row distance and row tolerance must be positive")
    if not math.isfinite(args.camera_yaw_offset_deg):
        raise ValueError("Camera yaw offset must be finite")
    if args.range_manifest and not args.pose_csv:
        raise ValueError("Measured range requires matched camera poses via --pose-csv")
    range_evidence = RangeEvidence(args.range_manifest, args.sync_tolerance_s) if args.range_manifest else None
    input_dir = Path(args.input).expanduser().resolve()
    map_path = Path(args.map_path).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not map_path.exists():
        raise FileNotFoundError(f"Map image does not exist: {map_path}")
    all_images = list_images(input_dir)
    if not all_images:
        raise FileNotFoundError(f"No panoramic images found in: {input_dir}")
    selected = _sample(all_images, args.max_frames)
    camera_yaw_offset_rad = math.radians(args.camera_yaw_offset_deg)

    output_dir.mkdir(parents=True, exist_ok=True)
    view_dir = output_dir / "views"
    annotated_dir = output_dir / "annotated"
    view_dir.mkdir(exist_ok=True)
    annotated_dir.mkdir(exist_ok=True)

    map_image = Image.open(map_path).convert("RGB")
    detector = build_detector(
        args.detector,
        args.weights,
        args.confidence,
        detector_weights=args.detector_weights,
        classifier_weights=args.classifier_weights,
        brightness_gain=args.brightness_gain,
        detector_imgsz=args.detector_imgsz,
        classifier_imgsz=args.classifier_imgsz,
    )
    if args.pose_csv:
        pose_lookup = load_pose_csv(args.pose_csv, map_image, args.map_width_m)
    else:
        pose_lookup = generate_demo_poses(all_images, map_image, args.map_width_m)

    detections = []
    used_poses = []
    gallery: list[dict[str, object]] = []
    timings, rejected, candidate_audit = [], [], []
    taxonomy = getattr(detector, "taxonomy", "paper_four_stage")
    print(f"Found {len(all_images)} panoramas; processing {len(selected)} with detector: {detector.name}")
    for index, panorama_path in enumerate(selected, start=1):
        frame_start = time.perf_counter()
        pose = pose_lookup.get(panorama_path.name) or pose_lookup.get(panorama_path.stem)
        if pose is None:
            raise KeyError(f"No matching pose for frame: {panorama_path.name}")
        used_poses.append(pose)
        with Image.open(panorama_path) as panorama:
            sides = extract_side_views(panorama, args.face_size, camera_yaw_offset_rad=camera_yaw_offset_rad)
            if args.export_six_faces:
                for face, image in extract_all_faces(
                    panorama, args.face_size, camera_yaw_offset_rad=camera_yaw_offset_rad
                ).items():
                    image.save(view_dir / f"{_safe_stem(panorama_path)}_{face}.jpg", quality=93)
        projection_end = time.perf_counter()
        stem = _safe_stem(panorama_path)
        frame_count = 0
        for side, view in sides.items():
            view_name = f"{stem}_{side}.jpg"
            annotated_name = f"{stem}_{side}_det.jpg"
            view.save(view_dir / view_name, quality=93, subsampling=0)
            side_detections = detector.detect(view)
            candidate_audit.extend(dict(record, frame=panorama_path.name, side=side)
                                   for record in getattr(detector, "audit", []))
            relative_annotated = (Path("annotated") / annotated_name).as_posix()
            ranges = range_evidence.load(panorama_path.name, side, view.size) if range_evidence else None
            accepted = 0
            accepted_detections = []
            for detection in side_detections:
                measured_range = foreground_range(ranges, detection.bbox) if ranges is not None else None
                if ranges is not None and measured_range is None:
                    rejected.append(dict(frame=pose.frame, side=side, bbox=detection.bbox, reason="invalid_range"))
                    continue
                spatial = project_detection(
                        detection=detection,
                        side=side,
                        pose=pose,
                        view_size=view.size,
                        map_size=map_image.size,
                        map_width_m=args.map_width_m,
                        row_offset_m=args.row_offset_m,
                        camera_height_m=args.camera_height_m,
                        annotated_image=relative_annotated,
                        range_m=measured_range,
                    )
                if ranges is not None:
                    lateral = abs(-(spatial.x-pose.x)*np.sin(pose.yaw) + (spatial.y-pose.y)*np.cos(pose.yaw))
                    if abs(lateral-args.row_offset_m) > args.row_tolerance_m:
                        rejected.append(dict(frame=pose.frame, side=side, bbox=detection.bbox, reason="outside_nearest_row_band", lateral_m=float(lateral)))
                        continue
                detections.append(spatial)
                accepted_detections.append(detection)
                accepted += 1
            annotate(view, accepted_detections).save(annotated_dir / annotated_name, quality=94, subsampling=0)
            frame_count += accepted
            gallery.append(
                {"frame": panorama_path.name, "side": side, "count": accepted, "path": relative_annotated}
            )
        timings.append(dict(frame=pose.frame, projection_s=projection_end-frame_start,
                            detection_and_io_s=time.perf_counter()-projection_end,
                            total_s=time.perf_counter()-frame_start))
        print(f"[{index:02d}/{len(selected):02d}] {panorama_path.name}: {frame_count} candidates")

    pose_source = used_poses[0].source if used_poses else "unknown"
    tracks, links = associate(detections, used_poses, args.association_distance_m, args.association_gap)
    for name, payload in (("tracks.json", tracks), ("association_links.json", links), ("rejected_observations.json", rejected)):
        (output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "candidate_audit.json").write_text(json.dumps(candidate_audit, indent=2), encoding="utf-8")
    write_csv_files(output_dir, used_poses, detections, args.row_offset_m, taxonomy)
    draw_map_overlay(map_image, used_poses, detections, output_dir / "map_overlay.png", taxonomy)
    map_image.save(output_dir / "map_base.png", quality=95)
    summary = write_summary(output_dir, used_poses, detections, detector.name, pose_source, len(selected), taxonomy)
    summary.update(processing_mode="offline_batch", range_source="registered_measured_range" if range_evidence else "assumed_row_plane",
                   nearest_row_filter="measured_range_band" if range_evidence else "not_verified_no_depth",
                   association_status="spatial_candidates" if range_evidence else "disabled_no_measured_geometry",
                   candidate_tracks=sum(t["status"] == "association_candidate" for t in tracks), rejected_observations=len(rejected))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    evidence = build_evidence_manifest(
        summary, used_poses, processing_mode=summary["processing_mode"], range_source=summary["range_source"]
    )
    write_evidence_manifest(output_dir, evidence)
    map_height_m = args.map_width_m * map_image.height / map_image.width
    write_html_report(
        output_dir,
        summary,
        used_poses,
        detections,
        gallery,
        args.map_width_m,
        map_height_m,
        map_image.size,
        args.row_offset_m,
        evidence=evidence,
    )

    config = {
        "taxonomy": taxonomy,
        "classifier_imgsz": getattr(detector, "classifier_imgsz", None),
        "model_sha256": {key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                         for key, value in (("weights", args.weights), ("detector", args.detector_weights), ("classifier", args.classifier_weights)) if value},
        "environment": {"python": sys.version, **{package: importlib.metadata.version(package)
                        for package in ("numpy", "Pillow")}},
        "input": str(input_dir),
        "map": str(map_path),
        "pose_csv": str(Path(args.pose_csv).resolve()) if args.pose_csv else None,
        "weights": str(Path(args.weights).resolve()) if args.weights else None,
        "detector_weights": str(Path(args.detector_weights).resolve()) if args.detector_weights else None,
        "classifier_weights": str(Path(args.classifier_weights).resolve()) if args.classifier_weights else None,
        "brightness_gain": args.brightness_gain,
        "detector_imgsz": args.detector_imgsz,
        "face_size": args.face_size,
        "map_width_m": args.map_width_m,
        "row_offset_m": args.row_offset_m,
        "camera_height_m": args.camera_height_m,
        "camera_yaw_offset_deg": args.camera_yaw_offset_deg,
        "confidence": args.confidence,
        "max_frames": args.max_frames,
        "export_six_faces": args.export_six_faces,
        "range_manifest": args.range_manifest,
        "calibration_id": range_evidence.calibration_id if range_evidence else None,
        "sync_tolerance_s": args.sync_tolerance_s,
        "row_tolerance_m": args.row_tolerance_m,
        "association_distance_m": args.association_distance_m,
        "association_gap": args.association_gap,
    }
    for package in ("ultralytics", "torch", "torchvision"):
        try:
            config["environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.perf_counter()-started
    (output_dir / "runtime.json").write_text(json.dumps(dict(
        mode="offline_batch", total_s=elapsed, frames_per_second=len(selected)/elapsed,
        frame_timings=timings, includes_model_loading_and_io=True), indent=2), encoding="utf-8")
    counts = Counter(item.class_name for item in detections)
    print(f"Completed: {output_dir / 'index.html'}")
    from ripeness_demo.detectors import class_order_for
    print("Class counts: " + ", ".join(f"{name}={counts.get(name, 0)}" for name in class_order_for([item.class_name for item in detections], taxonomy)))
    return output_dir / "index.html"


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
