#!/usr/bin/env python3
"""Export public, image-free timestamp/RTK pose evidence from a ROS1 bag.

This utility deliberately reads only a camera-trigger topic and a synchronized
RTK topic. It never subscribes to, decodes, copies, or writes image topics.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any


EARTH_RADIUS_M = 6_378_137.0


def local_enu_m(latitude: float, longitude: float, height: float, origin: tuple[float, float, float]) -> tuple[float, float, float]:
    """Approximate local east, north, up coordinates from geodetic RTK samples."""
    origin_latitude, origin_longitude, origin_height = origin
    east = math.radians(longitude - origin_longitude) * EARTH_RADIUS_M * math.cos(math.radians(origin_latitude))
    north = math.radians(latitude - origin_latitude) * EARTH_RADIUS_M
    return east, north, height - origin_height


def build_pose_rows(
    trigger_times: list[tuple[float, int]],
    rtk_samples: list[dict[str, float | int]],
    max_delta_s: float,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Assign every trigger its unique nearest bag-time RTK sample, or reject it."""
    if not rtk_samples:
        raise ValueError("No valid RTK samples were supplied")
    if not math.isfinite(max_delta_s) or max_delta_s <= 0:
        raise ValueError("max_delta_s must be finite and positive")
    rtk_samples = sorted(rtk_samples, key=lambda sample: float(sample["bag_time_s"]))
    pose_times = [float(sample["bag_time_s"]) for sample in rtk_samples]
    origin_sample = rtk_samples[0]
    origin = (float(origin_sample["latitude"]), float(origin_sample["longitude"]), float(origin_sample["height_m"]))
    rows: list[dict[str, float | int | str]] = []
    rejected: list[dict[str, float | int | str]] = []
    last_yaw = 0.0

    for index, (trigger_time_s, trigger_seq) in enumerate(sorted(trigger_times), start=1):
        insertion = bisect.bisect_left(pose_times, trigger_time_s)
        candidate_indices = [candidate for candidate in (insertion - 1, insertion) if 0 <= candidate < len(rtk_samples)]
        nearest_index = min(candidate_indices, key=lambda candidate: abs(pose_times[candidate] - trigger_time_s))
        sample = rtk_samples[nearest_index]
        delta_s = abs(float(sample["bag_time_s"]) - trigger_time_s)
        if delta_s > max_delta_s:
            rejected.append({
                "shot_index": index,
                "trigger_seq": trigger_seq,
                "trigger_time_s": trigger_time_s,
                "nearest_pose_time_s": float(sample["bag_time_s"]),
                "match_delta_s": delta_s,
                "reason": "nearest_rtk_exceeds_tolerance",
            })
            continue
        east_m, north_m, up_m = local_enu_m(
            float(sample["latitude"]), float(sample["longitude"]), float(sample["height_m"]), origin
        )
        velocity_east = float(sample["velocity_east_mps"])
        velocity_north = float(sample["velocity_north_mps"])
        speed_mps = math.hypot(velocity_east, velocity_north)
        if speed_mps >= 0.02:
            last_yaw = math.atan2(velocity_north, velocity_east)
        rows.append({
            "shot_index": index,
            "trigger_seq": trigger_seq,
            "trigger_time_s": trigger_time_s,
            "pose_time_s": float(sample["bag_time_s"]),
            "match_delta_s": delta_s,
            "x_east_m": east_m,
            "y_north_m": north_m,
            "z_relative_m": up_m,
            "yaw_rad_from_velocity": last_yaw,
            "speed_mps": speed_mps,
            "rtk_solution_type": int(sample["solution_type"]),
        })
    return rows, rejected


def read_rosbag(bag_path: Path, trigger_topic: str, rtk_topic: str) -> tuple[list[tuple[float, int]], list[dict[str, float | int]]]:
    """Read only trigger and RTK messages, using ROS bag record times as the common clock."""
    try:
        import rosbag  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - requires ROS1 runtime
        raise RuntimeError("This exporter requires a ROS1 environment with rosbag available") from exc
    triggers: list[tuple[float, int]] = []
    rtk_samples: list[dict[str, float | int]] = []
    with rosbag.Bag(str(bag_path), "r") as bag:  # pragma: no cover - requires ROS1 runtime
        for topic, message, bag_time in bag.read_messages(topics=[trigger_topic, rtk_topic]):
            timestamp_s = bag_time.to_sec()
            if topic == trigger_topic:
                triggers.append((timestamp_s, int(message.header.seq)))
            else:
                values = (
                    float(message.bestpos_lat), float(message.bestpos_lon), float(message.bestpos_hgt),
                    float(message.psrvel_east), float(message.psrvel_north),
                )
                if all(math.isfinite(value) for value in values):
                    rtk_samples.append({
                        "bag_time_s": timestamp_s,
                        "latitude": values[0],
                        "longitude": values[1],
                        "height_m": values[2],
                        "velocity_east_mps": values[3],
                        "velocity_north_mps": values[4],
                        "solution_type": int(message.bestpos_type.type),
                    })
    return triggers, rtk_samples


def write_evidence(output_dir: Path, rows: list[dict[str, Any]], rejected: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "shot_index", "trigger_seq", "trigger_time_s", "pose_time_s", "match_delta_s", "x_east_m", "y_north_m",
        "z_relative_m", "yaw_rad_from_velocity", "speed_mps", "rtk_solution_type",
    ]
    with (output_dir / "trigger_pose_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "rejected_triggers.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(rejected[0]) if rejected else ["shot_index", "trigger_seq", "trigger_time_s", "nearest_pose_time_s", "match_delta_s", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rejected)
    # These two files use opaque frame names and contain no panorama bytes. They
    # make the recorded timing stream directly usable as a PTRDMS interface test.
    with (output_dir / "ptrdms_frame_times.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "timestamp_s"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"frame": f"shot_{int(row['shot_index']):04d}.jpg", "timestamp_s": row["trigger_time_s"]})
    with (output_dir / "ptrdms_pose_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_s", "x", "y", "z", "yaw", "route"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "timestamp_s": row["pose_time_s"], "x": row["x_east_m"], "y": row["y_north_m"],
                "z": row["z_relative_m"], "yaw": row["yaw_rad_from_velocity"], "route": "test_bag_path",
            })
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export image-free trigger/RTK timestamp evidence from a ROS1 bag")
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trigger-topic", default="/camera_agent/shot_trigger")
    parser.add_argument("--rtk-topic", default="/rtk_agent/pvtsln_sync")
    parser.add_argument("--max-delta-s", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    triggers, rtk_samples = read_rosbag(args.bag, args.trigger_topic, args.rtk_topic)
    rows, rejected = build_pose_rows(triggers, rtk_samples, args.max_delta_s)
    deltas = sorted(float(row["match_delta_s"]) for row in rows)
    percentile = lambda fraction: deltas[round((len(deltas) - 1) * fraction)] if deltas else None
    metadata = {
        "schema_version": 1,
        "source_bag_filename": args.bag.name,
        "trigger_topic": args.trigger_topic,
        "rtk_topic": args.rtk_topic,
        "time_basis": "rosbag_record_time_s",
        "matching": "unique nearest recorded RTK sample for each camera trigger",
        "maximum_allowed_delta_s": args.max_delta_s,
        "trigger_messages": len(triggers),
        "valid_rtk_messages": len(rtk_samples),
        "matched_triggers": len(rows),
        "rejected_triggers": len(rejected),
        "match_delta_statistics_s": {
            "minimum": percentile(0.0), "median": percentile(0.5),
            "p95": percentile(0.95), "maximum": percentile(1.0),
        },
        "image_topics_read": False,
        "image_data_exported": False,
        "ptrdms_fixture": "ptrdms_frame_times.csv and ptrdms_pose_samples.csv use opaque shot_NNNN.jpg identifiers only",
        "coordinate_frame": "local ENU approximation, origin at first valid RTK sample",
        "yaw_note": "yaw_rad_from_velocity is derived from horizontal RTK velocity and is held at the last valid value below 0.02 m/s",
    }
    write_evidence(args.output, rows, rejected, metadata)
    if rejected:
        raise SystemExit(f"Rejected {len(rejected)} trigger(s); inspect {args.output / 'rejected_triggers.csv'}")
    print(f"Exported {len(rows)} image-free trigger/RTK matches to {args.output}")


if __name__ == "__main__":
    main()
