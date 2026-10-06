"""Replay calibrated depth arrays and export auditable navigation decisions."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ripeness_demo.depth_navigation import DepthConfig, DepthSupervisor, depth_to_points, assess_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON with geometry, intrinsics and optical_to_terrain")
    parser.add_argument("--frames", required=True, help="JSON list: depth_npy, timestamp_s, speed_mps, optional optical_to_terrain")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = DepthConfig(**spec["geometry"])
    supervisor = DepthSupervisor(cfg)
    manifest = Path(args.frames)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    decisions = []
    for index, record in enumerate(records):
        depth = np.load(manifest.parent / record["depth_npy"], allow_pickle=False)
        points = depth_to_points(depth, spec["intrinsics"],
                                 record.get("optical_to_terrain", spec.get("optical_to_terrain")),
                                 spec.get("encoding", "16UC1"), cfg.stride,
                                 cfg.min_depth_m, cfg.max_depth_m)
        assessment = assess_points(points, cfg)
        decision = supervisor.update(assessment, float(record["timestamp_s"]), float(record["speed_mps"]))
        np.save(output / f"obstacles_{index:06d}.npy", assessment["obstacle_points"])
        decisions.append(dict(frame=record["depth_npy"], timestamp_s=record["timestamp_s"], **decision))
    (output / "decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
