"""Run PTRDMS against live ROS1 panorama and localization topics.

This optional entry point keeps ROS imports inside :func:`main`, so the
offline batch workflow remains usable without ROS installed.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import threading
from pathlib import Path

from PIL import Image

from ripeness_demo.artifacts import build_evidence_manifest, write_evidence_manifest
from ripeness_demo.detectors import annotate, build_detector
from ripeness_demo.mapping import project_detection
from ripeness_demo.online import TimedPoseBuffer
from ripeness_demo.panorama import extract_side_views
from ripeness_demo.report import draw_map_overlay, write_csv_files, write_html_report, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTRDMS online ROS1 inference")
    parser.add_argument("--map", dest="map_path", required=True, help="Metric map image")
    parser.add_argument("--output", default="outputs/online", help="Online result directory")
    parser.add_argument("--image-topic", default="/camera/image/compressed")
    parser.add_argument("--pose-topic", default="/robot_pose")
    parser.add_argument("--result-topic", default="/ptrdms/observations", help="JSON observation topic")
    parser.add_argument("--route", default="online", help="Session or row identifier written with poses")
    parser.add_argument("--pose-tolerance-s", type=float, default=0.05)
    parser.add_argument("--max-pose-buffer", type=int, default=2000)
    parser.add_argument("--report-every", type=int, default=10, help="Refresh index.html after this many accepted frames")
    parser.add_argument("--map-width-m", type=float, default=35.0)
    parser.add_argument("--row-offset-m", type=float, default=1.20)
    parser.add_argument("--camera-height-m", type=float, default=1.15)
    parser.add_argument("--camera-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--face-size", type=int, default=720)
    parser.add_argument("--detector", choices=["auto", "yolo", "two-stage", "color"], default="auto")
    parser.add_argument("--weights")
    parser.add_argument("--detector-weights")
    parser.add_argument("--classifier-weights")
    parser.add_argument("--detector-imgsz", type=int, default=1280)
    parser.add_argument("--classifier-imgsz", type=int)
    return parser.parse_args()


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OnlineResultStore:
    """Persist rolling online results without retaining raw panorama frames."""

    def __init__(self, output_dir: Path, map_image: Image.Image, args: argparse.Namespace, detector_name: str, taxonomy: str):
        self.output_dir = output_dir
        self.map_image = map_image
        self.args = args
        self.detector_name = detector_name
        self.taxonomy = taxonomy
        self.poses = []
        self.detections = []
        self.gallery = []
        self.accepted_frames = 0
        self.rejected_frames = 0
        self._lock = threading.Lock()
        (output_dir / "annotated").mkdir(parents=True, exist_ok=True)
        (output_dir / "views").mkdir(parents=True, exist_ok=True)

    def reject_frame(self, frame_name: str, timestamp_s: float, reason: str) -> None:
        with self._lock:
            self.rejected_frames += 1
            payload = {"frame": frame_name, "timestamp_s": timestamp_s, "reason": reason}
            with (self.output_dir / "rejected_frames.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    def record(self, pose, side_results: list[tuple[str, Image.Image, list, list]]) -> None:
        with self._lock:
            self.accepted_frames += 1
            self.poses.append(pose)
            for side, view, raw_detections, spatial in side_results:
                stem = Path(pose.frame).stem
                view_name = f"{stem}_{side}.jpg"
                annotated_name = f"{stem}_{side}_det.jpg"
                view.save(self.output_dir / "views" / view_name, quality=93, subsampling=0)
                annotate(view, raw_detections).save(
                    self.output_dir / "annotated" / annotated_name, quality=94, subsampling=0
                )
                self.detections.extend(spatial)
                self.gallery.append({"frame": pose.frame, "side": side, "count": len(spatial), "path": (Path("annotated") / annotated_name).as_posix()})
            self._write_artifacts()

    def _write_artifacts(self) -> None:
        write_csv_files(self.output_dir, self.poses, self.detections, self.args.row_offset_m, self.taxonomy)
        draw_map_overlay(self.map_image, self.poses, self.detections, self.output_dir / "map_overlay.png", self.taxonomy)
        summary = write_summary(
            self.output_dir, self.poses, self.detections, self.detector_name, "ros_pose", self.accepted_frames, self.taxonomy
        )
        summary.update(
            processing_mode="online_ros",
            rejected_frames=self.rejected_frames,
            pose_matching="unique_nearest_timestamp",
            pose_tolerance_s=self.args.pose_tolerance_s,
        )
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        evidence = build_evidence_manifest(summary, self.poses, processing_mode="online_ros", range_source="assumed_row_plane")
        write_evidence_manifest(self.output_dir, evidence)
        if self.accepted_frames == 1 or self.accepted_frames % self.args.report_every == 0:
            width, height = self.map_image.size
            write_html_report(
                self.output_dir, summary, self.poses, self.detections, self.gallery,
                self.args.map_width_m, self.args.map_width_m * height / width,
                self.map_image.size, self.args.row_offset_m, evidence,
            )


class RosOnlineNode:
    def __init__(self, args: argparse.Namespace, rospy, result_publisher):
        self.args = args
        self.rospy = rospy
        self.result_publisher = result_publisher
        self.map_image = Image.open(args.map_path).convert("RGB")
        self.output_dir = Path(args.output).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector = build_detector(
            args.detector, args.weights, args.confidence,
            detector_weights=args.detector_weights, classifier_weights=args.classifier_weights,
            detector_imgsz=args.detector_imgsz, classifier_imgsz=args.classifier_imgsz,
        )
        self.taxonomy = getattr(self.detector, "taxonomy", "paper_four_stage")
        self.pose_buffer = TimedPoseBuffer(self.map_image.size, args.map_width_m, args.max_pose_buffer)
        self.results = OnlineResultStore(self.output_dir, self.map_image, args, self.detector.name, self.taxonomy)
        self.camera_yaw_offset_rad = math.radians(args.camera_yaw_offset_deg)

    @staticmethod
    def _stamp(header) -> tuple[int, int]:
        return int(header.stamp.secs), int(header.stamp.nsecs)

    def on_pose(self, message) -> None:
        seconds, nanoseconds = self._stamp(message.header)
        position = message.pose.position
        self.pose_buffer.add(
            seconds=seconds, nanoseconds=nanoseconds, frame_id=message.header.frame_id,
            x=position.x, y=position.y, z=position.z, yaw=_yaw_from_quaternion(message.pose.orientation), route=self.args.route,
        )

    def on_image(self, message) -> None:
        seconds, nanoseconds = self._stamp(message.header)
        frame_name = f"ros_{seconds}_{nanoseconds:09d}.jpg"
        pose = self.pose_buffer.match(
            seconds=seconds, nanoseconds=nanoseconds, frame_name=frame_name,
            tolerance_s=self.args.pose_tolerance_s,
        )
        if pose is None:
            self.results.reject_frame(frame_name, seconds + nanoseconds / 1_000_000_000, "no_unique_pose_within_tolerance")
            self.rospy.logwarn_throttle(5.0, "PTRDMS skipped panorama: no unique timestamp-matched pose")
            return
        try:
            panorama = Image.open(io.BytesIO(bytes(message.data))).convert("RGB")
            sides = extract_side_views(panorama, self.args.face_size, camera_yaw_offset_rad=self.camera_yaw_offset_rad)
            side_results = []
            for side, view in sides.items():
                raw = self.detector.detect(view)
                spatial = [
                    project_detection(
                        detection, side, pose, view.size, self.map_image.size,
                        map_width_m=self.args.map_width_m, row_offset_m=self.args.row_offset_m,
                        camera_height_m=self.args.camera_height_m,
                        annotated_image=(Path("annotated") / f"{Path(frame_name).stem}_{side}_det.jpg").as_posix(),
                    )
                    for detection in raw
                ]
                side_results.append((side, view, raw, spatial))
            self.results.record(pose, side_results)
            self.result_publisher.publish(json.dumps({
                "frame": frame_name,
                "frame_timestamp_s": pose.frame_timestamp_s,
                "pose_timestamp_s": pose.timestamp_s,
                "match_delta_s": pose.match_delta_s,
                "pose": {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw, "route": pose.route},
                "observations": [item.to_dict() for _, _, _, spatial in side_results for item in spatial],
            }, ensure_ascii=False))
            self.rospy.loginfo(f"PTRDMS processed {frame_name}: {sum(len(item[3]) for item in side_results)} observations")
        except Exception as exc:  # pragma: no cover - ROS callback safety
            self.results.reject_frame(frame_name, seconds + nanoseconds / 1_000_000_000, f"processing_error: {exc}")
            self.rospy.logerr(f"PTRDMS online frame failed: {exc}")


def main() -> None:
    args = parse_args()
    if args.report_every < 1 or args.pose_tolerance_s <= 0 or args.map_width_m <= 0:
        raise ValueError("report interval, pose tolerance, and map width must be positive")
    try:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
    except ImportError as exc:
        raise RuntimeError("Online mode requires a ROS1 environment with rospy, geometry_msgs, and sensor_msgs") from exc
    rospy.init_node("ptrdms_online", anonymous=True)
    result_publisher = rospy.Publisher(args.result_topic, String, queue_size=10)
    node = RosOnlineNode(args, rospy, result_publisher)
    rospy.Subscriber(args.pose_topic, PoseStamped, node.on_pose, queue_size=200)
    rospy.Subscriber(args.image_topic, CompressedImage, node.on_image, queue_size=2)
    rospy.loginfo(f"PTRDMS online mode listening on image={args.image_topic}, pose={args.pose_topic}, result={args.result_topic}")
    rospy.spin()


if __name__ == "__main__":
    main()
