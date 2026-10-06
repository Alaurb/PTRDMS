#!/usr/bin/env python3
"""D435i ROS1 near-field perception, costmap observations and velocity gate."""
import json
import math
import threading
import numpy as np
from ripeness_demo.depth_navigation import DepthConfig, DepthSupervisor, depth_to_points, assess_points, gate_velocity


def main():
    import rospy
    import tf2_ros
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2
    from sensor_msgs import point_cloud2
    from std_msgs.msg import Header, String
    from tf.transformations import quaternion_matrix

    rospy.init_node("ptrdms_depth_navigation")
    cfg = DepthConfig(**rospy.get_param("~geometry"))
    frame = rospy.get_param("~terrain_frame", "base_stabilized")
    nav_topic = rospy.get_param("~nav_cmd_topic", "/move_base/cmd_vel")
    output_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
    if rospy.resolve_name(nav_topic) == rospy.resolve_name(output_topic):
        raise ValueError("Navigation input and gated output topics must differ")
    turn_limit = float(rospy.get_param("~max_turn_radps", 0.3))
    if not math.isfinite(turn_limit) or turn_limit <= 0:
        raise ValueError("max_turn_radps must be positive")
    lock = threading.RLock()
    supervisor = DepthSupervisor(cfg)
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)
    bridge = CvBridge()
    storage = dict(info=None, nav=Twist(), nav_time=None, speed=0., odom_time=None,
                   gait=None, gait_time=None)
    obstacle_pub = rospy.Publisher("~obstacles", PointCloud2, queue_size=1)
    clearing_pub = rospy.Publisher("~clearing", PointCloud2, queue_size=1)
    state_pub = rospy.Publisher("~state", String, queue_size=1)
    gait_pub = rospy.Publisher("~gait_request", String, queue_size=1, latch=True)
    velocity_pub = rospy.Publisher(output_topic, Twist, queue_size=1)

    def info_callback(msg):
        with lock:
            storage["info"] = msg

    def nav_callback(msg):
        values = (msg.linear.x, msg.linear.y, msg.linear.z,
                  msg.angular.x, msg.angular.y, msg.angular.z)
        with lock:
            storage["nav"] = msg if all(math.isfinite(v) for v in values) else Twist()
            storage["nav_time"] = rospy.Time.now().to_sec()

    def odom_callback(msg):
        with lock:
            speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
            if math.isfinite(speed):
                storage["speed"] = speed
                storage["odom_time"] = msg.header.stamp.to_sec()

    def gait_callback(msg):
        with lock:
            if msg.data in ("normal", "terrain"):
                storage["gait"] = msg.data
                storage["gait_time"] = rospy.Time.now().to_sec()

    def depth_callback(msg):
        try:
            with lock:
                info = storage["info"]
                speed = max(storage["speed"], math.hypot(storage["nav"].linear.x, storage["nav"].linear.y))
            if info is None:
                return
            if info.header.frame_id != msg.header.frame_id or (info.width, info.height) != (msg.width, msg.height):
                raise ValueError("Depth image requires its own matching CameraInfo")
            # P describes the rectified image projection, not the distorted raw image.
            k = (info.P[0], info.P[5], info.P[2], info.P[6])
            if abs(info.P[3]) > 1e-8 or abs(info.P[7]) > 1e-8:
                raise ValueError("Use the depth-camera rectified projection with zero stereo translation")
            stamp = msg.header.stamp
            now = rospy.Time.now().to_sec()
            if stamp.to_sec() <= 0 or now - stamp.to_sec() > cfg.stale_s or stamp.to_sec() > now + .05:
                raise ValueError("Depth timestamp is missing, stale or ahead of ROS time")
            tf = tf_buffer.lookup_transform(frame, msg.header.frame_id, stamp, rospy.Duration(.05))
            q = tf.transform.rotation
            transform = quaternion_matrix([q.x, q.y, q.z, q.w])
            p = tf.transform.translation
            transform[:3, 3] = [p.x, p.y, p.z]
            depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            points = depth_to_points(depth, k, transform, msg.encoding, cfg.stride,
                                     cfg.min_depth_m, cfg.max_depth_m)
            result = assess_points(points, cfg)
            with lock:
                decision = supervisor.update(result, stamp.to_sec(), speed)
            header = Header(stamp=stamp, frame_id=frame)
            obstacle_pub.publish(point_cloud2.create_cloud_xyz32(header, result["obstacle_points"].tolist()))
            # Clearing rays start at the physical optical origin via sensor_frame.
            # Ground and obstacle returns are included; invalid pixels create no rays.
            clearing_pub.publish(point_cloud2.create_cloud_xyz32(header, points.tolist()))
            state_pub.publish(String(data=json.dumps(decision)))
            gait_pub.publish(String(data=decision["gait_mode"]))
        except Exception as exc:
            # Revoke the previous permit immediately on a bad image/transform.
            with lock:
                supervisor.last = None
            rospy.logwarn_throttle(2., "Depth navigation: %s", exc)

    def tick(event):
        with lock:
            now = rospy.Time.now().to_sec()
            decision = supervisor.command(now)
            nav = storage["nav"]
            fresh = all(storage[key] is not None and 0 <= now - storage[key] <= cfg.stale_s
                        for key in ("nav_time", "odom_time"))
            acknowledged = (storage["gait"] == decision["gait_mode"]
                            and storage["gait_time"] is not None
                            and 0 <= now - storage["gait_time"] <= cfg.stale_s)
            command = Twist()
            # This forward depth corridor does not certify reverse/lateral travel
            # or the swept volume of an in-place rotation.
            command.linear.x, command.angular.z = gate_velocity(
                nav.linear.x, nav.linear.y, nav.angular.z, decision, storage["speed"],
                cfg, fresh, acknowledged, turn_limit)
            velocity_pub.publish(command)
            gait_pub.publish(String(data=decision["gait_mode"]))

    rospy.Subscriber(rospy.get_param("~camera_info_topic", "/camera/depth/camera_info"), CameraInfo, info_callback, queue_size=1)
    rospy.Subscriber(rospy.get_param("~depth_topic", "/camera/depth/image_rect_raw"), Image, depth_callback, queue_size=1, buff_size=2**24)
    rospy.Subscriber(nav_topic, Twist, nav_callback, queue_size=1)
    rospy.Subscriber(rospy.get_param("~odom_topic", "/odom"), Odometry, odom_callback, queue_size=1)
    rospy.Subscriber(rospy.get_param("~gait_state_topic", "/robot/gait_state"), String, gait_callback, queue_size=1)
    timer = rospy.Timer(rospy.Duration(.05), tick)
    rospy.on_shutdown(lambda: velocity_pub.publish(Twist()))
    rospy.spin()


if __name__ == "__main__":
    main()
