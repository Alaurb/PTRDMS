#!/usr/bin/env python3
"""Bridge supervisory normal/terrain requests to a firmware-matched Unitree SDK."""
import threading
import time
from ripeness_demo.unitree_gait import UnitreeGaitAdapter


def main():
    import rospy
    from std_msgs.msg import String
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    rospy.init_node("ptrdms_unitree_gait")
    ChannelFactoryInitialize(0, rospy.get_param("~network_interface"))
    client = SportClient()
    client.SetTimeout(.5)
    client.Init()
    adapter = UnitreeGaitAdapter(client, rospy.get_param("~normal_gait_id"), rospy.get_param("~terrain_gait_id"))
    lock = threading.RLock()
    storage = dict(request=None, request_time=None, actual=None, actual_time=None,
                   attempt_time=0., requested_mode=None)
    state_pub = rospy.Publisher(rospy.get_param("~state_topic", "/robot/gait_state"), String, queue_size=1)

    def request_callback(msg):
        if msg.data in adapter.ids:
            with lock:
                storage["request"] = msg.data
                storage["request_time"] = time.monotonic()

    def state_callback(msg):
        with lock:
            storage["actual"] = next((name for name, value in adapter.ids.items() if value == msg.gait_type), None)
            storage["actual_time"] = time.monotonic()

    def tick(event):
        with lock:
            now = time.monotonic()
            request = storage["request"]
            fresh = storage["request_time"] is not None and now - storage["request_time"] <= .5
            actual_fresh = storage["actual_time"] is not None and now - storage["actual_time"] <= .5
            actual = storage["actual"] if actual_fresh else None
            if actual is not None:
                state_pub.publish(String(data=actual))
            retry = now - storage["attempt_time"] >= 1.
            needed = actual != request or storage["requested_mode"] != request
            if not (fresh and request is not None and needed and retry):
                return
            adapter.confirmed = actual
            storage["attempt_time"] = now
        # Do not hold the feedback lock during a blocking SDK request.
        if adapter.request(request):
            with lock:
                storage["requested_mode"] = request
        else:
            rospy.logwarn_throttle(2., "Unitree gait request was rejected")

    state_sub = ChannelSubscriber(rospy.get_param("~sport_state_topic", "rt/lf/sportmodestate"), SportModeState_)
    state_sub.Init(state_callback, 10)
    rospy.Subscriber(rospy.get_param("~request_topic", "/ptrdms_depth_navigation/gait_request"), String, request_callback, queue_size=1)
    timer = rospy.Timer(rospy.Duration(.1), tick)
    rospy.spin()


if __name__ == "__main__":
    main()
