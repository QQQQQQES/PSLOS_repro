#!/usr/bin/env python3
"""Read-only preflight check for the physical-vehicle PS-LOS inputs."""

import argparse
import time

import rospy
from geometry_msgs.msg import PointStamped, TwistStamped
from mavros_msgs.msg import State
from sensor_msgs.msg import Imu


def collect(topic, message_type, duration_s):
    stamps = []
    messages = []

    def callback(message):
        messages.append(message)
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        stamps.append(stamp.to_sec() if stamp and not stamp.is_zero() else rospy.get_time())

    subscriber = rospy.Subscriber(topic, message_type, callback, queue_size=20)
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.02)
    subscriber.unregister()
    rate = 0.0
    if len(stamps) >= 2 and stamps[-1] > stamps[0]:
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    return messages, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mavros-ns", default="/mavros")
    parser.add_argument("--pixel-topic", default="/pixelstrike/target_pixel")
    parser.add_argument("--duration-s", type=float, default=3.0)
    args = parser.parse_args()
    rospy.init_node("pslos_real_input_check", anonymous=True)

    state = rospy.wait_for_message(args.mavros_ns + "/state", State, timeout=5.0)
    imu, imu_rate = collect(args.mavros_ns + "/imu/data", Imu, args.duration_s)
    velocity, velocity_rate = collect(
        args.mavros_ns + "/local_position/velocity_local", TwistStamped, args.duration_s
    )
    pixels, pixel_rate = collect(args.pixel_topic, PointStamped, args.duration_s)
    valid_pixels = [p for p in pixels if p.point.z > 0.0]

    print("FCU connected={} armed={} mode={}".format(state.connected, state.armed, state.mode))
    print("IMU {:.1f} Hz ({} messages)".format(imu_rate, len(imu)))
    print("velocity {:.1f} Hz ({} messages)".format(velocity_rate, len(velocity)))
    print(
        "pixel {:.1f} Hz ({} messages, {} valid)".format(
            pixel_rate, len(pixels), len(valid_pixels)
        )
    )
    ok = bool(state.connected and imu_rate >= 50.0 and velocity and pixel_rate >= 15.0)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
