#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import PointStamped

from pslos_msgs.msg import TargetFeature


class FeatureAdapterNode:
    def __init__(self):
        self.width = int(rospy.get_param("~image_width", 640))
        self.height = int(rospy.get_param("~image_height", 480))
        self.fx = float(rospy.get_param("~fx", 184.752086))
        self.fy = float(rospy.get_param("~fy", 184.752086))
        self.cx = float(rospy.get_param("~cx", 320.0))
        self.cy = float(rospy.get_param("~cy", 240.0))
        self.minimum_confidence = float(rospy.get_param("~minimum_confidence", 0.1))
        if min(self.fx, self.fy) <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        self.publisher = rospy.Publisher("feature", TargetFeature, queue_size=10)
        self.subscriber = rospy.Subscriber(
            "external_pixel", PointStamped, self.pixel_callback, queue_size=10
        )
        rospy.loginfo(
            "PS-LOS feature adapter: %dx%d fx=%.3f fy=%.3f, no depth input",
            self.width,
            self.height,
            self.fx,
            self.fy,
        )

    def pixel_callback(self, pixel):
        u = float(pixel.point.x)
        v = float(pixel.point.y)
        confidence = float(pixel.point.z)
        finite = all(math.isfinite(value) for value in (u, v, confidence))
        in_image = 0.0 <= u < self.width and 0.0 <= v < self.height

        feature = TargetFeature()
        feature.header = pixel.header
        if feature.header.stamp.is_zero():
            rospy.logwarn_throttle(2.0, "External feature has no acquisition timestamp")
        feature.header.frame_id = "camera_optical"
        feature.u = u
        feature.v = v
        feature.confidence = confidence
        feature.detected = (
            not feature.header.stamp.is_zero()
            and finite
            and in_image
            and confidence >= self.minimum_confidence
        )
        if feature.detected:
            feature.x_normalized = (u - self.cx) / self.fx
            feature.y_normalized = (v - self.cy) / self.fy
        self.publisher.publish(feature)


def main():
    rospy.init_node("pslos_feature_adapter")
    FeatureAdapterNode()
    rospy.spin()


if __name__ == "__main__":
    main()
