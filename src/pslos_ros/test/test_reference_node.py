#!/usr/bin/env python3
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import rospy

from pslos_msgs.msg import RelativeState, TargetFeature


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reference_node.py"
SPEC = importlib.util.spec_from_file_location("reference_node", str(SCRIPT))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ReferenceNode = MODULE.ReferenceNode


class CapturePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def vector_namespace(values):
    return SimpleNamespace(x=values[0], y=values[1], z=values[2])


def control_result(x_normalized, y_normalized):
    ray_norm = np.linalg.norm([x_normalized, y_normalized, 1.0])
    fov = SimpleNamespace(
        s_long=y_normalized / ray_norm,
        s_tight=x_normalized / ray_norm,
        z1=abs(y_normalized / ray_norm),
        z2=x_normalized / ray_norm,
        c_long=0.8,
        c_tight=0.02,
        margin_long=0.7,
        margin_tight=0.01,
        l1=0.1,
        l2=0.1,
        k_h=0.1,
        k_v=0.1,
        fov_tangent=np.zeros(3),
        omega_los_body=np.zeros(3),
        formula_mode="paper_literal",
    )
    return SimpleNamespace(
        selected_force_world=np.array([0.0, 0.0, 9.80665]),
        desired_attitude=np.eye(3),
        desired_body_rate=np.zeros(3),
        velocity_error=np.zeros(3),
        fov=fov,
        force_law_mode="paper_equation_15",
    )


class TestCurrentPixelFOVAudit(unittest.TestCase):
    def setUp(self):
        self.node = ReferenceNode.__new__(ReferenceNode)
        self.node.max_state_age = 0.1
        self.node.max_feature_age = 0.12
        self.node.capture_radius = 0.3
        self.node.require_current_pixel_los = True
        self.node.use_current_los_for_position_direction = False
        self.node.gravity_force = np.array([0.0, 0.0, -9.80665])
        self.node.aerodynamic_force = np.zeros(3)
        self.node.reference_publisher = CapturePublisher()
        self.node.debug_publisher = CapturePublisher()
        self.node.current_pixel_fov_debug_publisher = CapturePublisher()

        state = RelativeState()
        state.header.stamp = rospy.Time.from_sec(9.99)
        state.valid = True
        state.range = 3.0
        state.position.x = 3.0
        state.los.z = 1.0
        state.interceptor_attitude.w = 1.0
        self.node.state = state

    @staticmethod
    def feature(stamp, x_normalized, y_normalized):
        feature = TargetFeature()
        feature.header.stamp = rospy.Time.from_sec(stamp)
        feature.header.frame_id = "camera_optical"
        feature.detected = True
        feature.x_normalized = x_normalized
        feature.y_normalized = y_normalized
        return feature

    def test_audit_records_the_exact_feature_snapshot_used_by_controller(self):
        self.node.use_current_los_for_position_direction = True
        consumed = self.feature(9.98, 0.01, -0.02)
        arriving_during_cycle = self.feature(9.99, 0.20, 0.30)
        self.node.feature = consumed
        result = control_result(consumed.x_normalized, consumed.y_normalized)
        self.node.controller = SimpleNamespace(
            rotation_body_camera=np.eye(3),
            parameters=SimpleNamespace(mass=1.0),
            formula_mode="paper_literal",
            force_law_mode="paper_equation_15",
        )

        def consume_feature(feature, *_args):
            self.assertIs(feature, consumed)
            self.node.feature = arriving_during_cycle
            return np.array([0.0, 0.0, 1.0]), 0.02

        self.node.controller.compute = mock.Mock(return_value=result)
        with mock.patch.object(MODULE.rospy, "is_shutdown", return_value=False), mock.patch.object(
            MODULE.rospy.Time, "now", return_value=rospy.Time.from_sec(10.0)
        ), mock.patch.object(
            MODULE, "current_pixel_los_world", side_effect=consume_feature
        ):
            self.node.timer_callback(None)

        self.assertIs(self.node.feature, arriving_during_cycle)
        self.assertEqual(len(self.node.debug_publisher.messages), 1)
        self.assertEqual(len(self.node.current_pixel_fov_debug_publisher.messages), 1)
        debug = self.node.debug_publisher.messages[0]
        audit = self.node.current_pixel_fov_debug_publisher.messages[0]
        self.assertEqual(audit.header.stamp, debug.header.stamp)
        self.assertEqual(audit.feature_stamp, consumed.header.stamp)
        self.assertAlmostEqual(audit.feature_age, 0.02)
        self.assertEqual(audit.feature_frame_id, "camera_optical")
        self.assertAlmostEqual(audit.x_normalized, consumed.x_normalized)
        self.assertAlmostEqual(audit.y_normalized, consumed.y_normalized)
        self.assertAlmostEqual(audit.s_long, debug.s_long)
        self.assertAlmostEqual(audit.s_tight, debug.s_tight)
        self.assertTrue(audit.feature_valid)
        self.assertEqual(audit.reason, "accepted")
        relative_position = self.node.controller.compute.call_args.args[0]
        np.testing.assert_allclose(relative_position, [0.0, 0.0, -3.0], atol=1e-12)

    def test_rejected_feature_has_same_cycle_stamp_and_failure_reason(self):
        rejected = self.feature(9.0, 0.01, 0.02)
        self.node.feature = rejected
        self.node.controller = SimpleNamespace(
            rotation_body_camera=np.eye(3),
            formula_mode="paper_literal",
            force_law_mode="paper_equation_15",
        )
        with mock.patch.object(MODULE.rospy, "is_shutdown", return_value=False), mock.patch.object(
            MODULE.rospy.Time, "now", return_value=rospy.Time.from_sec(10.0)
        ):
            self.node.timer_callback(None)

        self.assertEqual(len(self.node.debug_publisher.messages), 1)
        self.assertEqual(len(self.node.current_pixel_fov_debug_publisher.messages), 1)
        debug = self.node.debug_publisher.messages[0]
        audit = self.node.current_pixel_fov_debug_publisher.messages[0]
        self.assertEqual(audit.header.stamp, debug.header.stamp)
        self.assertFalse(audit.feature_valid)
        self.assertEqual(audit.reason, "feature_stale")
        self.assertAlmostEqual(audit.feature_age, 1.0)


if __name__ == "__main__":
    unittest.main()
