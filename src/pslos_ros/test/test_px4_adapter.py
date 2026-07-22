#!/usr/bin/env python3
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "px4_adapter_node.py"
SPEC = importlib.util.spec_from_file_location("px4_adapter_node", str(SCRIPT))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ControlEnableLease = MODULE.ControlEnableLease
PX4AdapterNode = MODULE.PX4AdapterNode
vertical_visibility_bounds = MODULE.vertical_visibility_bounds
attitude_alignment_thrust_scale = MODULE.attitude_alignment_thrust_scale


class TestRealVehicleSafetyLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        launch_path = SCRIPT.parents[1] / "launch" / "real_intercept.launch"
        cls.root = ET.parse(str(launch_path)).getroot()
        cls.config = (SCRIPT.parents[1] / "config" / "px4_real.yaml").read_text(
            encoding="ascii"
        )

    def test_output_is_disabled_by_default_and_authorization_is_required(self):
        arguments = {
            item.attrib["name"]: item.attrib.get("default")
            for item in self.root.findall("arg")
        }
        self.assertEqual(arguments["enable_px4_output"], "false")
        adapter = next(
            node
            for node in self.root.findall("node")
            if node.attrib.get("name") == "pslos_px4_adapter"
        )
        required = next(
            item
            for item in adapter.findall("param")
            if item.attrib.get("name") == "require_control_enable"
        )
        self.assertEqual(required.attrib.get("value"), "true")

    def test_real_config_uses_conservative_rates_and_tracking_feedback(self):
        self.assertIn("max_body_rate_rad_s: [0.6, 0.6, 0.6]", self.config)
        self.assertIn("feedback_mode: tracking_error", self.config)
        self.assertIn("thrust_attitude_projection:\n  enabled: true", self.config)


class TestControlEnableLease(unittest.TestCase):
    def test_output_is_fail_closed_until_first_true(self):
        lease = ControlEnableLease(required=True, timeout_s=0.25)
        permitted, _, reason, timed_out = lease.authorize(1.0, 10.0)
        self.assertFalse(permitted)
        self.assertEqual(reason, "control_disabled")
        self.assertFalse(timed_out)

    def test_true_heartbeat_renews_same_generation(self):
        lease = ControlEnableLease(required=True, timeout_s=0.25)
        first_generation = lease.update(True, 1.0, 10.0)
        permitted, generation, _, _ = lease.authorize(1.24, 10.24)
        self.assertTrue(permitted)
        self.assertEqual(generation, first_generation)

        renewed_generation = lease.update(True, 1.20, 10.20)
        permitted, generation, _, _ = lease.authorize(1.44, 10.44)
        self.assertTrue(permitted)
        self.assertEqual(renewed_generation, first_generation)
        self.assertEqual(generation, first_generation)

    def test_timeout_revokes_once_per_grant(self):
        lease = ControlEnableLease(required=True, timeout_s=0.25)
        lease.update(True, 1.0, 10.0)
        permitted, _, reason, timed_out = lease.authorize(1.251, 10.251)
        self.assertFalse(permitted)
        self.assertEqual(reason, "control_enable_timeout")
        self.assertTrue(timed_out)

        permitted, _, reason, timed_out = lease.authorize(1.252, 10.252)
        self.assertFalse(permitted)
        self.assertEqual(reason, "control_disabled")
        self.assertFalse(timed_out)

    def test_false_immediately_revokes_and_changes_generation(self):
        lease = ControlEnableLease(required=True, timeout_s=0.25)
        granted_generation = lease.update(True, 1.0, 10.0)
        revoked_generation = lease.update(False, 1.01, 10.01)
        permitted, generation, reason, _ = lease.authorize(1.01, 10.01)
        self.assertFalse(permitted)
        self.assertEqual(reason, "control_disabled")
        self.assertNotEqual(revoked_generation, granted_generation)
        self.assertEqual(generation, revoked_generation)

    def test_ros_time_reset_is_a_timeout(self):
        lease = ControlEnableLease(required=True, timeout_s=0.25)
        lease.update(True, 10.0, 20.0)
        permitted, _, reason, timed_out = lease.authorize(9.0, 20.01)
        self.assertFalse(permitted)
        self.assertEqual(reason, "control_enable_timeout")
        self.assertTrue(timed_out)

    def test_monotonic_timeout_catches_paused_ros_time(self):
        lease = ControlEnableLease(required=True, timeout_s=0.25)
        lease.update(True, 10.0, 20.0)
        permitted, _, reason, timed_out = lease.authorize(10.0, 20.251)
        self.assertFalse(permitted)
        self.assertEqual(reason, "control_enable_timeout")
        self.assertTrue(timed_out)

    def test_independent_wall_watchdog_tolerates_sim_scheduler_jitter(self):
        lease = ControlEnableLease(
            required=True, timeout_s=0.25, wall_timeout_s=1.0
        )
        lease.update(True, 10.0, 20.0)
        permitted, _, reason, timed_out = lease.authorize(10.05, 20.30)
        self.assertTrue(permitted)
        self.assertEqual(reason, "control_enabled")
        self.assertFalse(timed_out)

        permitted, _, reason, timed_out = lease.authorize(10.05, 21.001)
        self.assertFalse(permitted)
        self.assertEqual(reason, "control_enable_timeout")
        self.assertTrue(timed_out)

    def test_non_finite_or_reversed_wall_time_fails_closed(self):
        for wall_now in (math.nan, math.inf, 19.0):
            with self.subTest(wall_now=wall_now):
                lease = ControlEnableLease(required=True, timeout_s=0.25)
                lease.update(True, 10.0, 20.0)
                self.assertFalse(lease.authorize(10.01, wall_now)[0])

    def test_generation_detects_revoke_and_regrant_race(self):
        lease = ControlEnableLease(required=True, timeout_s=0.25)
        snapshot_generation = lease.update(True, 1.0, 10.0)
        lease.update(False, 1.01, 10.01)
        lease.update(True, 1.02, 10.02)
        permitted, final_generation, _, _ = lease.authorize(1.03, 10.03)
        self.assertTrue(permitted)
        self.assertNotEqual(final_generation, snapshot_generation)

    def test_lease_can_be_disabled_for_other_launches(self):
        lease = ControlEnableLease(required=False, timeout_s=0.25)
        permitted, _, reason, timed_out = lease.authorize(100.0, 200.0)
        self.assertTrue(permitted)
        self.assertEqual(reason, "control_enable_not_required")
        self.assertFalse(timed_out)

    def test_invalid_timeout_is_rejected(self):
        for timeout_s in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(timeout_s=timeout_s):
                with self.assertRaises(ValueError):
                    ControlEnableLease(required=True, timeout_s=timeout_s)

    def test_invalid_wall_timeout_is_rejected(self):
        for timeout_s in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(timeout_s=timeout_s):
                with self.assertRaises(ValueError):
                    ControlEnableLease(
                        required=True,
                        timeout_s=0.25,
                        wall_timeout_s=timeout_s,
                    )


class TestPX4AdapterFinalAuthorization(unittest.TestCase):
    class RecordingGuard:
        def __init__(self):
            self.apply_count = 0
            self.reset_count = 0

        def apply(self, *_args):
            self.apply_count += 1
            return SimpleNamespace(body_rate_rad_s=[0.0, 0.0, 0.0])

        def reset(self):
            self.reset_count += 1

    def make_node(self):
        node = object.__new__(PX4AdapterNode)
        node.lock = threading.Lock()
        node.control_enable_lease = ControlEnableLease(
            required=True, timeout_s=0.25
        )
        node.command_timeout_s = 0.10
        node.blocked_audits = []
        node.publish_blocked_audit = lambda stamp, reason: node.blocked_audits.append(
            (stamp.to_sec(), reason)
        )
        node.tight_fov_guard = self.RecordingGuard()
        node.tight_fov_guard_enabled = True
        node.tight_fov_guard_parameters = SimpleNamespace(
            maximum_feature_age_s=0.12
        )
        node.visibility_guard_enabled = False
        node.visibility_guard = None
        node.visibility_guard_max_feature_age_s = 0.12
        node.feature = None
        return node

    @staticmethod
    def reference(stamp_s):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=MODULE.rospy.Time.from_sec(stamp_s))
        )

    def test_false_ack_precedes_and_blocks_all_old_generation_outputs(self):
        node = self.make_node()
        generation = node.control_enable_lease.update(True, 1.0, 10.0)
        outputs = []

        with mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.01),
        ), mock.patch.object(MODULE.time, "monotonic", return_value=10.01):
            node.control_enable_callback(SimpleNamespace(data=False))
            published = node._publish_if_authorized(
                self.reference(1.0),
                generation,
                lambda _stamp, _permitted: outputs.extend(
                    ("setpoint", "permitted_audit")
                ),
            )

        self.assertFalse(published)
        self.assertEqual(node.blocked_audits, [(1.01, "control_disabled")])
        self.assertEqual(outputs, [])
        self.assertEqual(node.tight_fov_guard.reset_count, 1)

    def test_revoke_and_regrant_race_blocks_invalid_audit(self):
        node = self.make_node()
        generation = node.control_enable_lease.update(True, 1.0, 10.0)
        node.control_enable_lease.update(False, 1.01, 10.01)
        node.control_enable_lease.update(True, 1.02, 10.02)
        permitted_audits = []

        with mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.03),
        ), mock.patch.object(MODULE.time, "monotonic", return_value=10.03):
            published = node._publish_if_authorized(
                self.reference(1.0),
                generation,
                lambda stamp, permitted: permitted_audits.append(
                    (stamp.to_sec(), permitted, "invalid_command")
                ),
            )

        self.assertFalse(published)
        self.assertEqual(permitted_audits, [])

    def test_timeout_ack_is_single_and_blocks_all_outputs(self):
        node = self.make_node()
        generation = node.control_enable_lease.update(True, 1.0, 10.0)
        outputs = []

        with mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.30),
        ), mock.patch.object(MODULE.time, "monotonic", return_value=10.30):
            first = node._publish_if_authorized(
                self.reference(1.29), generation, lambda *_args: outputs.append(1)
            )
            second = node._publish_if_authorized(
                self.reference(1.29), generation, lambda *_args: outputs.append(2)
            )

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(
            node.blocked_audits, [(1.30, "control_enable_timeout")]
        )
        self.assertEqual(outputs, [])
        self.assertEqual(node.tight_fov_guard.reset_count, 1)

    def test_old_generation_cannot_rebuild_guard_after_revoke_and_regrant(self):
        node = self.make_node()
        old_generation = node.control_enable_lease.update(True, 1.0, 10.0)
        node.control_enable_lease.update(False, 1.01, 10.01)
        node.tight_fov_guard.reset()
        node.control_enable_lease.update(True, 1.02, 10.02)

        with mock.patch.object(MODULE.time, "monotonic", return_value=10.03):
            authorized, result, feature_valid, _, error = (
                node._apply_tight_fov_guard_for_generation(
                    MODULE.rospy.Time.from_sec(1.03),
                    old_generation,
                    [0.0, 0.0, 0.0],
                )
            )

        self.assertFalse(authorized)
        self.assertIsNone(result)
        self.assertFalse(feature_valid)
        self.assertIsNone(error)
        self.assertEqual(node.tight_fov_guard.apply_count, 0)
        self.assertEqual(node.tight_fov_guard.reset_count, 1)

    def test_guard_stage_timeout_resets_before_regrant(self):
        node = self.make_node()
        generation = node.control_enable_lease.update(True, 1.0, 10.0)

        with mock.patch.object(MODULE.time, "monotonic", return_value=10.30):
            authorized, result, feature_valid, _, error = (
                node._apply_tight_fov_guard_for_generation(
                    MODULE.rospy.Time.from_sec(1.30),
                    generation,
                    [0.0, 0.0, 0.0],
                )
            )

        self.assertFalse(authorized)
        self.assertIsNone(result)
        self.assertFalse(feature_valid)
        self.assertIsNone(error)
        self.assertEqual(node.tight_fov_guard.apply_count, 0)
        self.assertEqual(node.tight_fov_guard.reset_count, 1)
        self.assertEqual(
            node.blocked_audits, [(1.30, "control_enable_timeout")]
        )

    def test_timeout_then_regrant_consumes_only_fresh_generation_feature(self):
        node = self.make_node()
        node.feature = SimpleNamespace(
            header=SimpleNamespace(
                stamp=MODULE.rospy.Time.from_sec(1.04),
                frame_id="camera_optical",
            ),
            detected=True,
            x_normalized=0.01,
            y_normalized=0.0,
        )
        first_generation = node.control_enable_lease.update(True, 1.0, 10.0)

        with mock.patch.object(MODULE.time, "monotonic", return_value=10.05):
            authorized, result, feature_valid, _, error = (
                node._apply_tight_fov_guard_for_generation(
                    MODULE.rospy.Time.from_sec(1.05),
                    first_generation,
                    [0.0, 0.0, 0.0],
                )
            )
        self.assertTrue(authorized)
        self.assertIsNotNone(result)
        self.assertTrue(feature_valid)
        self.assertIsNone(error)
        self.assertEqual(node.tight_fov_guard.apply_count, 1)

        with mock.patch.object(MODULE.time, "monotonic", return_value=10.30):
            self.assertFalse(
                node._apply_tight_fov_guard_for_generation(
                    MODULE.rospy.Time.from_sec(1.30),
                    first_generation,
                    [0.0, 0.0, 0.0],
                )[0]
            )
        self.assertEqual(node.tight_fov_guard.reset_count, 1)

        new_generation = node.control_enable_lease.update(True, 1.31, 10.31)
        node.feature.header.stamp = MODULE.rospy.Time.from_sec(1.31)
        with mock.patch.object(MODULE.time, "monotonic", return_value=10.32):
            authorized, result, feature_valid, _, error = (
                node._apply_tight_fov_guard_for_generation(
                    MODULE.rospy.Time.from_sec(1.32),
                    new_generation,
                    [0.0, 0.0, 0.0],
                )
            )

        self.assertNotEqual(new_generation, first_generation)
        self.assertTrue(authorized)
        self.assertIsNotNone(result)
        self.assertTrue(feature_valid)
        self.assertIsNone(error)
        self.assertEqual(node.tight_fov_guard.apply_count, 2)
        self.assertEqual(node.tight_fov_guard.reset_count, 1)

    def test_fresh_matching_generation_runs_action_once(self):
        node = self.make_node()
        generation = node.control_enable_lease.update(True, 1.0, 10.0)
        outputs = []

        with mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.05),
        ), mock.patch.object(MODULE.time, "monotonic", return_value=10.05):
            published = node._publish_if_authorized(
                self.reference(1.0),
                generation,
                lambda stamp, permitted: outputs.append(
                    (stamp.to_sec(), permitted)
                ),
            )

        self.assertTrue(published)
        self.assertEqual(outputs, [(1.05, True)])


class TestVerticalVisibilityGuardAdapter(unittest.TestCase):
    class RecordingGuard:
        def __init__(self, axis, replacement):
            self.axis = axis
            self.replacement = replacement
            self.inputs = []
            self.reset_count = 0

        def apply(self, *args):
            rate = list(args[-1])
            self.inputs.append((args[:-1], list(rate)))
            rate[self.axis] = self.replacement
            return SimpleNamespace(body_rate_rad_s=rate)

        def reset(self):
            self.reset_count += 1

    @staticmethod
    def make_node():
        node = object.__new__(PX4AdapterNode)
        node.lock = threading.Lock()
        node.control_enable_lease = ControlEnableLease(required=True, timeout_s=0.25)
        node.blocked_audits = []
        node.publish_blocked_audit = lambda stamp, reason: node.blocked_audits.append(
            (stamp.to_sec(), reason)
        )
        node.tight_fov_guard_enabled = True
        node.tight_fov_guard_parameters = SimpleNamespace(
            maximum_feature_age_s=0.12
        )
        node.tight_fov_guard = TestVerticalVisibilityGuardAdapter.RecordingGuard(
            axis=2, replacement=-0.4
        )
        node.visibility_guard_enabled = True
        node.visibility_guard_max_feature_age_s = 0.10
        node.visibility_guard = TestVerticalVisibilityGuardAdapter.RecordingGuard(
            axis=1, replacement=0.6
        )
        node.feature = SimpleNamespace(
            header=SimpleNamespace(
                stamp=MODULE.rospy.Time.from_sec(1.04),
                frame_id="camera_optical",
            ),
            detected=True,
            x_normalized=0.01,
            y_normalized=0.20,
        )
        return node

    def test_calibration_produces_asymmetric_half_open_vertical_bounds(self):
        bounds = vertical_visibility_bounds(480, 554.254691191187, 240.5, 120.0)

        self.assertAlmostEqual(bounds[0], -240.5 / 554.254691191187)
        self.assertAlmostEqual(bounds[1], 239.5 / 554.254691191187)
        self.assertAlmostEqual(bounds[2], -120.5 / 554.254691191187)
        self.assertAlmostEqual(bounds[3], 119.5 / 554.254691191187)
        self.assertNotAlmostEqual(abs(bounds[0]), abs(bounds[1]))

    def test_invalid_calibration_and_margin_are_rejected(self):
        for args in (
            (0, 554.0, 240.0, 120.0),
            (480, 0.0, 240.0, 120.0),
            (480, 554.0, 480.0, 120.0),
            (480, 554.0, 240.0, 240.0),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                vertical_visibility_bounds(*args)

    def test_visibility_recovery_thrust_scale_blends_to_configured_minimum(self):
        parameters = SimpleNamespace(
            acceptance_lower_normalized=-0.5,
            acceptance_upper_normalized=0.5,
            activation_lower_normalized=-0.3,
            activation_upper_normalized=0.3,
        )
        inactive = SimpleNamespace(active=False)
        upper_midpoint = SimpleNamespace(
            active=True,
            active_boundary="upper",
            measured_coordinate_normalized=0.4,
        )
        lower_edge = SimpleNamespace(
            active=True,
            active_boundary="lower",
            measured_coordinate_normalized=-0.5,
        )

        self.assertAlmostEqual(
            MODULE.visibility_recovery_thrust_scale(inactive, parameters, 0.6),
            1.0,
        )
        self.assertAlmostEqual(
            MODULE.visibility_recovery_thrust_scale(
                upper_midpoint, parameters, 0.6
            ),
            0.8,
        )
        self.assertAlmostEqual(
            MODULE.visibility_recovery_thrust_scale(
                lower_edge, parameters, 0.6
            ),
            0.6,
        )

    def test_visibility_recovery_thrust_scale_rejects_unsafe_configuration(self):
        with self.assertRaises(ValueError):
            MODULE.visibility_recovery_thrust_scale(None, None, 0.0)

    def test_attitude_thrust_projection_uses_body_z_alignment(self):
        half_angle = math.radians(6.0)
        desired = [0.0, math.sin(half_angle), 0.0, math.cos(half_angle)]
        self.assertAlmostEqual(
            attitude_alignment_thrust_scale(desired, [0.0, 0.0, 0.0, 1.0], 0.5),
            math.cos(math.radians(12.0)),
        )
        self.assertAlmostEqual(
            attitude_alignment_thrust_scale(
                [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 0.5
            ),
            0.5,
        )

    def test_attitude_thrust_projection_preserves_vertical_component_during_lag(self):
        desired_half_angle = math.radians(6.0)
        measured_half_angle = math.radians(3.0)
        desired = [
            0.0,
            math.sin(desired_half_angle),
            0.0,
            math.cos(desired_half_angle),
        ]
        measured = [
            0.0,
            math.sin(measured_half_angle),
            0.0,
            math.cos(measured_half_angle),
        ]

        self.assertAlmostEqual(
            attitude_alignment_thrust_scale(desired, measured, 0.5),
            math.cos(math.radians(12.0)) / math.cos(math.radians(6.0)),
        )

    def test_attitude_thrust_projection_rejects_invalid_scale(self):
        with self.assertRaises(ValueError):
            attitude_alignment_thrust_scale(
                [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], 0.0
            )

    def test_both_guards_consume_same_feature_and_chain_body_rate(self):
        node = self.make_node()
        generation = node.control_enable_lease.update(True, 1.0, 10.0)

        with mock.patch.object(MODULE.time, "monotonic", return_value=10.05):
            result = node._apply_pixel_guards_for_generation(
                MODULE.rospy.Time.from_sec(1.05),
                generation,
                [0.1, -0.2, 0.3],
            )

        authorized, tight, visibility, feature_valid = result[:4]
        self.assertTrue(authorized)
        self.assertTrue(feature_valid)
        self.assertEqual(len(node.tight_fov_guard.inputs), 1)
        self.assertEqual(len(node.visibility_guard.inputs), 1)
        self.assertEqual(
            node.tight_fov_guard.inputs[0][0][-1],
            node.visibility_guard.inputs[0][0][-1],
        )
        self.assertEqual(node.visibility_guard.inputs[0][1], [0.1, -0.2, -0.4])
        self.assertEqual(tight.body_rate_rad_s, [0.1, -0.2, -0.4])
        self.assertEqual(visibility.body_rate_rad_s, [0.1, 0.6, -0.4])

    def test_tighter_feature_age_limit_applies_to_both_guards(self):
        node = self.make_node()
        generation = node.control_enable_lease.update(True, 1.0, 10.0)

        with mock.patch.object(MODULE.time, "monotonic", return_value=10.15):
            result = node._apply_pixel_guards_for_generation(
                MODULE.rospy.Time.from_sec(1.15), generation, [0.0, 0.0, 0.0]
            )

        self.assertTrue(result[0])
        self.assertFalse(result[3])
        self.assertEqual(node.tight_fov_guard.inputs, [])
        self.assertEqual(node.visibility_guard.inputs, [])
        self.assertEqual(node.tight_fov_guard.reset_count, 1)
        self.assertEqual(node.visibility_guard.reset_count, 1)

    def test_revoke_resets_both_guards_before_ack(self):
        node = self.make_node()
        node.control_enable_lease.update(True, 1.0, 10.0)

        with mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.01),
        ), mock.patch.object(MODULE.time, "monotonic", return_value=10.01):
            node.control_enable_callback(SimpleNamespace(data=False))

        self.assertEqual(node.tight_fov_guard.reset_count, 1)
        self.assertEqual(node.visibility_guard.reset_count, 1)
        self.assertEqual(node.blocked_audits, [(1.01, "control_disabled")])


class TestPX4AdapterCommandChain(unittest.TestCase):
    class RecordingPublisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    @staticmethod
    def vector(message):
        return np.array([message.x, message.y, message.z], dtype=float)

    @staticmethod
    def enable_body_rate_prediction(node):
        node.tight_fov_guard_parameters = MODULE.TightFOVGuardParameters(
            activation_rad=math.radians(5.0),
            rate_gain=2.0,
            maximum_guarded_rate_rad_s=0.4,
            maximum_feature_age_s=0.12,
            derivative_enabled=False,
            body_rate_prediction_enabled=True,
            body_rate_prediction_horizon_s=0.18,
            body_rate_deadband_rad_s=0.025,
        )
        node.tight_fov_guard = MODULE.TightFOVGuard(
            node.tight_fov_acceptance_rad,
            node.tight_fov_guard_parameters,
        )

    def make_node(self):
        node = object.__new__(PX4AdapterNode)
        node.lock = threading.Lock()
        node.reference = SimpleNamespace(
            header=SimpleNamespace(stamp=MODULE.rospy.Time.from_sec(1.0)),
            valid=True,
            mode=MODULE.ControlReference.MODE_TRACK,
            desired_attitude=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            desired_body_rate=SimpleNamespace(x=1.5, y=-0.4, z=0.5),
            normalized_thrust=1.0,
        )
        node.feature = SimpleNamespace(
            header=SimpleNamespace(
                stamp=MODULE.rospy.Time.from_sec(1.04),
                frame_id="camera_optical",
            ),
            detected=True,
            u=500.0,
            v=300.0,
            x_normalized=0.5,
            y_normalized=0.3,
        )
        node.control_enable_lease = ControlEnableLease(
            required=True, timeout_s=0.25
        )
        node.control_enable_lease.update(True, 1.0, 10.0)
        node.command_timeout_s = 0.10
        node.require_control_enable = True

        node.rate_reversal_boost_enabled = True
        node.rate_reversal_booster = MODULE.BodyRateReversalBooster(
            MODULE.RateReversalBoostParameters(
                gains=(0.0, 0.6, 1.2),
                damping_gains=(0.0, 0.2, 0.6),
                tracking_error_gains=(0.0, 0.0, 0.2),
                axis_feedback_modes=(
                    "legacy_reversal",
                    "legacy_reversal",
                    "tracking_error",
                ),
                filter_tau_s=0.04,
                max_correction_rad_s=(0.35, 0.35, 0.35),
            )
        )
        node.rate_reversal_booster.update([0.0, 0.2, -0.5], 1.04)
        node.imu_stamp = MODULE.rospy.Time.from_sec(1.04)
        node.max_imu_age_s = 0.10

        node.tight_fov_guard_enabled = True
        node.tight_fov_acceptance_rad = math.radians(20.0)
        node.tight_fov_guard_parameters = MODULE.TightFOVGuardParameters(
            activation_rad=math.radians(5.0),
            rate_gain=2.0,
            maximum_guarded_rate_rad_s=0.4,
            maximum_feature_age_s=0.12,
            derivative_enabled=False,
        )
        node.tight_fov_guard = MODULE.TightFOVGuard(
            node.tight_fov_acceptance_rad,
            node.tight_fov_guard_parameters,
        )

        node.visibility_guard_enabled = True
        node.visibility_guard_max_feature_age_s = 0.12
        node.visibility_guard_parameters = MODULE.PixelAxisGuardParameters(
            acceptance_lower_normalized=-0.5,
            acceptance_upper_normalized=0.5,
            activation_lower_normalized=-0.2,
            activation_upper_normalized=0.2,
            rate_gain=3.0,
            maximum_guarded_rate_rad_s=0.6,
            derivative_enabled=False,
        )
        node.visibility_guard = MODULE.PixelAxisGuard(
            node.visibility_guard_parameters,
            body_rate_axis_index=1,
            restoring_sign=1.0,
        )

        node.parameters = MODULE.CommandSafetyParameters(
            max_body_rate_rad_s=(1.2, 1.2, 0.8)
        )
        node.control_mode = "body_rate"
        node.attitude_target_type_mask = MODULE.AttitudeTarget.IGNORE_ATTITUDE
        node.publisher = self.RecordingPublisher()
        node.audit_publisher = self.RecordingPublisher()
        node.visibility_audit_publisher = self.RecordingPublisher()
        return node

    def test_mixed_feedback_guards_sanitize_and_audits_form_one_chain(self):
        node = self.make_node()
        original_sanitize = MODULE.sanitize_command
        with mock.patch.object(
            node.rate_reversal_booster,
            "compensate",
            wraps=node.rate_reversal_booster.compensate,
        ) as feedback_spy, mock.patch.object(
            node.tight_fov_guard,
            "apply",
            wraps=node.tight_fov_guard.apply,
        ) as tight_spy, mock.patch.object(
            node.visibility_guard,
            "apply",
            wraps=node.visibility_guard.apply,
        ) as visibility_spy, mock.patch.object(
            MODULE,
            "sanitize_command",
            wraps=original_sanitize,
        ) as sanitize_spy, mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.05),
        ), mock.patch.object(MODULE.time, "monotonic", return_value=10.05):
            node.timer_callback(None)

        self.assertEqual(feedback_spy.call_count, 1)
        self.assertEqual(tight_spy.call_count, 1)
        self.assertEqual(visibility_spy.call_count, 1)
        self.assertEqual(sanitize_spy.call_count, 1)
        self.assertEqual(len(node.publisher.messages), 1)
        self.assertEqual(len(node.audit_publisher.messages), 1)
        self.assertEqual(len(node.visibility_audit_publisher.messages), 1)

        reference_rate = np.array([1.5, -0.4, 0.5])
        feedback_delta = np.array([0.0, -0.16, 0.2])
        feedback_output = reference_rate + feedback_delta
        np.testing.assert_allclose(feedback_spy.call_args.args[0], reference_rate)
        np.testing.assert_allclose(tight_spy.call_args.args[-1], feedback_output)

        audit = node.audit_publisher.messages[0]
        np.testing.assert_allclose(
            self.vector(audit.damping_correction_body_rate), feedback_delta
        )
        self.assertTrue(audit.rate_feedback_enabled)
        self.assertEqual(
            list(audit.rate_axis_feedback_modes),
            ["legacy_reversal", "legacy_reversal", "tracking_error"],
        )
        np.testing.assert_allclose(
            self.vector(audit.rate_reversal_gains), [0.0, 0.6, 1.2]
        )
        np.testing.assert_allclose(
            self.vector(audit.rate_damping_gains), [0.0, 0.2, 0.6]
        )
        np.testing.assert_allclose(
            self.vector(audit.rate_tracking_error_gains), [0.0, 0.0, 0.2]
        )
        self.assertAlmostEqual(audit.rate_filter_tau_s, 0.04)
        np.testing.assert_allclose(
            self.vector(audit.rate_max_correction_rad_s), [0.35, 0.35, 0.35]
        )
        self.assertAlmostEqual(audit.rate_reset_gap_s, 0.20)
        self.assertAlmostEqual(audit.rate_max_imu_age_s, 0.10)
        tight_correction = self.vector(audit.guard_correction_body_rate)
        tight_output = feedback_output + tight_correction
        np.testing.assert_allclose(visibility_spy.call_args.args[-1], tight_output)
        self.assertLess(tight_output[2], 0.0)

        visibility_audit = node.visibility_audit_publisher.messages[0]
        self.assertAlmostEqual(visibility_audit.input_pitch_rate, tight_output[1])
        self.assertAlmostEqual(visibility_audit.output_pitch_rate, 0.3)
        self.assertAlmostEqual(
            visibility_audit.output_pitch_rate
            - visibility_audit.input_pitch_rate,
            0.86,
        )
        visibility_output = tight_output.copy()
        visibility_output[1] = visibility_audit.output_pitch_rate
        np.testing.assert_allclose(sanitize_spy.call_args.args[1], visibility_output)
        np.testing.assert_allclose(
            self.vector(audit.pre_safety_body_rate), visibility_output
        )

        expected_sent_rate = np.array([1.2, 0.3, -0.4])
        target = node.publisher.messages[0]
        sent_setpoint_rate = self.vector(target.body_rate)
        np.testing.assert_allclose(sent_setpoint_rate, expected_sent_rate)
        np.testing.assert_allclose(
            self.vector(audit.sent_body_rate), expected_sent_rate
        )
        self.assertTrue(audit.rate_saturated)
        self.assertFalse(audit.thrust_saturated)
        self.assertEqual(audit.reason, "saturated")
        self.assertAlmostEqual(target.thrust, 0.5)
        self.assertAlmostEqual(audit.sent_thrust, target.thrust)

        self.assertTrue(audit.guard_active)
        self.assertTrue(audit.guard_limited)
        self.assertLess(audit.measured_margin_tight, 0.0)
        self.assertLessEqual(audit.measured_s_tight * sent_setpoint_rate[2], 0.0)
        self.assertTrue(visibility_audit.active)
        self.assertTrue(visibility_audit.intervened)
        self.assertTrue(visibility_audit.within_acceptance)
        self.assertEqual(visibility_audit.reason, "accepted")

    def test_body_rate_prediction_consumes_fresh_filtered_imu_and_is_audited(self):
        node = self.make_node()
        self.enable_body_rate_prediction(node)
        node.feature.x_normalized = 0.05
        node.feature.y_normalized = 0.0
        node.rate_reversal_booster.reset()
        node.rate_reversal_booster.update([0.0, 0.2, 0.5], 1.04)

        with mock.patch.object(
            node.tight_fov_guard,
            "apply",
            wraps=node.tight_fov_guard.apply,
        ) as tight_spy, mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.05),
        ), mock.patch.object(MODULE.time, "monotonic", return_value=10.05):
            node.timer_callback(None)

        self.assertEqual(tight_spy.call_count, 1)
        self.assertAlmostEqual(
            tight_spy.call_args.kwargs["measured_body_yaw_rate_rad_s"], 0.5
        )
        self.assertAlmostEqual(tight_spy.call_args.kwargs["feature_age_s"], 0.01)

        self.assertEqual(len(node.audit_publisher.messages), 1)
        audit = node.audit_publisher.messages[0]
        self.assertTrue(audit.guard_body_rate_prediction_enabled)
        self.assertAlmostEqual(audit.guard_body_rate_prediction_horizon_s, 0.18)
        self.assertAlmostEqual(audit.guard_body_rate_deadband_rad_s, 0.025)
        self.assertTrue(audit.guard_body_rate_prediction_initialized)
        self.assertTrue(audit.guard_body_rate_prediction_active)
        self.assertAlmostEqual(audit.guard_measured_body_yaw_rate_rad_s, 0.5)

        ray_z = 1.0 / math.sqrt(1.0 + 0.05**2)
        measured_s_tight = 0.05 * ray_z
        prediction_interval = 0.01 + 0.18
        self.assertAlmostEqual(audit.guard_rotational_s_tight_rate, 0.5 * ray_z)
        self.assertAlmostEqual(
            audit.guard_body_rate_prediction_interval_s, prediction_interval
        )
        self.assertAlmostEqual(
            audit.guard_body_rate_predicted_s_tight,
            measured_s_tight + prediction_interval * 0.5 * ray_z,
        )
        self.assertLess(audit.guard_body_rate_prediction_rate_rad_s, 0.0)
        self.assertLess(audit.guard_correction_body_rate.z, 0.0)

    def test_body_rate_prediction_blocks_output_until_imu_is_fresh(self):
        for imu_state in ("stale", "uninitialized"):
            with self.subTest(imu_state=imu_state):
                node = self.make_node()
                self.enable_body_rate_prediction(node)
                node.feature.x_normalized = 0.05
                node.feature.y_normalized = 0.0
                if imu_state == "stale":
                    node.imu_stamp = MODULE.rospy.Time.from_sec(0.90)
                else:
                    node.imu_stamp = None
                    node.rate_reversal_booster.reset()

                with mock.patch.object(
                    node.tight_fov_guard,
                    "apply",
                    wraps=node.tight_fov_guard.apply,
                ) as tight_spy, mock.patch.object(
                    MODULE.rospy.Time,
                    "now",
                    return_value=MODULE.rospy.Time.from_sec(1.05),
                ), mock.patch.object(MODULE.time, "monotonic", return_value=10.05):
                    node.timer_callback(None)

                self.assertEqual(tight_spy.call_count, 0)
                self.assertEqual(len(node.publisher.messages), 0)
                self.assertEqual(len(node.audit_publisher.messages), 0)

    def test_body_rate_prediction_holds_last_rate_across_short_imu_gap(self):
        node = self.make_node()
        self.enable_body_rate_prediction(node)
        node.feature.x_normalized = 0.05
        node.feature.y_normalized = 0.0
        node.imu_stamp = MODULE.rospy.Time.from_sec(0.90)
        node.latest_measured_body_yaw_rate_rad_s = 0.3

        with mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.05),
        ), mock.patch.object(MODULE.time, "monotonic", return_value=10.05):
            node.timer_callback(None)

        self.assertEqual(len(node.publisher.messages), 1)
        self.assertEqual(len(node.audit_publisher.messages), 1)
        audit = node.audit_publisher.messages[0]
        self.assertTrue(audit.guard_body_rate_prediction_initialized)
        self.assertAlmostEqual(audit.guard_measured_body_yaw_rate_rad_s, 0.3)


class TestPixelGuardFailClosed(unittest.TestCase):
    class RecordingPublisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class PassthroughGuard:
        def __init__(self):
            self.reset_count = 0

        def apply(self, *_args):
            body_rate = list(_args[-1])
            return SimpleNamespace(
                active=False,
                limited=False,
                intervention_body_rate_rad_s=[0.0, 0.0, 0.0],
                derivative_initialized=False,
                measured_s_tight=0.0,
                measured_margin_tight=0.01,
                filtered_s_tight_rate_s_inv=0.0,
                outward_s_tight_rate_s_inv=0.0,
                derivative_rate_rad_s=0.0,
                body_rate_rad_s=body_rate,
            )

        def reset(self):
            self.reset_count += 1

    @staticmethod
    def feature(stamp_s, detected=True, frame_id="camera_optical"):
        return SimpleNamespace(
            header=SimpleNamespace(
                stamp=MODULE.rospy.Time.from_sec(stamp_s),
                frame_id=frame_id,
            ),
            detected=detected,
            u=320.5,
            v=240.5,
            x_normalized=0.0,
            y_normalized=0.0,
        )

    def make_node(self):
        node = object.__new__(PX4AdapterNode)
        node.lock = threading.Lock()
        node.reference = SimpleNamespace(
            header=SimpleNamespace(stamp=MODULE.rospy.Time.from_sec(1.0)),
            valid=True,
            mode=MODULE.ControlReference.MODE_TRACK,
            desired_attitude=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            desired_body_rate=SimpleNamespace(x=0.0, y=2.0, z=0.0),
            normalized_thrust=1.0,
        )
        node.feature = self.feature(1.04)
        node.control_enable_lease = ControlEnableLease(
            required=True, timeout_s=0.25
        )
        node.control_enable_lease.update(True, 1.0, 10.0)
        node.command_timeout_s = 0.10
        node.require_control_enable = True
        node.rate_reversal_boost_enabled = False
        node.rate_reversal_booster = MODULE.BodyRateReversalBooster(
            MODULE.RateReversalBoostParameters()
        )
        node.max_imu_age_s = 0.10
        node.tight_fov_guard_enabled = True
        node.tight_fov_guard_parameters = SimpleNamespace(
            maximum_feature_age_s=0.12,
            derivative_enabled=False,
        )
        node.tight_fov_guard = self.PassthroughGuard()
        node.visibility_guard_enabled = False
        node.visibility_guard_max_feature_age_s = 0.12
        node.visibility_guard = None
        node.parameters = MODULE.CommandSafetyParameters(
            max_body_rate_rad_s=(1.2, 1.2, 0.8)
        )
        node.control_mode = "body_rate"
        node.attitude_target_type_mask = MODULE.AttitudeTarget.IGNORE_ATTITUDE
        node.publisher = self.RecordingPublisher()
        node.audit_publisher = self.RecordingPublisher()
        node.visibility_audit_publisher = self.RecordingPublisher()
        return node

    def test_new_invalid_feature_blocks_old_reference_without_saturation(self):
        node = self.make_node()
        node.feature_callback(self.feature(1.05, detected=False))

        with mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.05),
        ), mock.patch.object(MODULE.time, "monotonic", return_value=10.05):
            node.timer_callback(None)

        self.assertEqual(node.publisher.messages, [])
        self.assertEqual(len(node.audit_publisher.messages), 1)
        audit = node.audit_publisher.messages[0]
        self.assertFalse(audit.output_permitted)
        self.assertTrue(audit.control_enabled)
        self.assertFalse(audit.rate_feedback_enabled)
        self.assertEqual(
            list(audit.rate_axis_feedback_modes), ["legacy_reversal"] * 3
        )
        self.assertAlmostEqual(audit.rate_filter_tau_s, 0.04)
        self.assertAlmostEqual(audit.rate_reset_gap_s, 0.20)
        self.assertAlmostEqual(audit.rate_max_imu_age_s, 0.10)
        self.assertEqual(audit.reason, "feature_not_detected")
        self.assertFalse(audit.rate_saturated)
        self.assertEqual(
            (audit.sent_body_rate.x, audit.sent_body_rate.y, audit.sent_body_rate.z),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(node.visibility_audit_publisher.messages, [])

    def test_newer_feature_does_not_invalidate_fresh_guard_snapshot(self):
        node = self.make_node()
        node.visibility_audit_publisher = None
        original_sanitize = MODULE.sanitize_command

        def invalidate_before_final_authorization(*args, **kwargs):
            command = original_sanitize(*args, **kwargs)
            node.feature_callback(self.feature(1.05, frame_id="map"))
            return command

        with mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.05),
        ), mock.patch.object(
            MODULE.time, "monotonic", return_value=10.05
        ), mock.patch.object(
            MODULE,
            "sanitize_command",
            side_effect=invalidate_before_final_authorization,
        ):
            node.timer_callback(None)

        self.assertEqual(len(node.publisher.messages), 1)
        self.assertEqual(len(node.audit_publisher.messages), 1)
        audit = node.audit_publisher.messages[0]
        self.assertTrue(audit.output_permitted)
        self.assertEqual(audit.reason, "saturated")
        self.assertTrue(audit.rate_saturated)
        self.assertIsNone(node.visibility_audit_publisher)

    def test_guard_error_is_distinct_and_blocks_all_command_outputs(self):
        node = self.make_node()
        node.tight_fov_guard.apply = mock.Mock(
            side_effect=ValueError("invalid guard input")
        )

        with mock.patch.object(
            MODULE.rospy.Time,
            "now",
            return_value=MODULE.rospy.Time.from_sec(1.05),
        ), mock.patch.object(MODULE.time, "monotonic", return_value=10.05):
            node.timer_callback(None)

        self.assertEqual(node.publisher.messages, [])
        self.assertEqual(len(node.audit_publisher.messages), 1)
        audit = node.audit_publisher.messages[0]
        self.assertFalse(audit.output_permitted)
        self.assertEqual(audit.reason, "tight_fov_guard_error")
        self.assertFalse(audit.rate_saturated)
        self.assertEqual(node.visibility_audit_publisher.messages, [])


if __name__ == "__main__":
    unittest.main()
