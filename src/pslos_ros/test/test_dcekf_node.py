#!/usr/bin/env python3
from collections import deque
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dcekf_node.py"
SPEC = importlib.util.spec_from_file_location("dcekf_node", str(SCRIPT))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DCEKFNode = MODULE.DCEKFNode
stationary_gravity_world_alignment = MODULE.stationary_gravity_world_alignment


class Stamp:
    def __init__(self, value):
        self.value = float(value)

    def to_sec(self):
        return self.value

    def is_zero(self):
        return self.value == 0.0


def feature(stamp):
    return SimpleNamespace(
        detected=True,
        header=SimpleNamespace(stamp=Stamp(stamp)),
        confidence=1.0,
        x_normalized=0.0,
        y_normalized=0.0,
    )


def imu(
    stamp,
    gyro=(0.0, 0.0, 0.0),
    acceleration=(0.0, 0.0, 9.80665),
    orientation=(0.0, 0.0, 0.0, 1.0),
):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=Stamp(stamp)),
        angular_velocity=SimpleNamespace(x=gyro[0], y=gyro[1], z=gyro[2]),
        linear_acceleration=SimpleNamespace(
            x=acceleration[0],
            y=acceleration[1],
            z=acceleration[2],
        ),
        orientation=SimpleNamespace(
            x=orientation[0],
            y=orientation[1],
            z=orientation[2],
            w=orientation[3],
        ),
        orientation_covariance=[0.0] * 9,
    )


class FakeEstimator:
    def __init__(self, last_feature_stamp, stamp=0.0):
        self.history = [object()]
        self.last_feature_stamp = float(last_feature_stamp)
        self._stamp = float(stamp)
        self._state = np.zeros(18)
        self._state[4:7] = [0.0, 0.0, -40.0]
        self.reset_calls = 0
        self.measurement_count = 7
        self.update_calls = 0
        self.propagate_calls = 0
        self.initialize_calls = []
        self.velocity_calls = []
        self.parameters = SimpleNamespace(
            feature_measurement_noise_std=0.01,
            gravity_world=(0.0, 0.0, -9.80665),
            max_history_s=0.5,
        )

    @property
    def initialized(self):
        return bool(self.history)

    @property
    def stamp(self):
        return self._stamp if self.initialized else None

    @property
    def state(self):
        return self._state.copy()

    def reset(self):
        self.history = []
        self.last_feature_stamp = None
        self.measurement_count = 0
        self.reset_calls += 1

    def propagate(self, stamp, *_args):
        self._stamp = float(stamp)
        self.propagate_calls += 1

    def initialize(
        self,
        stamp,
        attitude_xyzw,
        feature_normalized,
        initial_range_m,
        initial_gyro_bias,
        initial_accel_bias,
    ):
        self.history = [object()]
        self._stamp = float(stamp)
        self.last_feature_stamp = float(stamp)
        self.measurement_count = 1
        self.initialize_calls.append(
            (
                tuple(attitude_xyzw),
                tuple(feature_normalized),
                float(initial_range_m),
                np.asarray(initial_gyro_bias),
                np.asarray(initial_accel_bias),
            )
        )

    def update_feature(self, *_args):
        self.update_calls += 1
        return SimpleNamespace(accepted=True)

    def update_velocity(self, stamp, velocity, *_args, **_kwargs):
        self.velocity_calls.append(
            (
                float(stamp),
                tuple(velocity),
                bool(_kwargs.get("radial_only", False)),
                float(_kwargs.get("transverse_measurement_std", math.inf)),
            )
        )
        return SimpleNamespace(accepted=True)


class TestFeatureLossReset(unittest.TestCase):
    @staticmethod
    def node(last_feature_stamp, reset_s=0.5):
        node = DCEKFNode.__new__(DCEKFNode)
        node.estimator_lock = threading.RLock()
        node.estimator = FakeEstimator(last_feature_stamp)
        node.feature_loss_reset_s = reset_s
        node.require_vision_ready = False
        node.initialize_from_feature = mock.Mock(return_value=True)
        node.publish_state = mock.Mock()
        return node

    def test_long_feature_gap_reinitializes_from_fresh_prior(self):
        node = self.node(1.0)

        node.feature_callback(feature(1.6))

        self.assertEqual(node.estimator.reset_calls, 1)
        node.initialize_from_feature.assert_called_once()
        self.assertEqual(node.estimator.update_calls, 0)
        node.publish_state.assert_called_once_with(True)

    def test_short_feature_gap_keeps_continuous_track(self):
        node = self.node(1.0)

        node.feature_callback(feature(1.2))

        self.assertEqual(node.estimator.reset_calls, 0)
        node.initialize_from_feature.assert_not_called()
        self.assertEqual(node.estimator.update_calls, 1)
        node.publish_state.assert_called_once_with(True)

    def test_zero_reset_threshold_preserves_legacy_behavior(self):
        node = self.node(1.0, reset_s=0.0)

        node.feature_callback(feature(100.0))

        self.assertEqual(node.estimator.reset_calls, 0)
        self.assertEqual(node.estimator.update_calls, 1)


class TestStationaryGravityAttitudeAlignment(unittest.TestCase):
    def test_identity_attitude_and_vertical_specific_force_need_no_correction(self):
        correction, angle = stationary_gravity_world_alignment(
            [imu(1.0), imu(2.0)], math.radians(5.0)
        )

        np.testing.assert_allclose(correction, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(angle, 0.0, places=12)

    def test_reported_pitch_bias_is_removed_by_stationary_gravity(self):
        pitch = math.radians(-1.0)
        orientation = (0.0, math.sin(pitch / 2.0), 0.0, math.cos(pitch / 2.0))

        correction, angle = stationary_gravity_world_alignment(
            [imu(1.0, orientation=orientation)], math.radians(5.0)
        )
        reported = MODULE.quaternion_xyzw_to_rotation(orientation)

        np.testing.assert_allclose(correction.dot(reported), np.eye(3), atol=1e-12)
        self.assertAlmostEqual(math.degrees(angle), 1.0, places=9)

    def test_excessive_frame_error_is_rejected(self):
        pitch = math.radians(8.0)
        orientation = (0.0, math.sin(pitch / 2.0), 0.0, math.cos(pitch / 2.0))

        with self.assertRaisesRegex(ValueError, "exceeds"):
            stationary_gravity_world_alignment(
                [imu(1.0, orientation=orientation)], math.radians(5.0)
            )


class TestRuntimeReinitializationPrior(unittest.TestCase):
    def test_runtime_reset_reuses_last_range_and_measurement_continuity(self):
        node = DCEKFNode.__new__(DCEKFNode)
        node.estimator = FakeEstimator(last_feature_stamp=1.9, stamp=2.0)
        node.estimator._state[4:7] = [0.0, 0.0, -12.5]
        node.estimator.measurement_count = 321
        node.initial_range_m = 100.0
        node.pending_runtime_range_m = None
        node.pending_runtime_measurement_count = 0
        node.minimum_measurements = 3
        node.preinit_imu = deque([imu(2.0)])
        node.require_vision_ready = True
        node.frozen_initial_gyro_bias = np.zeros(3)
        node.frozen_initial_accel_bias = np.zeros(3)
        node.attitude_world_alignment_rotation = np.eye(3)
        node.attitude_aiding_enabled = False

        self.assertAlmostEqual(node.cache_runtime_reinitialization_prior(), 12.5)
        node.estimator.reset()
        self.assertTrue(node.initialize_from_feature(feature(2.0)))

        self.assertAlmostEqual(node.estimator.initialize_calls[0][2], 12.5)
        self.assertEqual(node.estimator.measurement_count, 321)
        self.assertIsNone(node.pending_runtime_range_m)
        self.assertEqual(node.pending_runtime_measurement_count, 0)

    def test_runtime_range_is_capped_by_cold_start_prior(self):
        node = DCEKFNode.__new__(DCEKFNode)
        node.estimator = FakeEstimator(last_feature_stamp=1.0)
        node.estimator._state[4:7] = [0.0, 0.0, -250.0]
        node.initial_range_m = 100.0
        node.pending_runtime_range_m = None
        node.pending_runtime_measurement_count = 0

        self.assertAlmostEqual(node.cache_runtime_reinitialization_prior(), 100.0)


class TestRecentImuCache(unittest.TestCase):
    @staticmethod
    def node(last_feature_stamp=2.91, stamp=1.0):
        node = DCEKFNode.__new__(DCEKFNode)
        node.estimator_lock = threading.RLock()
        node.estimator = FakeEstimator(last_feature_stamp, stamp)
        node.feature_loss_reset_s = 0.5
        node.bias_calibration_s = 2.0
        node.minimum_bias_samples = 100
        node.initial_range_m = 3.0
        node.preinit_imu = deque()
        node.latest_body_rate = np.zeros(3)
        node.attitude_aiding_enabled = False
        node.publish_state = mock.Mock()
        node.require_vision_ready = False
        return node

    def test_initialized_filter_keeps_recent_imu_window(self):
        node = self.node()

        for stamp in np.arange(1.01, 4.01, 0.01):
            node.imu_callback(imu(stamp))

        cached_stamps = [message.header.stamp.to_sec() for message in node.preinit_imu]
        self.assertGreaterEqual(cached_stamps[0], 1.5)
        self.assertAlmostEqual(cached_stamps[-1], 4.0, places=6)
        self.assertGreaterEqual(cached_stamps[-1] - cached_stamps[0], 2.49)
        self.assertEqual(node.estimator.propagate_calls, 300)

    def test_long_feature_gap_reinitializes_from_cached_stationary_imu(self):
        node = self.node()
        gyro_bias = (0.01, -0.02, 0.03)
        acceleration = (0.1, -0.1, 9.85665)
        for stamp in np.arange(1.01, 3.52, 0.01):
            node.imu_callback(imu(stamp, gyro_bias, acceleration))

        node.feature_callback(feature(3.51))

        self.assertEqual(node.estimator.reset_calls, 1)
        self.assertEqual(len(node.estimator.initialize_calls), 1)
        initialized = node.estimator.initialize_calls[0]
        np.testing.assert_allclose(initialized[3], gyro_bias, atol=1e-12)
        np.testing.assert_allclose(initialized[4], (0.1, -0.1, 0.05), atol=1e-12)
        self.assertTrue(node.estimator.initialized)
        self.assertGreaterEqual(len(node.preinit_imu), 250)
        node.publish_state.assert_called_with(True)


class TestVelocityAidingTimestamp(unittest.TestCase):
    @staticmethod
    def node():
        node = DCEKFNode.__new__(DCEKFNode)
        node.estimator_lock = threading.RLock()
        node.estimator = FakeEstimator(0.9, stamp=1.0)
        node.velocity_aiding_enabled = True
        node.velocity_radial_only = True
        node.velocity_transverse_measurement_std_mps = 1.0
        node.velocity_transverse_activation_range_m = 0.0
        node.velocity_measurement_std_mps = 0.1
        node.velocity_enforce_innovation_gate = False
        node.velocity_max_residual_mps = 3.0
        node.velocity_max_future_lead_s = 0.02
        node.publish_state = mock.Mock()
        return node

    @staticmethod
    def velocity(stamp):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=Stamp(stamp)),
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=1.0, y=2.0, z=3.0)
            ),
        )

    def test_one_sensor_frame_future_velocity_uses_latest_imu_state(self):
        node = self.node()

        node.velocity_callback(self.velocity(1.008))

        self.assertEqual(
            node.estimator.velocity_calls,
            [(1.0, (1.0, 2.0, 3.0), True, 1.0)],
        )
        node.publish_state.assert_called_once_with(False)

    def test_excessive_future_velocity_is_rejected(self):
        node = self.node()

        with mock.patch.object(MODULE.rospy, "logwarn_throttle"):
            node.velocity_callback(self.velocity(1.03))

        self.assertEqual(node.estimator.velocity_calls, [])
        node.publish_state.assert_not_called()

    def test_transverse_stabilizer_waits_until_activation_range(self):
        node = self.node()
        node.velocity_transverse_activation_range_m = 30.0

        node.velocity_callback(self.velocity(1.0))

        self.assertEqual(
            node.estimator.velocity_calls,
            [(1.0, (1.0, 2.0, 3.0), True, math.inf)],
        )


class TestVisionReadyGate(unittest.TestCase):
    @staticmethod
    def node():
        node = DCEKFNode.__new__(DCEKFNode)
        node.estimator_lock = threading.RLock()
        node.estimator = FakeEstimator(0.0)
        node.estimator.reset()
        node.feature_loss_reset_s = 0.5
        node.bias_calibration_s = 2.0
        node.minimum_bias_samples = 100
        node.initial_range_m = 3.0
        node.preinit_imu = deque()
        node.latest_body_rate = np.zeros(3)
        node.attitude_aiding_enabled = False
        node.require_vision_ready = True
        node.vision_ready = False
        node.vision_ready_cutoff_stamp_s = None
        node.frozen_initial_gyro_bias = None
        node.frozen_initial_accel_bias = None
        node.publish_state = mock.Mock()
        return node

    def test_pre_ready_features_are_ignored(self):
        node = self.node()
        node.initialize_from_feature = mock.Mock(return_value=True)

        node.feature_callback(feature(3.0))

        node.initialize_from_feature.assert_not_called()
        node.publish_state.assert_not_called()

    def test_first_complete_preflight_window_freezes_and_is_not_overwritten(self):
        node = self.node()
        gyro_bias = (0.01, -0.02, 0.03)
        acceleration = (0.1, -0.1, 9.85665)
        for stamp in np.arange(0.01, 2.12, 0.01):
            node.imu_callback(imu(stamp, gyro_bias, acceleration))

        frozen_gyro = node.frozen_initial_gyro_bias.copy()
        frozen_accel = node.frozen_initial_accel_bias.copy()
        np.testing.assert_allclose(frozen_gyro, gyro_bias, atol=1e-12)
        np.testing.assert_allclose(frozen_accel, (0.1, -0.1, 0.05), atol=1e-12)

        for stamp in np.arange(2.12, 4.01, 0.01):
            node.imu_callback(imu(stamp, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
        np.testing.assert_array_equal(node.frozen_initial_gyro_bias, frozen_gyro)
        np.testing.assert_array_equal(node.frozen_initial_accel_bias, frozen_accel)

    def test_ready_is_fail_closed_until_prior_exists(self):
        node = self.node()

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=Stamp(10.0)):
            node.vision_ready_callback(SimpleNamespace(data=True))

        self.assertFalse(node.vision_ready)
        self.assertIsNone(node.vision_ready_cutoff_stamp_s)

    def test_ready_cutoff_rejects_queued_features_and_repeated_true_is_idempotent(self):
        node = self.node()
        node.frozen_initial_gyro_bias = np.zeros(3)
        node.frozen_initial_accel_bias = np.zeros(3)
        node.initialize_from_feature = mock.Mock(return_value=True)

        with mock.patch.object(
            MODULE.rospy.Time, "now", side_effect=[Stamp(10.0), Stamp(20.0)]
        ):
            node.vision_ready_callback(SimpleNamespace(data=True))
            node.vision_ready_callback(SimpleNamespace(data=True))

        self.assertEqual(node.vision_ready_cutoff_stamp_s, 10.0)
        node.feature_callback(feature(10.0))
        node.initialize_from_feature.assert_not_called()
        node.feature_callback(feature(10.01))
        node.initialize_from_feature.assert_called_once()
        node.publish_state.assert_called_once_with(True)

    def test_ready_revoke_resets_initialized_filter(self):
        node = self.node()
        node.estimator.history = [object()]
        node.vision_ready = True
        node.vision_ready_cutoff_stamp_s = 10.0

        node.vision_ready_callback(SimpleNamespace(data=False))

        self.assertFalse(node.vision_ready)
        self.assertIsNone(node.vision_ready_cutoff_stamp_s)
        self.assertFalse(node.estimator.initialized)
        self.assertEqual(node.estimator.reset_calls, 2)

    def test_gated_initialization_uses_frozen_prior(self):
        node = self.node()
        node.frozen_initial_gyro_bias = np.array([0.01, -0.02, 0.03])
        node.frozen_initial_accel_bias = np.array([0.1, -0.1, 0.05])
        node.preinit_imu.extend([imu(3.0), imu(3.01), imu(3.02)])

        self.assertTrue(node.initialize_from_feature(feature(3.01)))

        initialized = node.estimator.initialize_calls[0]
        np.testing.assert_array_equal(initialized[3], node.frozen_initial_gyro_bias)
        np.testing.assert_array_equal(initialized[4], node.frozen_initial_accel_bias)

    def test_real_launch_requires_explicit_vision_ready(self):
        launch_dir = SCRIPT.parents[1] / "launch"
        launch = (launch_dir / "real_intercept.launch").read_text(encoding="ascii")
        config = (SCRIPT.parents[1] / "config" / "estimator_real.yaml").read_text(
            encoding="ascii"
        )

        self.assertIn('/pslos/flight/vision_ready', launch)
        self.assertIn('require_vision_ready: true', config)
        self.assertIn('gravity_align_attitude: false', config)


if __name__ == "__main__":
    unittest.main()
