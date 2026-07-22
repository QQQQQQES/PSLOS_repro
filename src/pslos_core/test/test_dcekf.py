#!/usr/bin/env python3
import unittest

import numpy as np

from pslos_core.dcekf import (
    B_ACC,
    B_GYR,
    DelayCompensatedEKF,
    EKFParameters,
    FEATURE,
    P_R,
    Q,
    V_R,
    attitude_residual,
    delta_quaternion_xyzw,
)


class TestDelayCompensatedEKF(unittest.TestCase):
    def make_filter(self):
        parameters = EKFParameters(
            max_step_s=0.02,
            history_sync_tolerance_s=0.001,
            innovation_gate_chi2=1e6,
        )
        estimator = DelayCompensatedEKF(parameters, np.eye(3))
        estimator.initialize(0.0, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0], 20.0)
        return estimator

    @staticmethod
    def stationary_imu():
        return np.zeros(3), np.array([0.0, 0.0, 9.80665])

    def propagate_to(self, estimator, end_stamp, step=0.01):
        gyro, acceleration = self.stationary_imu()
        while estimator.stamp < end_stamp - 1e-12:
            estimator.propagate(estimator.stamp + step, gyro, acceleration)

    def test_stationary_propagation(self):
        estimator = self.make_filter()
        initial_state = estimator.state
        self.propagate_to(estimator, 0.10)
        np.testing.assert_allclose(estimator.state[P_R], initial_state[P_R], atol=1e-8)
        np.testing.assert_allclose(estimator.state[V_R], np.zeros(3), atol=1e-8)
        np.testing.assert_allclose(estimator.state[FEATURE], np.zeros(2), atol=1e-8)

    def test_image_feature_propagation_uses_previous_state_depth(self):
        estimator = self.make_filter()
        state = estimator.state
        state[P_R] = [0.0, 0.0, -20.0]
        state[V_R] = [1.0, -0.5, 2.0]
        state[FEATURE] = [0.1, -0.06]
        gyro = np.array([0.02, -0.03, 0.04])
        specific_force = np.array([0.3, -0.4, 10.30665])
        dt = 0.02

        x, y = state[FEATURE]
        previous_depth = 20.0
        translation_jacobian = np.array(
            [
                [-1.0 / previous_depth, 0.0, x / previous_depth],
                [0.0, -1.0 / previous_depth, y / previous_depth],
            ]
        )
        rotation_jacobian = np.array(
            [
                [x * y, -(1.0 + x * x), y],
                [1.0 + y * y, -x * y, -x],
            ]
        )
        expected_feature = state[FEATURE] + dt * (
            translation_jacobian.dot(state[V_R])
            + rotation_jacobian.dot(gyro)
        )

        propagated = estimator._propagate_state(
            state, gyro, specific_force, dt
        )

        np.testing.assert_allclose(
            propagated[FEATURE], expected_feature, atol=1e-12
        )

    def test_delayed_update_matches_on_time_update_then_replay(self):
        on_time = self.make_filter()
        delayed = self.make_filter()

        self.propagate_to(on_time, 0.05)
        update_on_time = on_time.update_feature(0.05, [0.04, -0.02])
        self.assertTrue(update_on_time.accepted)
        self.propagate_to(on_time, 0.10)

        self.propagate_to(delayed, 0.10)
        update_delayed = delayed.update_feature(0.05, [0.04, -0.02])
        self.assertTrue(update_delayed.accepted)
        self.assertEqual(update_delayed.replayed_steps, 5)
        np.testing.assert_allclose(delayed.state, on_time.state, atol=1e-9)
        np.testing.assert_allclose(delayed.covariance, on_time.covariance, atol=1e-9)

    def test_too_old_measurement_is_rejected(self):
        estimator = self.make_filter()
        self.propagate_to(estimator, 0.70)
        result = estimator.update_feature(0.05, [0.0, 0.0])
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "no_history_at_feature_stamp")

    def test_initial_range_is_a_prior_with_large_uncertainty(self):
        estimator = self.make_filter()
        self.assertAlmostEqual(np.linalg.norm(estimator.state[P_R]), 20.0, places=9)
        self.assertGreater(estimator.position_standard_deviation(), 20.0)

    def test_imu_gap_is_split_without_losing_current_timestamp(self):
        estimator = self.make_filter()
        gyro, acceleration = self.stationary_imu()
        estimator.propagate(0.055, gyro, acceleration)
        self.assertAlmostEqual(estimator.stamp, 0.055, places=12)
        self.assertEqual(len(estimator.history), 4)
        self.assertTrue(
            all(
                snapshot.step_from_previous is None
                or snapshot.step_from_previous.dt <= estimator.parameters.max_step_s
                for snapshot in estimator.history
            )
        )

    def test_sparse_jacobian_matches_finite_difference_for_non_attitude_blocks(self):
        estimator = self.make_filter()
        state = estimator.state
        state[V_R] = [0.3, -0.2, 0.1]
        state[FEATURE] = [0.04, -0.03]
        gyro = np.array([0.02, -0.01, 0.03])
        acceleration = np.array([0.1, -0.1, 9.80665])
        dt = 0.008
        nominal = estimator._propagate_state(state, gyro, acceleration, dt)
        sparse = estimator._state_jacobian(state, gyro, acceleration, dt, nominal)

        epsilon = estimator.parameters.jacobian_epsilon
        for index in range(P_R.start, len(state)):
            perturbed = state.copy()
            perturbed[index] += epsilon
            propagated = estimator._propagate_state(perturbed, gyro, acceleration, dt)
            numerical_column = (propagated - nominal) / epsilon
            np.testing.assert_allclose(
                sparse[:, index], numerical_column, atol=2e-5, rtol=2e-4
            )

    def test_imu_process_noise_stays_tangent_and_preserves_cross_covariance(self):
        estimator = self.make_filter()
        state = estimator.state
        state[FEATURE] = [0.08, -0.04]
        gyro = np.array([0.02, -0.01, 0.03])
        acceleration = np.array([0.1, -0.2, 9.90665])
        dt = 0.01
        nominal = estimator._propagate_state(state, gyro, acceleration, dt)

        covariance = estimator._process_covariance(state, dt, nominal)

        radial_variance = nominal[Q].dot(covariance[Q, Q]).dot(nominal[Q])
        self.assertAlmostEqual(radial_variance, 0.0, places=14)
        expected_position_velocity_cross = (
            np.eye(3) * 0.5 * estimator.parameters.accel_noise_std ** 2 * dt ** 3
        )
        np.testing.assert_allclose(
            covariance[P_R, V_R], expected_position_velocity_cross, atol=1e-15
        )
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
        self.assertGreater(np.linalg.norm(covariance[Q, FEATURE]), 0.0)

    def test_target_maneuver_noise_is_opt_in_and_increases_relative_motion_covariance(self):
        baseline = self.make_filter()
        maneuvering = DelayCompensatedEKF(
            EKFParameters(target_accel_process_std=1.5), np.eye(3)
        )
        maneuvering.initialize(0.0, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0], 20.0)
        dt = 0.01
        gyro = np.zeros(3)
        acceleration = np.array([0.0, 0.0, 9.80665])
        baseline_nominal = baseline._propagate_state(
            baseline.state, gyro, acceleration, dt
        )
        maneuver_nominal = maneuvering._propagate_state(
            maneuvering.state, gyro, acceleration, dt
        )
        baseline_q = baseline._process_covariance(
            baseline.state, dt, baseline_nominal
        )
        maneuver_q = maneuvering._process_covariance(
            maneuvering.state, dt, maneuver_nominal
        )

        expected_velocity_increment = np.eye(3) * (1.5 * dt) ** 2
        np.testing.assert_allclose(
            maneuver_q[V_R, V_R] - baseline_q[V_R, V_R],
            expected_velocity_increment,
            atol=1e-15,
        )

    def test_initial_biases_are_explicit_state_priors(self):
        parameters = EKFParameters()
        estimator = DelayCompensatedEKF(parameters, np.eye(3))
        estimator.initialize(
            0.0,
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
            20.0,
            initial_gyro_bias=[0.01, -0.02, 0.03],
            initial_accel_bias=[0.1, -0.2, 0.3],
        )
        np.testing.assert_allclose(estimator.state[B_GYR], [0.01, -0.02, 0.03])
        np.testing.assert_allclose(estimator.state[B_ACC], [0.1, -0.2, 0.3])

    def test_reset_discards_invalid_estimate(self):
        estimator = self.make_filter()
        estimator.reset()
        self.assertFalse(estimator.initialized)
        self.assertEqual(estimator.measurement_count, 0)

    def test_failed_delayed_replay_does_not_partially_modify_history(self):
        estimator = self.make_filter()
        self.propagate_to(estimator, 0.10)
        before_state = estimator.state
        before_covariance = estimator.covariance
        delayed_snapshot = min(
            estimator.history, key=lambda snapshot: abs(snapshot.stamp - 0.05)
        )
        delayed_snapshot.covariance[P_R.stop - 1, FEATURE.start] = 0.01
        delayed_snapshot.covariance[FEATURE.start, P_R.stop - 1] = 0.01

        result = estimator.update_feature(0.05, [1.0, 0.0])

        self.assertFalse(result.accepted)
        self.assertTrue(result.reason.startswith("replay_"))
        np.testing.assert_allclose(estimator.state, before_state)
        np.testing.assert_allclose(estimator.covariance, before_covariance)

    def test_attitude_aiding_reduces_rotation_error_without_counting_as_vision(self):
        estimator = self.make_filter()
        estimator.history[-1].state[Q] = delta_quaternion_xyzw([0.0, 0.0, 1.0], 0.2)
        before_error = np.linalg.norm(
            attitude_residual([0.0, 0.0, 0.0, 1.0], estimator.state[Q])
        )
        before_count = estimator.measurement_count

        result = estimator.update_attitude(
            0.0, [0.0, 0.0, 0.0, 1.0], measurement_std=0.01
        )

        self.assertTrue(result.accepted)
        after_error = np.linalg.norm(
            attitude_residual([0.0, 0.0, 0.0, 1.0], estimator.state[Q])
        )
        self.assertLess(after_error, before_error * 0.2)
        self.assertEqual(estimator.measurement_count, before_count)

    def test_attitude_observation_is_invariant_to_quaternion_sign(self):
        positive = self.make_filter()
        negative = self.make_filter()
        rotation = delta_quaternion_xyzw([0.2, -0.1, 0.3], 0.15)
        positive.history[-1].state[Q] = rotation
        negative.history[-1].state[Q] = rotation

        positive_result = positive.update_attitude(0.0, rotation, 0.01)
        negative_result = negative.update_attitude(0.0, -rotation, 0.01)

        self.assertTrue(positive_result.accepted)
        self.assertTrue(negative_result.accepted)
        np.testing.assert_allclose(positive.state, negative.state, atol=1e-9)
        np.testing.assert_allclose(positive.covariance, negative.covariance, atol=1e-9)

    def test_platform_attitude_can_bypass_statistical_gate_but_not_jump_limit(self):
        gated = self.make_filter()
        aided = self.make_filter()
        for estimator in (gated, aided):
            estimator.history[-1].state[Q] = delta_quaternion_xyzw(
                [0.0, 1.0, 0.0], 0.2
            )
            estimator.history[-1].covariance[Q, Q] = np.eye(4) * 1e-12

        gated_result = gated.update_attitude(
            0.0, [0.0, 0.0, 0.0, 1.0], 1e-4
        )
        aided_result = aided.update_attitude(
            0.0,
            [0.0, 0.0, 0.0, 1.0],
            1e-4,
            enforce_innovation_gate=False,
            max_residual_rad=0.3,
        )
        jump_result = aided.update_attitude(
            0.0,
            delta_quaternion_xyzw([1.0, 0.0, 0.0], 0.5),
            1e-4,
            enforce_innovation_gate=False,
            max_residual_rad=0.3,
        )

        self.assertFalse(gated_result.accepted)
        self.assertEqual(gated_result.reason, "innovation_gate")
        self.assertTrue(aided_result.accepted)
        self.assertFalse(jump_result.accepted)
        self.assertEqual(jump_result.reason, "attitude_residual_limit")

    def test_stationary_target_velocity_aiding_corrects_drift_without_counting_as_vision(self):
        estimator = self.make_filter()
        estimator.history[-1].state[V_R] = [0.8, -0.4, 0.2]
        before_count = estimator.measurement_count

        result = estimator.update_velocity(
            0.0,
            [0.0, 0.0, 0.0],
            0.01,
            enforce_innovation_gate=False,
            max_residual_mps=2.0,
        )

        self.assertTrue(result.accepted)
        self.assertLess(np.linalg.norm(estimator.state[V_R]), 0.01)
        self.assertEqual(estimator.measurement_count, before_count)

    def test_radial_velocity_aiding_preserves_transverse_target_motion(self):
        estimator = self.make_filter()
        estimator.history[-1].state[V_R] = [0.8, -0.4, 0.2]
        direction = estimator.state[P_R] / np.linalg.norm(estimator.state[P_R])
        velocity_before = estimator.state[V_R]
        transverse_before = velocity_before - direction * np.dot(
            direction, velocity_before
        )

        result = estimator.update_velocity(
            0.0,
            [0.0, 0.0, 0.0],
            0.01,
            enforce_innovation_gate=False,
            max_residual_mps=2.0,
            radial_only=True,
        )

        velocity_after = estimator.state[V_R]
        transverse_after = velocity_after - direction * np.dot(
            direction, velocity_after
        )
        self.assertTrue(result.accepted)
        self.assertLess(abs(np.dot(direction, velocity_after)), 0.01)
        np.testing.assert_allclose(transverse_after, transverse_before, atol=1e-12)

    def test_radial_velocity_aiding_can_weakly_stabilize_transverse_velocity(self):
        estimator = self.make_filter()
        estimator.history[-1].state[V_R] = [0.8, -0.4, 0.2]
        estimator.history[-1].covariance[V_R, V_R] = np.eye(3) * 0.1 ** 2
        direction = estimator.state[P_R] / np.linalg.norm(estimator.state[P_R])
        velocity_before = estimator.state[V_R]
        transverse_before = velocity_before - direction * np.dot(
            direction, velocity_before
        )

        result = estimator.update_velocity(
            0.0,
            [0.0, 0.0, 0.0],
            0.01,
            enforce_innovation_gate=False,
            radial_only=True,
            transverse_measurement_std=1.0,
        )

        velocity_after = estimator.state[V_R]
        transverse_after = velocity_after - direction * np.dot(
            direction, velocity_after
        )
        self.assertTrue(result.accepted)
        self.assertLess(abs(np.dot(direction, velocity_after)), 0.01)
        self.assertLess(np.linalg.norm(transverse_after), np.linalg.norm(transverse_before))
        self.assertGreater(
            np.linalg.norm(transverse_after),
            0.9 * np.linalg.norm(transverse_before),
        )

    def test_delayed_feature_replay_preserves_velocity_aiding(self):
        on_time = self.make_filter()
        delayed = self.make_filter()
        gyro, acceleration = self.stationary_imu()

        for step in range(1, 11):
            stamp = step * 0.01
            for estimator in (on_time, delayed):
                estimator.propagate(stamp, gyro, acceleration)
                self.assertTrue(
                    estimator.update_velocity(
                        stamp,
                        [0.1, -0.05, 0.0],
                        0.02,
                        enforce_innovation_gate=False,
                    ).accepted
                )
            if step == 5:
                self.assertTrue(on_time.update_feature(stamp, [0.04, -0.02]).accepted)

        result = delayed.update_feature(0.05, [0.04, -0.02])

        self.assertTrue(result.accepted)
        np.testing.assert_allclose(delayed.state, on_time.state, atol=1e-9)
        np.testing.assert_allclose(delayed.covariance, on_time.covariance, atol=1e-9)

    def test_delayed_feature_replay_preserves_radial_velocity_aiding(self):
        on_time = self.make_filter()
        delayed = self.make_filter()
        gyro, acceleration = self.stationary_imu()

        for step in range(1, 11):
            stamp = step * 0.01
            for estimator in (on_time, delayed):
                estimator.propagate(stamp, gyro, acceleration)
                self.assertTrue(
                    estimator.update_velocity(
                        stamp,
                        [0.1, -0.05, 0.0],
                        0.02,
                        enforce_innovation_gate=False,
                        radial_only=True,
                    ).accepted
                )
            if step == 5:
                self.assertTrue(on_time.update_feature(stamp, [0.04, -0.02]).accepted)

        result = delayed.update_feature(0.05, [0.04, -0.02])

        self.assertTrue(result.accepted)
        np.testing.assert_allclose(delayed.state, on_time.state, atol=1e-9)
        np.testing.assert_allclose(delayed.covariance, on_time.covariance, atol=1e-9)

    def test_delayed_feature_replay_preserves_platform_attitude_updates(self):
        on_time = self.make_filter()
        delayed = self.make_filter()
        gyro, acceleration = self.stationary_imu()
        identity = [0.0, 0.0, 0.0, 1.0]

        for step in range(1, 11):
            stamp = step * 0.01
            on_time.propagate(stamp, gyro, acceleration)
            self.assertTrue(on_time.update_attitude(stamp, identity, 0.02).accepted)
            if step == 5:
                self.assertTrue(on_time.update_feature(stamp, [0.04, -0.02]).accepted)

            delayed.propagate(stamp, gyro, acceleration)
            self.assertTrue(delayed.update_attitude(stamp, identity, 0.02).accepted)

        result = delayed.update_feature(0.05, [0.04, -0.02])

        self.assertTrue(result.accepted)
        self.assertEqual(result.replayed_steps, 5)
        np.testing.assert_allclose(delayed.state, on_time.state, atol=1e-9)
        np.testing.assert_allclose(delayed.covariance, on_time.covariance, atol=1e-9)

    def test_a_e06_dropout_predicts_then_expires_without_stale_validity(self):
        estimator = self.make_filter()
        self.assertTrue(estimator.update_feature(0.0, [0.0, 0.0]).accepted)
        self.assertTrue(estimator.update_feature(0.0, [0.0, 0.0]).accepted)
        initial_position_covariance = np.trace(estimator.covariance[P_R, P_R])

        self.propagate_to(estimator, 0.30)
        self.assertAlmostEqual(estimator.vision_age(), 0.30, places=9)
        self.assertGreater(
            np.trace(estimator.covariance[P_R, P_R]), initial_position_covariance
        )
        self.assertTrue(estimator.estimate_valid(3, 50.0, 0.50))

        self.propagate_to(estimator, 0.60)
        self.assertAlmostEqual(estimator.vision_age(), 0.60, places=9)
        self.assertFalse(estimator.estimate_valid(3, 50.0, 0.50))


if __name__ == "__main__":
    unittest.main()
