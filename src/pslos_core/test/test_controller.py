#!/usr/bin/env python3
import math
import unittest
from dataclasses import replace

import numpy as np

from pslos_core.controller import (
    HARD_LONG_AXIS_BARRIER,
    LYAPUNOV_CORRECTED,
    PAPER_EQUATION_15,
    PAPER_LITERAL,
    SYMMETRIC_GRADIENT,
    PSLOSController,
    PSLOSParameters,
    SOFT_LONG_AXIS_RECOVERY,
    evaluate_pslos,
)
from pslos_core.geometry import skew


def rotation_increment(rotation_vector):
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.eye(3) + skew(rotation_vector)
    axis_skew = skew(rotation_vector / angle)
    return (
        np.eye(3)
        + math.sin(angle) * axis_skew
        + (1.0 - math.cos(angle)) * axis_skew.dot(axis_skew)
    )


class TestController(unittest.TestCase):
    def setUp(self):
        self.parameters = PSLOSParameters(
            alpha_lon_rad=math.radians(55.0),
            alpha_lat_rad=math.radians(1.0),
            c1=0.6,
            c2=1.0,
            c_omega=1.2,
            mass=1.0,
        )

    def evaluate(self, los, mode=PAPER_LITERAL):
        return evaluate_pslos(
            los,
            20.0,
            np.eye(3),
            np.eye(3),
            self.parameters,
            mode,
        )

    def test_a_f03_image_boundary_is_equation_12_projection(self):
        alpha = self.parameters.alpha_lat_rad
        x_center = math.tan(alpha)
        y = 1.0
        x_off_center = math.tan(alpha) * math.sqrt(1.0 + y * y)
        self.assertGreater(x_off_center, x_center)
        ray = np.array([x_off_center, y, 1.0])
        ray /= np.linalg.norm(ray)
        self.assertAlmostEqual(abs(ray[0]), math.sin(alpha), places=12)

    def test_a_f04_barrier_reference_values(self):
        result = self.evaluate([0.01, 0.2, 1.0])
        expected_denominator = result.c_long ** 2 - result.z1 ** 2
        self.assertAlmostEqual(result.k_h, result.z1 / expected_denominator, places=12)
        self.assertAlmostEqual(
            result.l1,
            0.5 * math.log(result.c_long ** 2 / expected_denominator),
            places=12,
        )
        self.assertAlmostEqual(result.k_v, result.z2, places=12)

    def test_a_f05_fov_force_is_tangent_to_los(self):
        result = self.evaluate([0.01, 0.2, 1.0])
        self.assertLess(abs(float(np.dot(result.los_world, result.fov_tangent))), 1e-12)

    def test_tight_axis_gain_scales_only_the_narrow_axis_correction(self):
        los = [0.02, 0.1, 1.0]
        baseline = self.evaluate(los, SYMMETRIC_GRADIENT)
        strengthened = evaluate_pslos(
            los,
            20.0,
            np.eye(3),
            np.eye(3),
            replace(self.parameters, tight_axis_gain=4.0),
            SYMMETRIC_GRADIENT,
        )

        self.assertAlmostEqual(strengthened.k_h, baseline.k_h)
        self.assertAlmostEqual(strengthened.k_v, 4.0 * baseline.k_v)
        self.assertGreater(
            abs(strengthened.fov_tangent[0]), abs(baseline.fov_tangent[0])
        )

    def test_a_f06_literal_mirror_gap_and_signed_resolution(self):
        positive_literal = self.evaluate([0.0, 0.2, 1.0], PAPER_LITERAL)
        negative_literal = self.evaluate([0.0, -0.2, 1.0], PAPER_LITERAL)
        self.assertGreater(positive_literal.fov_tangent[1], 0.0)
        self.assertGreater(negative_literal.fov_tangent[1], 0.0)

        positive_signed = self.evaluate([0.0, 0.2, 1.0], SYMMETRIC_GRADIENT)
        negative_signed = self.evaluate([0.0, -0.2, 1.0], SYMMETRIC_GRADIENT)
        self.assertGreater(positive_signed.fov_tangent[1], 0.0)
        self.assertLess(negative_signed.fov_tangent[1], 0.0)

    def test_a_f08_rejects_outside_barrier(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.evaluate([0.0, math.tan(math.radians(60.0)), 1.0])

    def test_engineering_soft_recovery_stays_finite_outside_long_barrier(self):
        los = [0.0, math.tan(math.radians(60.0)), 1.0]
        result = evaluate_pslos(
            los,
            20.0,
            np.eye(3),
            np.eye(3),
            self.parameters,
            SYMMETRIC_GRADIENT,
            SOFT_LONG_AXIS_RECOVERY,
            0.01,
        )

        self.assertLess(result.margin_long, 0.0)
        self.assertTrue(np.all(np.isfinite(result.fov_tangent)))
        self.assertTrue(np.all(np.isfinite(result.omega_los_body)))
        self.assertGreater(result.k_h, 0.0)

    def test_controller_defaults_to_paper_hard_long_barrier(self):
        controller = PSLOSController(self.parameters, np.eye(3))
        self.assertEqual(
            controller.long_axis_barrier_policy, HARD_LONG_AXIS_BARRIER
        )

    def test_controller_soft_long_recovery_computes_bounded_command(self):
        parameters = replace(
            self.parameters, alpha_lon_rad=math.radians(20.0)
        )
        controller = PSLOSController(
            parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
            maximum_closing_speed_mps=3.0,
            maximum_command_acceleration_mps2=2.0,
            long_axis_barrier_policy=SOFT_LONG_AXIS_RECOVERY,
            minimum_long_axis_denominator=0.01,
        )
        los = np.array([0.0, math.sin(math.radians(22.0)), math.cos(math.radians(22.0))])
        result = controller.compute(
            np.array([0.0, 0.0, -5.0]),
            np.zeros(3),
            np.eye(3),
            gravity_force_world=np.zeros(3),
            los_world=los,
        )

        self.assertLess(result.fov.margin_long, 0.0)
        self.assertLessEqual(
            np.linalg.norm(result.corrected_acceleration_world), 2.0 + 1e-12
        )
        self.assertTrue(np.all(np.isfinite(result.desired_body_rate)))

    def test_a_t01_desired_relative_velocity_closes_range(self):
        controller = PSLOSController(
            self.parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
        )
        position = np.array([0.0, 0.0, -3.0])
        result = controller.compute(position, np.zeros(3), np.eye(3))
        self.assertLess(float(np.dot(position, result.desired_relative_velocity)), 0.0)

    def test_a_t02_center_los_has_zero_fov_correction(self):
        result = self.evaluate([0.0, 0.0, 1.0], SYMMETRIC_GRADIENT)
        np.testing.assert_allclose(result.fov_tangent, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(result.omega_los_body, np.zeros(3), atol=1e-12)

    def test_long_range_outer_loop_bounds_speed_and_acceleration(self):
        controller = PSLOSController(
            self.parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
            maximum_closing_speed_mps=5.0,
            maximum_command_acceleration_mps2=2.0,
        )
        result = controller.compute(
            np.array([0.0, 0.0, -100.0]),
            np.zeros(3),
            np.eye(3),
            gravity_force_world=np.zeros(3),
        )

        self.assertAlmostEqual(np.linalg.norm(result.desired_relative_velocity), 5.0)
        self.assertAlmostEqual(np.linalg.norm(result.corrected_acceleration_world), 2.0)
        self.assertTrue(controller.bounded_velocity_mode)

    def test_long_range_outer_loop_stops_accelerating_at_speed_limit(self):
        controller = PSLOSController(
            self.parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
            maximum_closing_speed_mps=5.0,
            maximum_command_acceleration_mps2=2.0,
        )
        result = controller.compute(
            np.array([0.0, 0.0, -100.0]),
            np.array([0.0, 0.0, 5.0]),
            np.eye(3),
        )

        np.testing.assert_allclose(
            result.corrected_acceleration_world, np.zeros(3), atol=1e-12
        )

    def test_long_range_outer_loop_closing_floor_prevents_false_slowdown(self):
        controller = PSLOSController(
            self.parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
            maximum_closing_speed_mps=3.0,
            maximum_command_acceleration_mps2=2.0,
            minimum_closing_range_m=5.0,
        )
        result = controller.compute(
            np.array([0.0, 0.0, -1.0]),
            np.zeros(3),
            np.eye(3),
            gravity_force_world=np.zeros(3),
        )

        self.assertAlmostEqual(np.linalg.norm(result.desired_relative_velocity), 3.0)

    def test_long_range_outer_loop_requires_both_limits(self):
        with self.assertRaisesRegex(ValueError, "enabled together"):
            PSLOSController(
                self.parameters,
                np.eye(3),
                SYMMETRIC_GRADIENT,
                LYAPUNOV_CORRECTED,
                maximum_closing_speed_mps=5.0,
            )

    def test_world_los_rate_guidance_adds_bounded_tangent_acceleration(self):
        controller = PSLOSController(
            self.parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
            maximum_closing_speed_mps=8.0,
            maximum_command_acceleration_mps2=8.0,
            los_rate_guidance_gain=3.0,
            maximum_los_rate_guidance_acceleration_mps2=4.0,
        )
        result = controller.compute(
            np.array([0.0, 0.0, -100.0]),
            np.zeros(3),
            np.eye(3),
            gravity_force_world=np.zeros(3),
            los_rate_world=np.array([0.0, 0.5, 0.3]),
        )

        np.testing.assert_allclose(
            result.los_rate_guidance_acceleration_world,
            np.array([0.0, 4.0, 0.0]),
            atol=1e-12,
        )
        self.assertTrue(result.los_rate_guidance_active)
        self.assertGreater(result.corrected_acceleration_world[1], 0.0)
        self.assertLessEqual(
            np.linalg.norm(result.corrected_acceleration_world), 8.0 + 1e-12
        )

    def test_world_los_rate_guidance_is_opt_in_and_parameter_paired(self):
        baseline = PSLOSController(
            self.parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
            maximum_closing_speed_mps=8.0,
            maximum_command_acceleration_mps2=8.0,
        )
        result = baseline.compute(
            np.array([0.0, 0.0, -100.0]),
            np.zeros(3),
            np.eye(3),
            gravity_force_world=np.zeros(3),
            los_rate_world=np.array([0.0, 0.5, 0.0]),
        )
        np.testing.assert_allclose(
            result.los_rate_guidance_acceleration_world, np.zeros(3)
        )
        self.assertFalse(result.los_rate_guidance_active)

        with self.assertRaisesRegex(ValueError, "enabled together"):
            PSLOSController(
                self.parameters,
                np.eye(3),
                SYMMETRIC_GRADIENT,
                LYAPUNOV_CORRECTED,
                maximum_closing_speed_mps=8.0,
                maximum_command_acceleration_mps2=8.0,
                los_rate_guidance_gain=3.0,
            )

    def test_world_los_rate_guidance_smoothly_reduces_near_target(self):
        controller = PSLOSController(
            self.parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
            maximum_closing_speed_mps=8.0,
            maximum_command_acceleration_mps2=8.0,
            minimum_closing_range_m=8.0,
            los_rate_guidance_gain=3.0,
            maximum_los_rate_guidance_acceleration_mps2=20.0,
            los_rate_guidance_terminal_start_range_m=16.0,
            los_rate_guidance_terminal_minimum_scale=0.25,
        )
        far = controller.compute(
            np.array([0.0, 0.0, -16.0]), np.zeros(3), np.eye(3),
            gravity_force_world=np.zeros(3), los_rate_world=np.array([0.0, 0.1, 0.0]),
        )
        middle = controller.compute(
            np.array([0.0, 0.0, -8.0]), np.zeros(3), np.eye(3),
            gravity_force_world=np.zeros(3), los_rate_world=np.array([0.0, 0.1, 0.0]),
        )
        near = controller.compute(
            np.array([0.0, 0.0, -0.01]), np.zeros(3), np.eye(3),
            gravity_force_world=np.zeros(3), los_rate_world=np.array([0.0, 0.1, 0.0]),
        )

        self.assertAlmostEqual(far.los_rate_guidance_range_scale, 1.0)
        self.assertAlmostEqual(middle.los_rate_guidance_range_scale, 0.625)
        self.assertAlmostEqual(near.los_rate_guidance_range_scale, 0.25, places=3)
        self.assertGreater(
            np.linalg.norm(far.los_rate_guidance_acceleration_world),
            np.linalg.norm(middle.los_rate_guidance_acceleration_world),
        )

    def test_radial_velocity_feedback_ignores_unobservable_transverse_velocity(self):
        controller = PSLOSController(
            self.parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
            maximum_closing_speed_mps=8.0,
            maximum_command_acceleration_mps2=8.0,
            radial_velocity_feedback_only=True,
        )
        result = controller.compute(
            np.array([0.0, 0.0, -100.0]),
            np.array([0.0, 6.0, 8.0]),
            np.eye(3),
        )

        np.testing.assert_allclose(result.velocity_error, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(
            result.corrected_acceleration_world, np.zeros(3), atol=1e-12
        )

    def test_explicit_measured_los_controls_fov_without_changing_range_state(self):
        controller = PSLOSController(
            self.parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
        )
        position = np.array([-10.0, -2.0, 0.0])
        measured_los = np.array([0.0, 0.0, 1.0])

        result = controller.compute(
            position,
            np.zeros(3),
            np.eye(3),
            gravity_force_world=np.zeros(3),
            los_world=measured_los,
        )

        np.testing.assert_allclose(result.fov.los_world, measured_los)
        self.assertAlmostEqual(result.fov.margin_tight, result.fov.c_tight)
        np.testing.assert_allclose(result.fov.fov_tangent, np.zeros(3), atol=1e-12)

    def test_engineering_attitude_tracking_gain_does_not_scale_fov_rate(self):
        baseline = PSLOSController(self.parameters, np.eye(3))
        boosted = PSLOSController(
            self.parameters,
            np.eye(3),
            attitude_tracking_gain=3.0 * self.parameters.c_omega,
        )
        position = np.array([0.0, 2.0, -10.0])
        velocity = np.zeros(3)
        baseline_result = baseline.compute(position, velocity, np.eye(3))
        boosted_result = boosted.compute(position, velocity, np.eye(3))

        np.testing.assert_allclose(
            boosted_result.fov.omega_los_body,
            baseline_result.fov.omega_los_body,
            atol=1e-12,
        )
        baseline_attitude_rate = (
            baseline_result.desired_body_rate
            - baseline_result.fov.omega_los_body
        )
        boosted_attitude_rate = (
            boosted_result.desired_body_rate
            - boosted_result.fov.omega_los_body
        )
        np.testing.assert_allclose(
            boosted_attitude_rate,
            3.0 * baseline_attitude_rate,
            atol=1e-12,
        )

    def test_engineering_long_axis_fov_rate_weight_preserves_tight_feedback(self):
        baseline = PSLOSController(self.parameters, np.eye(3))
        weighted = PSLOSController(
            self.parameters,
            np.eye(3),
            long_axis_fov_angular_rate_weight=0.2,
        )
        position = np.array([0.0, 2.0, -10.0])
        velocity = np.zeros(3)
        baseline_result = baseline.compute(position, velocity, np.eye(3))
        weighted_result = weighted.compute(position, velocity, np.eye(3))

        np.testing.assert_allclose(
            weighted_result.fov.omega_los_body,
            baseline_result.fov.omega_los_body,
            atol=1e-12,
        )
        rate_change = weighted_result.desired_body_rate - baseline_result.desired_body_rate
        expected_long_rate = baseline_result.fov.omega_los_body
        np.testing.assert_allclose(
            rate_change,
            -0.8 * expected_long_rate,
            atol=1e-12,
        )

        tight_only_position = np.array([2.0, 0.0, -10.0])
        tight_baseline = baseline.compute(tight_only_position, velocity, np.eye(3))
        tight_weighted = weighted.compute(tight_only_position, velocity, np.eye(3))
        np.testing.assert_allclose(
            tight_weighted.desired_body_rate,
            tight_baseline.desired_body_rate,
            atol=1e-12,
        )

    def test_engineering_long_axis_fov_rate_weight_is_bounded(self):
        for weight in (-0.01, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                PSLOSController(
                    self.parameters,
                    np.eye(3),
                    long_axis_fov_angular_rate_weight=weight,
                )

    def test_adaptive_long_axis_weight_prioritizes_speed_then_recovers_smoothly(self):
        controller = PSLOSController(
            self.parameters,
            np.eye(3),
            long_axis_fov_angular_rate_weight=0.2,
            long_axis_fov_recovery_start_rad=math.radians(30.0),
            long_axis_fov_full_recovery_rad=math.radians(50.0),
        )
        velocity = np.zeros(3)

        def result_at_angle(angle_deg):
            angle = math.radians(angle_deg)
            position = np.array(
                [0.0, 10.0 * math.sin(angle), -10.0 * math.cos(angle)]
            )
            return controller.compute(position, velocity, np.eye(3))

        interior = result_at_angle(20.0)
        transition = result_at_angle(40.0)
        recovery = result_at_angle(50.0)
        self.assertAlmostEqual(
            interior.effective_long_axis_fov_angular_rate_weight, 0.2
        )
        self.assertAlmostEqual(
            transition.effective_long_axis_fov_angular_rate_weight, 0.6
        )
        self.assertAlmostEqual(
            recovery.effective_long_axis_fov_angular_rate_weight, 1.0
        )

    def test_adaptive_long_axis_recovery_angles_are_validated(self):
        invalid = (
            (-1.0, 0.0),
            (math.radians(20.0), 0.0),
            (math.radians(30.0), math.radians(20.0)),
            (math.radians(30.0), math.radians(60.0)),
        )
        for start, full in invalid:
            with self.assertRaises(ValueError):
                PSLOSController(
                    self.parameters,
                    np.eye(3),
                    long_axis_fov_recovery_start_rad=start,
                    long_axis_fov_full_recovery_rad=full,
                )

    def test_a_t04_four_direction_los_steps_recover(self):
        long_angle = math.radians(40.0)
        tight_angle = math.radians(4.0)
        cases = (
            (np.array([0.0, math.sin(long_angle), math.cos(long_angle)]), "long"),
            (np.array([0.0, -math.sin(long_angle), math.cos(long_angle)]), "long"),
            (np.array([math.sin(tight_angle), 0.0, math.cos(tight_angle)]), "tight"),
            (np.array([-math.sin(tight_angle), 0.0, math.cos(tight_angle)]), "tight"),
        )
        for los, axis in cases:
            rotation_world_body = np.eye(3)
            initial = evaluate_pslos(
                los,
                20.0,
                rotation_world_body,
                np.eye(3),
                self.parameters,
                SYMMETRIC_GRADIENT,
            )
            for _ in range(400):
                result = evaluate_pslos(
                    los,
                    20.0,
                    rotation_world_body,
                    np.eye(3),
                    self.parameters,
                    SYMMETRIC_GRADIENT,
                )
                rotation_world_body = rotation_world_body.dot(
                    rotation_increment(result.omega_los_body * 0.005)
                )
            final = evaluate_pslos(
                los,
                20.0,
                rotation_world_body,
                np.eye(3),
                self.parameters,
                SYMMETRIC_GRADIENT,
            )
            initial_error = abs(initial.s_long if axis == "long" else initial.s_tight)
            final_error = abs(final.s_long if axis == "long" else final.s_tight)
            self.assertLess(final_error, initial_error)
            self.assertGreater(final.margin_long, 0.0)
            self.assertGreaterEqual(final.margin_tight, 0.0)

    def test_equation_15_literal_algebra(self):
        controller = PSLOSController(
            self.parameters, np.eye(3), PAPER_LITERAL, PAPER_EQUATION_15
        )
        position = np.array([0.0, 0.0, -10.0])
        velocity = np.array([0.0, 0.0, 1.0])
        result = controller.compute(position, velocity, np.eye(3))
        desired_velocity = -self.parameters.c1 * position
        velocity_error = velocity - desired_velocity
        bracket = (
            -self.parameters.c1 * velocity
            - self.parameters.c2 * velocity_error
            - position
            - result.fov.fov_tangent
        )
        expected = -np.array([0.0, 0.0, 9.80665]) - self.parameters.mass * bracket
        np.testing.assert_allclose(result.paper_force_world, expected, atol=1e-12)

    def test_paper_equation_15_outer_sign_increases_lyapunov_in_1d(self):
        controller = PSLOSController(
            self.parameters, np.eye(3), PAPER_LITERAL, PAPER_EQUATION_15
        )
        position = np.array([0.0, 0.0, -10.0])
        velocity = np.zeros(3)
        result = controller.compute(
            position,
            velocity,
            np.eye(3),
            gravity_force_world=np.zeros(3),
        )
        z4 = result.velocity_error
        paper_acceleration = result.paper_force_world / self.parameters.mass
        lyapunov_derivative = float(
            np.dot(position, velocity)
            + np.dot(z4, paper_acceleration + self.parameters.c1 * velocity)
        )
        self.assertGreater(lyapunov_derivative, 0.0)

    def test_lyapunov_corrected_force_decreases_lyapunov_in_1d(self):
        controller = PSLOSController(
            self.parameters,
            np.eye(3),
            SYMMETRIC_GRADIENT,
            LYAPUNOV_CORRECTED,
        )
        position = np.array([0.0, 0.0, -10.0])
        velocity = np.zeros(3)
        result = controller.compute(
            position,
            velocity,
            np.eye(3),
            gravity_force_world=np.zeros(3),
        )
        z4 = result.velocity_error
        lyapunov_derivative = float(
            np.dot(position, velocity)
            + np.dot(
                z4,
                result.corrected_acceleration_world + self.parameters.c1 * velocity,
            )
        )
        self.assertLess(lyapunov_derivative, 0.0)
        np.testing.assert_allclose(
            result.selected_force_world, result.corrected_force_world, atol=1e-12
        )

    def test_corrected_force_rejects_literal_mirror_mode(self):
        with self.assertRaisesRegex(ValueError, "requires symmetric_gradient"):
            PSLOSController(
                self.parameters,
                np.eye(3),
                PAPER_LITERAL,
                LYAPUNOV_CORRECTED,
            )


if __name__ == "__main__":
    unittest.main()
