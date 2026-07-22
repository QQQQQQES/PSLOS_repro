#!/usr/bin/env python3
import math
import unittest

import numpy as np

from pslos_core.command_safety import CommandSafetyParameters, sanitize_command
from pslos_core.rate_damping import (
    BodyRateReversalBooster,
    RateReversalBoostParameters,
)
from pslos_core.tight_fov_guard import (
    PixelAxisGuard,
    PixelAxisGuardParameters,
    TightFOVGuard,
    TightFOVGuardParameters,
    apply_pixel_axis_guard,
    apply_tight_fov_guard,
)


class TestCommandSafety(unittest.TestCase):
    def setUp(self):
        self.parameters = CommandSafetyParameters()

    def test_hover_ratio_maps_to_hover_thrust(self):
        result = sanitize_command(
            [0.0, 0.0, 0.0, 1.0], [0.1, -0.2, 0.3], 1.0, self.parameters
        )
        self.assertTrue(result.valid)
        self.assertFalse(result.saturated)
        self.assertAlmostEqual(result.thrust, 0.5)

    def test_rate_and_thrust_are_limited(self):
        result = sanitize_command(
            [0.0, 0.0, 0.0, 1.0], [2.0, -2.0, 1.0], 2.0, self.parameters
        )
        self.assertTrue(result.valid)
        self.assertTrue(result.saturated)
        np.testing.assert_allclose(result.body_rate_rad_s, [1.2, -1.2, 0.8])
        self.assertAlmostEqual(result.thrust, 0.75)

    def test_tilt_over_limit_is_rejected(self):
        angle = math.radians(40.0)
        result = sanitize_command(
            [0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)],
            [0.0, 0.0, 0.0],
            1.0,
            self.parameters,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "tilt_limit")

    def test_body_rate_mode_can_ignore_unused_attitude_tilt(self):
        angle = math.radians(40.0)
        result = sanitize_command(
            [0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)],
            [0.1, -0.2, 0.3],
            1.0,
            self.parameters,
            enforce_tilt_limit=False,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "accepted")

    def test_non_finite_command_is_rejected(self):
        result = sanitize_command(
            [0.0, 0.0, 0.0, 1.0], [float("nan"), 0.0, 0.0], 1.0, self.parameters
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "non_finite_command")

    def test_non_positive_thrust_is_rejected(self):
        result = sanitize_command(
            [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0.0, self.parameters
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "non_positive_thrust")


class TestBodyRateReversalBooster(unittest.TestCase):
    def setUp(self):
        self.booster = BodyRateReversalBooster(
            RateReversalBoostParameters(
                gains=(0.0, 0.6, 1.2),
                damping_gains=(0.0, 0.2, 0.6),
                filter_tau_s=0.04,
                max_correction_rad_s=(0.35, 0.50, 0.35),
                reset_gap_s=0.20,
            )
        )

    def test_opposite_rotation_boosts_pitch_and_yaw_reversals(self):
        self.booster.update([0.3, 0.2, 0.2], 1.0)
        result = self.booster.compensate([0.1, -0.1, -0.1])

        self.assertAlmostEqual(result.body_rate_rad_s[0], 0.1)
        self.assertAlmostEqual(result.body_rate_rad_s[1], -0.26)
        self.assertAlmostEqual(result.body_rate_rad_s[2], -0.45)
        np.testing.assert_allclose(result.damping_correction_rad_s, [0.0, 0.04, 0.12])
        np.testing.assert_allclose(result.reversal_correction_rad_s, [0.0, 0.12, 0.24])
        np.testing.assert_allclose(result.correction_rad_s, [0.0, 0.16, 0.35])

    def test_same_direction_rotation_receives_only_rate_damping(self):
        self.booster.update([0.0, -0.2, 0.2], 1.0)
        result = self.booster.compensate([0.1, -0.1, 0.1])

        np.testing.assert_allclose(result.reversal_correction_rad_s, np.zeros(3))
        np.testing.assert_allclose(result.correction_rad_s, [0.0, -0.04, 0.12])
        np.testing.assert_allclose(result.body_rate_rad_s, [0.1, -0.06, -0.02])

    def test_tracking_error_feedback_is_continuous_through_zero(self):
        booster = BodyRateReversalBooster(
            RateReversalBoostParameters(
                feedback_mode="tracking_error",
                tracking_error_gains=(0.0, 0.6, 1.2),
                filter_tau_s=0.0,
                max_correction_rad_s=(0.35, 0.50, 0.35),
            )
        )
        booster.update([0.0, 0.2, -0.2], 1.0)

        negative = booster.compensate([0.0, 0.0, -1e-6])
        zero = booster.compensate([0.0, 0.0, 0.0])
        positive = booster.compensate([0.0, 0.0, 1e-6])

        self.assertLess(negative.body_rate_rad_s[2], zero.body_rate_rad_s[2])
        self.assertLess(zero.body_rate_rad_s[2], positive.body_rate_rad_s[2])
        self.assertAlmostEqual(
            positive.body_rate_rad_s[2] - zero.body_rate_rad_s[2],
            2.2e-6,
        )
        np.testing.assert_allclose(
            zero.tracking_error_rad_s, [0.0, -0.2, 0.2]
        )
        np.testing.assert_allclose(
            zero.tracking_error_correction_rad_s, [0.0, -0.12, 0.24]
        )

    def test_tracking_error_feedback_correction_is_bounded(self):
        booster = BodyRateReversalBooster(
            RateReversalBoostParameters(
                feedback_mode="tracking_error",
                tracking_error_gains=(0.0, 0.6, 1.2),
                filter_tau_s=0.0,
                max_correction_rad_s=(0.35, 0.50, 0.35),
            )
        )
        booster.update([0.0, 2.0, 2.0], 1.0)
        result = booster.compensate([0.0, -0.1, -0.1])

        np.testing.assert_allclose(
            result.tracking_error_correction_rad_s, [0.0, -0.5, -0.35]
        )
        np.testing.assert_allclose(result.body_rate_rad_s, [0.0, -0.6, -0.45])
        np.testing.assert_allclose(result.damping_correction_rad_s, np.zeros(3))
        np.testing.assert_allclose(result.reversal_correction_rad_s, np.zeros(3))

    def test_tracking_error_zero_gain_preserves_command(self):
        booster = BodyRateReversalBooster(
            RateReversalBoostParameters(
                feedback_mode="tracking_error",
                tracking_error_gains=(0.0, 0.0, 0.0),
                filter_tau_s=0.0,
            )
        )
        booster.update([0.3, -0.4, 0.5], 1.0)
        command = np.array([-0.2, 0.1, -0.6])

        result = booster.compensate(command)

        np.testing.assert_allclose(result.body_rate_rad_s, command)
        np.testing.assert_allclose(
            result.tracking_error_rad_s, command - np.array([0.3, -0.4, 0.5])
        )
        np.testing.assert_allclose(result.tracking_error_correction_rad_s, np.zeros(3))

    def test_tracking_error_matching_command_and_measurement_has_no_correction(self):
        booster = BodyRateReversalBooster(
            RateReversalBoostParameters(
                feedback_mode="tracking_error",
                tracking_error_gains=(0.2, 0.4, 0.6),
                filter_tau_s=0.0,
            )
        )
        command = np.array([0.2, -0.3, 0.4])
        booster.update(command, 1.0)

        result = booster.compensate(command)

        np.testing.assert_allclose(result.body_rate_rad_s, command)
        np.testing.assert_allclose(result.tracking_error_rad_s, np.zeros(3))
        np.testing.assert_allclose(result.tracking_error_correction_rad_s, np.zeros(3))

    def test_tracking_error_feedback_is_sign_symmetric(self):
        parameters = RateReversalBoostParameters(
            feedback_mode="tracking_error",
            tracking_error_gains=(0.2, 0.4, 0.6),
            filter_tau_s=0.0,
            max_correction_rad_s=(0.35, 0.35, 0.35),
        )
        positive = BodyRateReversalBooster(parameters)
        negative = BodyRateReversalBooster(parameters)
        measured = np.array([0.1, -0.2, 0.3])
        command = np.array([0.4, 0.3, -0.5])
        positive.update(measured, 1.0)
        negative.update(-measured, 1.0)

        positive_result = positive.compensate(command)
        negative_result = negative.compensate(-command)

        np.testing.assert_allclose(
            negative_result.body_rate_rad_s, -positive_result.body_rate_rad_s
        )
        np.testing.assert_allclose(
            negative_result.tracking_error_rad_s,
            -positive_result.tracking_error_rad_s,
        )
        np.testing.assert_allclose(
            negative_result.tracking_error_correction_rad_s,
            -positive_result.tracking_error_correction_rad_s,
        )

    def test_mixed_axis_feedback_keeps_pitch_legacy_and_tracks_yaw(self):
        booster = BodyRateReversalBooster(
            RateReversalBoostParameters(
                gains=(0.0, 0.6, 1.2),
                damping_gains=(0.0, 0.2, 0.6),
                tracking_error_gains=(0.0, 0.6, 1.2),
                axis_feedback_modes=(
                    "legacy_reversal",
                    "legacy_reversal",
                    "tracking_error",
                ),
                filter_tau_s=0.0,
                max_correction_rad_s=(0.35, 0.50, 0.35),
            )
        )
        booster.update([0.0, 1.0, -0.5], 1.0)

        result = booster.compensate([0.0, -0.1, 0.2])

        np.testing.assert_allclose(result.body_rate_rad_s, [0.0, -0.6, 0.55])
        np.testing.assert_allclose(result.correction_rad_s, [0.0, 0.5, -0.35])
        np.testing.assert_allclose(
            result.damping_correction_rad_s, [0.0, 0.2, 0.0]
        )
        np.testing.assert_allclose(
            result.reversal_correction_rad_s, [0.0, 0.6, 0.0]
        )
        np.testing.assert_allclose(
            result.tracking_error_correction_rad_s, [0.0, 0.0, 0.35]
        )

    def test_axis_feedback_modes_fall_back_to_global_mode(self):
        legacy = RateReversalBoostParameters(feedback_mode="legacy_reversal")
        tracking = RateReversalBoostParameters(feedback_mode="tracking_error")

        self.assertEqual(
            legacy.resolved_axis_feedback_modes(), ("legacy_reversal",) * 3
        )
        self.assertEqual(
            tracking.resolved_axis_feedback_modes(), ("tracking_error",) * 3
        )

    def test_legacy_positional_parameter_order_is_preserved(self):
        parameters = RateReversalBoostParameters(
            (0.1, 0.2, 0.3),
            (0.4, 0.5, 0.6),
            0.07,
            (0.2, 0.3, 0.4),
            0.25,
        )

        self.assertEqual(parameters.gains, (0.1, 0.2, 0.3))
        self.assertEqual(parameters.damping_gains, (0.4, 0.5, 0.6))
        self.assertEqual(parameters.filter_tau_s, 0.07)
        self.assertEqual(parameters.max_correction_rad_s, (0.2, 0.3, 0.4))
        self.assertEqual(parameters.reset_gap_s, 0.25)
        self.assertEqual(parameters.feedback_mode, "legacy_reversal")
        self.assertEqual(parameters.tracking_error_gains, (0.0, 0.0, 0.0))
        self.assertIsNone(parameters.axis_feedback_modes)

        existing_extended = RateReversalBoostParameters(
            (0.1, 0.2, 0.3),
            (0.4, 0.5, 0.6),
            0.07,
            (0.2, 0.3, 0.4),
            0.25,
            "tracking_error",
            (0.7, 0.8, 0.9),
        )
        self.assertEqual(existing_extended.feedback_mode, "tracking_error")
        self.assertEqual(existing_extended.tracking_error_gains, (0.7, 0.8, 0.9))
        self.assertIsNone(existing_extended.axis_feedback_modes)

    def test_tight_fov_guard_is_zero_inside_activation_band(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.5),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.25,
        )
        result = apply_tight_fov_guard(
            math.tan(math.radians(0.4)),
            0.0,
            math.radians(1.0),
            [0.1, -0.2, 0.3],
            parameters,
        )

        self.assertFalse(result.active)
        self.assertFalse(result.limited)
        np.testing.assert_allclose(result.body_rate_rad_s, [0.1, -0.2, 0.3])
        np.testing.assert_allclose(result.intervention_body_rate_rad_s, np.zeros(3))
        self.assertGreater(result.measured_margin_tight, 0.0)

    def test_tight_fov_guard_opposes_both_sides_and_is_bounded(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.5),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.25,
        )
        positive = apply_tight_fov_guard(
            math.tan(math.radians(2.0)),
            0.0,
            math.radians(1.0),
            [0.1, -0.2, 0.7],
            parameters,
        )
        negative = apply_tight_fov_guard(
            -math.tan(math.radians(2.0)),
            0.0,
            math.radians(1.0),
            [0.1, -0.2, -0.7],
            parameters,
        )

        self.assertTrue(positive.active)
        self.assertTrue(negative.active)
        self.assertTrue(positive.limited)
        self.assertTrue(negative.limited)
        self.assertAlmostEqual(positive.body_rate_rad_s[2], -0.25)
        self.assertAlmostEqual(negative.body_rate_rad_s[2], 0.25)
        self.assertAlmostEqual(positive.intervention_body_rate_rad_s[2], -0.95)
        self.assertAlmostEqual(negative.intervention_body_rate_rad_s[2], 0.95)
        self.assertLess(positive.measured_margin_tight, 0.0)

    def test_tight_fov_los_rate_feedforward_cancels_external_image_rate(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.5),
            los_rate_feedforward_enabled=True,
            los_rate_feedforward_gain=1.0,
            maximum_los_rate_feedforward_rad_s=0.7,
        )
        result = apply_tight_fov_guard(
            0.0,
            0.0,
            math.radians(1.0),
            [0.1, -0.2, 0.0],
            parameters,
            filtered_s_tight_rate_s_inv=0.20,
            derivative_initialized=True,
            measured_body_yaw_rate_rad_s=0.05,
        )

        self.assertTrue(result.active)
        self.assertTrue(result.los_rate_feedforward_initialized)
        self.assertAlmostEqual(result.external_s_tight_rate_s_inv, 0.15)
        self.assertAlmostEqual(
            result.los_rate_feedforward_yaw_rate_rad_s, -0.15
        )
        self.assertAlmostEqual(result.body_rate_rad_s[2], -0.15)

    def test_tight_fov_los_rate_feedforward_removes_body_rotation_and_limits(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.5),
            los_rate_feedforward_enabled=True,
            los_rate_feedforward_gain=1.0,
            maximum_los_rate_feedforward_rad_s=0.3,
        )
        cancelled = apply_tight_fov_guard(
            0.0,
            0.0,
            math.radians(1.0),
            [0.0, 0.0, 0.1],
            parameters,
            filtered_s_tight_rate_s_inv=0.20,
            derivative_initialized=True,
            measured_body_yaw_rate_rad_s=0.20,
        )
        limited = apply_tight_fov_guard(
            0.0,
            0.0,
            math.radians(1.0),
            [0.0, 0.0, 0.0],
            parameters,
            filtered_s_tight_rate_s_inv=-0.80,
            derivative_initialized=True,
            measured_body_yaw_rate_rad_s=0.0,
        )

        self.assertAlmostEqual(cancelled.external_s_tight_rate_s_inv, 0.0)
        self.assertAlmostEqual(
            cancelled.los_rate_feedforward_yaw_rate_rad_s, 0.0
        )
        self.assertAlmostEqual(cancelled.body_rate_rad_s[2], 0.1)
        self.assertAlmostEqual(limited.body_rate_rad_s[2], 0.3)
        self.assertTrue(limited.limited)

    def test_tight_fov_guard_preserves_moderate_restoring_rate(self):
        parameters = TightFOVGuardParameters()
        result = apply_tight_fov_guard(
            math.tan(math.radians(0.6)),
            0.0,
            math.radians(1.0),
            [0.1, -0.2, -0.1],
            parameters,
        )

        self.assertTrue(result.active)
        self.assertFalse(result.limited)
        self.assertAlmostEqual(result.body_rate_rad_s[2], -0.1)
        np.testing.assert_allclose(result.intervention_body_rate_rad_s, np.zeros(3))

    def test_tight_fov_guard_replaces_outward_or_weak_rate(self):
        parameters = TightFOVGuardParameters()
        outward = apply_tight_fov_guard(
            math.tan(math.radians(0.6)),
            0.0,
            math.radians(1.0),
            [0.0, 0.0, 0.2],
            parameters,
        )
        weak = apply_tight_fov_guard(
            math.tan(math.radians(0.6)),
            0.0,
            math.radians(1.0),
            [0.0, 0.0, -0.01],
            parameters,
        )

        self.assertLess(outward.body_rate_rad_s[2], 0.0)
        self.assertAlmostEqual(outward.body_rate_rad_s[2], weak.body_rate_rad_s[2])

    def test_tight_fov_guard_caps_excessive_restoring_rate(self):
        parameters = TightFOVGuardParameters(maximum_guarded_rate_rad_s=0.25)
        result = apply_tight_fov_guard(
            -math.tan(math.radians(0.6)),
            0.0,
            math.radians(1.0),
            [0.0, 0.0, 0.7],
            parameters,
        )

        self.assertTrue(result.limited)
        self.assertAlmostEqual(result.body_rate_rad_s[2], 0.25)

    def test_tight_fov_guard_rejects_activation_outside_acceptance_limit(self):
        parameters = TightFOVGuardParameters(activation_rad=math.radians(1.0))
        with self.assertRaisesRegex(ValueError, "inside the acceptance limit"):
            apply_tight_fov_guard(
                0.0, 0.0, math.radians(1.0), np.zeros(3), parameters
            )

    def test_tight_fov_guard_rejects_invalid_acceptance_limit(self):
        parameters = TightFOVGuardParameters()
        for acceptance_limit in (math.inf, 0.0, 0.5 * math.pi):
            with self.assertRaisesRegex(ValueError, "acceptance limit"):
                apply_tight_fov_guard(
                    0.0, 0.0, acceptance_limit, np.zeros(3), parameters
                )

    def test_body_rate_prediction_rejects_outward_commands_symmetrically(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.70,
            body_rate_prediction_enabled=True,
            body_rate_prediction_horizon_s=0.15,
            body_rate_deadband_rad_s=0.03,
        )

        positive = apply_tight_fov_guard(
            0.0,
            0.0,
            math.radians(1.0),
            [0.0, 0.0, 0.2],
            parameters,
            measured_body_yaw_rate_rad_s=0.10,
            feature_age_s=0.02,
        )
        negative = apply_tight_fov_guard(
            0.0,
            0.0,
            math.radians(1.0),
            [0.0, 0.0, -0.2],
            parameters,
            measured_body_yaw_rate_rad_s=-0.10,
            feature_age_s=0.02,
        )

        self.assertTrue(positive.body_rate_prediction_active)
        self.assertTrue(negative.body_rate_prediction_active)
        self.assertAlmostEqual(positive.body_rate_rad_s[2], 0.0)
        self.assertAlmostEqual(negative.body_rate_rad_s[2], 0.0)
        self.assertAlmostEqual(positive.body_rate_prediction_rate_rad_s, -0.2)
        self.assertAlmostEqual(negative.body_rate_prediction_rate_rad_s, 0.2)
        self.assertAlmostEqual(positive.body_rate_prediction_interval_s, 0.17)

    def test_body_rate_prediction_preserves_existing_braking_without_amplifying_it(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            maximum_guarded_rate_rad_s=0.70,
            body_rate_prediction_enabled=True,
            body_rate_prediction_horizon_s=0.18,
            body_rate_deadband_rad_s=0.03,
        )

        for measured_rate, command in ((0.10, -0.12), (-0.10, 0.12)):
            with self.subTest(measured_rate=measured_rate):
                result = apply_tight_fov_guard(
                    0.0,
                    0.0,
                    math.radians(1.0),
                    [0.0, 0.0, command],
                    parameters,
                    measured_body_yaw_rate_rad_s=measured_rate,
                    feature_age_s=0.02,
                )

                self.assertTrue(result.body_rate_prediction_active)
                self.assertAlmostEqual(result.body_rate_rad_s[2], command)
                self.assertAlmostEqual(
                    result.intervention_body_rate_rad_s[2], 0.0
                )

    def test_body_rate_prediction_uses_optical_ray_z_and_feature_age(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.70,
            body_rate_prediction_enabled=True,
            body_rate_prediction_horizon_s=0.15,
            body_rate_deadband_rad_s=0.0,
        )
        x_normalized = 0.1
        y_normalized = 0.2
        result = apply_tight_fov_guard(
            x_normalized,
            y_normalized,
            math.radians(1.0),
            np.zeros(3),
            parameters,
            measured_body_yaw_rate_rad_s=0.2,
            feature_age_s=0.04,
        )
        ray_z = 1.0 / math.sqrt(1.0 + x_normalized ** 2 + y_normalized ** 2)
        measured_s = x_normalized * ray_z

        self.assertAlmostEqual(result.rotational_s_tight_rate_s_inv, 0.2 * ray_z)
        self.assertAlmostEqual(
            result.body_rate_predicted_s_tight,
            measured_s + 0.19 * 0.2 * ray_z,
        )

    def test_body_rate_prediction_deadband_and_disabled_mode_preserve_command(self):
        command = np.array([0.1, -0.2, 0.03])
        enabled = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            body_rate_prediction_enabled=True,
            body_rate_deadband_rad_s=0.03,
        )
        disabled = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            body_rate_prediction_enabled=False,
        )

        inside_deadband = apply_tight_fov_guard(
            0.0,
            0.0,
            math.radians(1.0),
            command,
            enabled,
            measured_body_yaw_rate_rad_s=0.029,
            feature_age_s=0.10,
        )
        disabled_result = apply_tight_fov_guard(
            0.0,
            0.0,
            math.radians(1.0),
            command,
            disabled,
            measured_body_yaw_rate_rad_s=0.5,
            feature_age_s=0.10,
        )

        self.assertTrue(inside_deadband.body_rate_prediction_initialized)
        self.assertFalse(inside_deadband.body_rate_prediction_active)
        np.testing.assert_allclose(inside_deadband.body_rate_rad_s, command)
        self.assertFalse(disabled_result.body_rate_prediction_initialized)
        np.testing.assert_allclose(disabled_result.body_rate_rad_s, command)

    def test_body_rate_prediction_brakes_current_restoring_floor_after_crossing(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.70,
            body_rate_prediction_enabled=True,
            body_rate_prediction_horizon_s=0.30,
            body_rate_deadband_rad_s=0.0,
        )
        result = apply_tight_fov_guard(
            math.tan(math.radians(1.1)),
            0.0,
            math.radians(1.0),
            np.zeros(3),
            parameters,
            measured_body_yaw_rate_rad_s=-0.2,
        )

        self.assertLess(result.measured_margin_tight, 0.0)
        self.assertTrue(result.body_rate_prediction_active)
        self.assertLess(result.body_rate_predicted_s_tight, 0.0)
        self.assertAlmostEqual(result.body_rate_rad_s[2], 0.0)

    def test_body_rate_prediction_does_not_brake_while_feature_still_escapes(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            rate_gain=10.0,
            maximum_guarded_rate_rad_s=3.0,
            derivative_enabled=True,
            body_rate_prediction_enabled=True,
            body_rate_prediction_horizon_s=0.12,
            body_rate_deadband_rad_s=0.03,
        )
        result = apply_tight_fov_guard(
            -0.15 / math.sqrt(1.0 - 0.15 * 0.15),
            0.0,
            math.radians(1.0),
            [0.0, 0.0, 1.1],
            parameters,
            filtered_s_tight_rate_s_inv=-0.08,
            raw_s_tight_rate_s_inv=-0.20,
            derivative_initialized=True,
            measured_body_yaw_rate_rad_s=1.2,
            feature_age_s=0.04,
        )

        self.assertGreater(result.body_rate_predicted_s_tight, 0.0)
        self.assertFalse(result.body_rate_prediction_active)
        self.assertAlmostEqual(
            result.body_rate_rad_s[2],
            max(1.1, result.requested_tight_rate_rad_s),
        )

    def test_body_rate_prediction_never_self_excites_a_zero_command(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            maximum_guarded_rate_rad_s=0.70,
            body_rate_prediction_enabled=True,
            body_rate_prediction_horizon_s=0.18,
            body_rate_deadband_rad_s=0.03,
        )

        for measured_rate in (-0.4, -0.1, 0.1, 0.4):
            with self.subTest(measured_rate=measured_rate):
                result = apply_tight_fov_guard(
                    0.0,
                    0.0,
                    math.radians(1.0),
                    np.zeros(3),
                    parameters,
                    measured_body_yaw_rate_rad_s=measured_rate,
                    feature_age_s=0.08,
                )

                self.assertTrue(result.body_rate_prediction_active)
                np.testing.assert_allclose(result.body_rate_rad_s, np.zeros(3))
                np.testing.assert_allclose(
                    result.intervention_body_rate_rad_s, np.zeros(3)
                )

    def test_body_rate_prediction_parameter_validation(self):
        for overrides in (
            {"body_rate_prediction_horizon_s": -0.1},
            {"body_rate_prediction_horizon_s": math.inf},
            {"body_rate_deadband_rad_s": -0.1},
            {"body_rate_deadband_rad_s": math.inf},
        ):
            with self.assertRaisesRegex(ValueError, "body-rate"):
                apply_tight_fov_guard(
                    0.0,
                    0.0,
                    math.radians(1.0),
                    np.zeros(3),
                    TightFOVGuardParameters(**overrides),
                )

    @staticmethod
    def make_derivative_guard(**overrides):
        values = {
            "activation_rad": math.radians(0.25),
            "rate_gain": 30.0,
            "maximum_guarded_rate_rad_s": 0.70,
            "derivative_enabled": True,
            "prediction_horizon_s": 0.05,
            "derivative_filter_tau_s": 0.0,
            "derivative_reset_gap_s": 0.12,
            "maximum_raw_derivative_s_inv": 0.25,
            "maximum_derivative_rate_rad_s": 0.20,
        }
        values.update(overrides)
        return TightFOVGuard(
            math.radians(1.0), TightFOVGuardParameters(**values)
        )

    def test_derivative_guard_leads_outward_motion_on_both_sides(self):
        positive = self.make_derivative_guard()
        positive.apply(math.tan(math.radians(0.20)), 0.0, 1.00, [0.0, 0.0, 0.0])
        positive_result = positive.apply(
            math.tan(math.radians(0.40)), 0.0, 1.05, [0.0, 0.0, 0.0]
        )
        negative = self.make_derivative_guard()
        negative.apply(-math.tan(math.radians(0.20)), 0.0, 1.00, [0.0, 0.0, 0.0])
        negative_result = negative.apply(
            -math.tan(math.radians(0.40)), 0.0, 1.05, [0.0, 0.0, 0.0]
        )

        self.assertTrue(positive_result.derivative_initialized)
        self.assertGreater(positive_result.derivative_rate_rad_s, 0.0)
        self.assertLess(positive_result.body_rate_rad_s[2], 0.0)
        self.assertGreater(negative_result.body_rate_rad_s[2], 0.0)
        self.assertAlmostEqual(
            positive_result.body_rate_rad_s[2],
            -negative_result.body_rate_rad_s[2],
        )

    def test_derivative_guard_does_not_weaken_inward_braking(self):
        guard = self.make_derivative_guard()
        guard.apply(math.tan(math.radians(0.70)), 0.0, 1.00, [0.0, 0.0, 0.0])
        result = guard.apply(
            math.tan(math.radians(0.50)), 0.0, 1.05, [0.0, 0.0, 0.0]
        )
        proportional_only = apply_tight_fov_guard(
            math.tan(math.radians(0.50)),
            0.0,
            math.radians(1.0),
            [0.0, 0.0, 0.0],
            TightFOVGuardParameters(
                activation_rad=math.radians(0.25),
                rate_gain=30.0,
                maximum_guarded_rate_rad_s=0.70,
            ),
        )

        self.assertEqual(result.outward_s_tight_rate_s_inv, 0.0)
        self.assertEqual(result.derivative_rate_rad_s, 0.0)
        self.assertAlmostEqual(
            result.body_rate_rad_s[2], proportional_only.body_rate_rad_s[2]
        )

    def test_derivative_guard_brakes_before_crossing_center_both_directions(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.70,
            derivative_enabled=True,
            prediction_horizon_s=0.12,
            derivative_filter_tau_s=0.0,
            maximum_derivative_rate_rad_s=0.20,
        )
        negative_to_positive = apply_tight_fov_guard(
            -math.tan(math.radians(0.26)),
            0.0,
            math.radians(1.0),
            np.zeros(3),
            parameters,
            filtered_s_tight_rate_s_inv=0.019,
            raw_s_tight_rate_s_inv=0.113,
            derivative_initialized=True,
        )
        positive_to_negative = apply_tight_fov_guard(
            math.tan(math.radians(0.26)),
            0.0,
            math.radians(1.0),
            np.zeros(3),
            parameters,
            filtered_s_tight_rate_s_inv=-0.019,
            raw_s_tight_rate_s_inv=-0.113,
            derivative_initialized=True,
        )

        self.assertTrue(negative_to_positive.active)
        self.assertTrue(positive_to_negative.active)
        self.assertGreater(negative_to_positive.outward_s_tight_rate_s_inv, 0.0)
        self.assertGreater(positive_to_negative.outward_s_tight_rate_s_inv, 0.0)
        self.assertLess(negative_to_positive.body_rate_rad_s[2], 0.0)
        self.assertGreater(positive_to_negative.body_rate_rad_s[2], 0.0)
        self.assertAlmostEqual(
            negative_to_positive.body_rate_rad_s[2],
            -positive_to_negative.body_rate_rad_s[2],
        )

    def test_cross_center_prediction_uses_raw_rate_at_recorded_r4_sample(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.70,
            derivative_enabled=True,
            prediction_horizon_s=0.12,
            maximum_derivative_rate_rad_s=0.20,
        )
        measured_s_tight = -0.0045
        measured_x_normalized = measured_s_tight / math.sqrt(
            1.0 - measured_s_tight * measured_s_tight
        )
        result = apply_tight_fov_guard(
            measured_x_normalized,
            0.0,
            math.radians(1.0),
            np.zeros(3),
            parameters,
            filtered_s_tight_rate_s_inv=0.019,
            raw_s_tight_rate_s_inv=0.113,
            derivative_initialized=True,
        )

        self.assertLess(
            measured_s_tight + parameters.prediction_horizon_s * 0.019,
            0.0,
        )
        self.assertGreater(
            measured_s_tight + parameters.prediction_horizon_s * 0.113,
            math.sin(parameters.activation_rad),
        )
        self.assertLess(result.body_rate_rad_s[2], 0.0)
        self.assertGreater(result.derivative_rate_rad_s, 0.0)

    def test_stateful_guard_retains_raw_cross_center_rate_on_duplicate_ticks(self):
        guard = self.make_derivative_guard(
            prediction_horizon_s=0.12,
            derivative_filter_tau_s=0.08,
        )
        guard.apply(-0.0100, 0.0, 1.00, np.zeros(3))
        updated = guard.apply(-0.0045, 0.0, 1.05, np.zeros(3))
        duplicate = guard.apply(-0.0045, 0.0, 1.05, np.zeros(3))

        self.assertAlmostEqual(guard.raw_s_tight_rate_s_inv, 0.11, places=3)
        self.assertLess(updated.body_rate_rad_s[2], 0.0)
        self.assertEqual(updated.feature_sample_status, "updated")
        self.assertEqual(duplicate.feature_sample_status, "duplicate")
        self.assertAlmostEqual(
            duplicate.requested_tight_rate_rad_s,
            updated.requested_tight_rate_rad_s,
        )

    def test_raw_cross_center_override_is_disabled_inside_activation(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.70,
            derivative_enabled=True,
            prediction_horizon_s=0.12,
        )
        result = apply_tight_fov_guard(
            -math.tan(math.radians(0.10)),
            0.0,
            math.radians(1.0),
            [0.1, -0.2, 0.03],
            parameters,
            filtered_s_tight_rate_s_inv=0.03,
            raw_s_tight_rate_s_inv=0.25,
            derivative_initialized=True,
        )

        self.assertFalse(result.active)
        self.assertEqual(result.derivative_rate_rad_s, 0.0)
        np.testing.assert_allclose(result.body_rate_rad_s, [0.1, -0.2, 0.03])

    def test_cross_center_prediction_keeps_derivative_and_yaw_limits(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.70,
            derivative_enabled=True,
            prediction_horizon_s=0.12,
            maximum_derivative_rate_rad_s=0.20,
        )
        result = apply_tight_fov_guard(
            -math.tan(math.radians(0.26)),
            0.0,
            math.radians(1.0),
            [0.0, 0.0, -1.2],
            parameters,
            filtered_s_tight_rate_s_inv=0.019,
            raw_s_tight_rate_s_inv=0.25,
            derivative_initialized=True,
        )

        self.assertTrue(result.active)
        self.assertTrue(result.limited)
        self.assertAlmostEqual(result.derivative_rate_rad_s, 0.20)
        self.assertAlmostEqual(result.body_rate_rad_s[2], -0.70)

    def test_current_acceptance_violation_overrides_cross_center_prediction(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.70,
            derivative_enabled=True,
            prediction_horizon_s=0.12,
            maximum_derivative_rate_rad_s=0.20,
        )

        for measured_deg, raw_rate, commanded_yaw in (
            (1.12, -0.25, 0.10),
            (-1.12, 0.25, -0.10),
        ):
            result = apply_tight_fov_guard(
                math.tan(math.radians(measured_deg)),
                0.0,
                math.radians(1.0),
                [0.0, 0.0, commanded_yaw],
                parameters,
                filtered_s_tight_rate_s_inv=raw_rate,
                raw_s_tight_rate_s_inv=raw_rate,
                derivative_initialized=True,
            )

            measured_direction = math.copysign(1.0, result.measured_s_tight)
            predicted_s_tight = (
                result.measured_s_tight
                + parameters.prediction_horizon_s * raw_rate
            )
            self.assertLess(result.measured_margin_tight, 0.0)
            self.assertLess(predicted_s_tight * result.measured_s_tight, 0.0)
            self.assertLess(result.body_rate_rad_s[2] * measured_direction, 0.0)
            self.assertEqual(result.derivative_rate_rad_s, 0.0)

    def test_acceptance_boundary_itself_overrides_cross_center_prediction(self):
        parameters = TightFOVGuardParameters(
            activation_rad=math.radians(0.25),
            rate_gain=30.0,
            maximum_guarded_rate_rad_s=0.70,
            derivative_enabled=True,
            prediction_horizon_s=0.12,
            maximum_derivative_rate_rad_s=0.20,
        )
        result = apply_tight_fov_guard(
            math.tan(math.radians(1.0)),
            0.0,
            math.radians(1.0),
            [0.0, 0.0, 0.1],
            parameters,
            filtered_s_tight_rate_s_inv=-0.25,
            raw_s_tight_rate_s_inv=-0.25,
            derivative_initialized=True,
        )

        self.assertAlmostEqual(result.measured_margin_tight, 0.0, places=12)
        self.assertLess(result.body_rate_rad_s[2], 0.0)
        self.assertEqual(result.derivative_rate_rad_s, 0.0)

    def test_rate_feedback_guard_and_safety_keep_current_violation_restoring(self):
        booster = BodyRateReversalBooster(
            RateReversalBoostParameters(
                feedback_mode="tracking_error",
                tracking_error_gains=(0.0, 0.6, 1.2),
                filter_tau_s=0.0,
                max_correction_rad_s=(0.35, 0.50, 0.35),
            )
        )
        booster.update([0.0, 0.0, -0.419], 1.0)
        reference_rate = np.array([0.0, 0.0, -0.230])
        feedback = booster.compensate(reference_rate)
        guard = apply_tight_fov_guard(
            0.01954 / math.sqrt(1.0 - 0.01954 * 0.01954),
            0.0,
            math.radians(1.0),
            feedback.body_rate_rad_s,
            TightFOVGuardParameters(
                activation_rad=math.radians(0.25),
                rate_gain=30.0,
                maximum_guarded_rate_rad_s=0.70,
                derivative_enabled=True,
                prediction_horizon_s=0.12,
                maximum_derivative_rate_rad_s=0.10,
            ),
            filtered_s_tight_rate_s_inv=-0.10,
            raw_s_tight_rate_s_inv=-0.25,
            derivative_initialized=True,
        )
        command = sanitize_command(
            [0.0, 0.0, 0.0, 1.0],
            guard.body_rate_rad_s,
            1.0,
            CommandSafetyParameters(),
            enforce_tilt_limit=False,
        )

        self.assertTrue(command.valid)
        self.assertLess(command.body_rate_rad_s[2], 0.0)
        self.assertAlmostEqual(
            reference_rate[2]
            + feedback.tracking_error_correction_rad_s[2]
            + guard.intervention_body_rate_rad_s[2],
            guard.body_rate_rad_s[2],
        )

    def test_derivative_guard_reuses_duplicate_feature_without_reintegrating(self):
        guard = self.make_derivative_guard(derivative_filter_tau_s=0.08)
        guard.apply(math.tan(math.radians(0.20)), 0.0, 1.00, [0.0, 0.0, 0.0])
        updated = guard.apply(
            math.tan(math.radians(0.40)), 0.0, 1.05, [0.0, 0.0, 0.0]
        )
        duplicate = guard.apply(
            math.tan(math.radians(0.40)), 0.0, 1.05, [0.0, 0.0, 0.0]
        )

        self.assertEqual(updated.feature_sample_status, "updated")
        self.assertEqual(duplicate.feature_sample_status, "duplicate")
        self.assertAlmostEqual(
            duplicate.filtered_s_tight_rate_s_inv,
            updated.filtered_s_tight_rate_s_inv,
        )
        self.assertAlmostEqual(
            duplicate.requested_tight_rate_rad_s,
            updated.requested_tight_rate_rad_s,
        )

    def test_same_stamp_changed_feature_resets_derivative(self):
        guard = self.make_derivative_guard(
            prediction_horizon_s=0.12,
            derivative_filter_tau_s=0.08,
        )
        guard.apply(-0.0100, 0.0, 1.00, np.zeros(3))
        guard.apply(-0.0045, 0.0, 1.05, np.zeros(3))
        changed = guard.apply(-0.0020, 0.0, 1.05, np.zeros(3))

        self.assertEqual(changed.feature_sample_status, "reset_same_stamp_changed")
        self.assertFalse(changed.derivative_initialized)
        self.assertEqual(changed.filtered_s_tight_rate_s_inv, 0.0)
        self.assertEqual(changed.derivative_rate_rad_s, 0.0)

    def test_derivative_guard_resets_on_reversed_stamp_and_long_gap(self):
        for next_stamp, expected_status in (
            (0.99, "reset_non_monotonic"),
            (1.30, "reset_gap"),
        ):
            with self.subTest(expected_status=expected_status):
                guard = self.make_derivative_guard()
                guard.apply(
                    math.tan(math.radians(0.20)), 0.0, 1.00, [0.0, 0.0, 0.0]
                )
                result = guard.apply(
                    math.tan(math.radians(0.70)),
                    0.0,
                    next_stamp,
                    [0.0, 0.0, 0.0],
                )

                self.assertEqual(result.feature_sample_status, expected_status)
                self.assertFalse(result.derivative_initialized)
                self.assertEqual(result.derivative_rate_rad_s, 0.0)

    def test_derivative_guard_explicit_reset_clears_dropped_feature_history(self):
        guard = self.make_derivative_guard()
        guard.apply(math.tan(math.radians(0.20)), 0.0, 1.00, [0.0, 0.0, 0.0])
        guard.apply(math.tan(math.radians(0.40)), 0.0, 1.05, [0.0, 0.0, 0.0])
        guard.reset()
        result = guard.apply(
            math.tan(math.radians(0.70)), 0.0, 1.10, [0.0, 0.0, 0.0]
        )

        self.assertEqual(result.feature_sample_status, "initialized")
        self.assertFalse(result.derivative_initialized)
        self.assertEqual(result.filtered_s_tight_rate_s_inv, 0.0)
        self.assertEqual(result.derivative_rate_rad_s, 0.0)

    def test_derivative_guard_limits_raw_derivative_term_and_total_rate(self):
        guard = self.make_derivative_guard(
            prediction_horizon_s=1.0,
            maximum_raw_derivative_s_inv=0.10,
            maximum_derivative_rate_rad_s=0.20,
        )
        guard.apply(0.0, 0.0, 1.00, [0.0, 0.0, 0.8])
        result = guard.apply(
            math.tan(math.radians(2.0)), 0.0, 1.001, [0.0, 0.0, 0.8]
        )

        self.assertAlmostEqual(result.filtered_s_tight_rate_s_inv, 0.10)
        self.assertAlmostEqual(result.derivative_rate_rad_s, 0.20)
        self.assertAlmostEqual(result.body_rate_rad_s[2], -0.70)
        self.assertTrue(result.limited)

    def test_disabled_derivative_keeps_legacy_guard_behavior(self):
        guard = self.make_derivative_guard(derivative_enabled=False)
        guard.apply(math.tan(math.radians(0.20)), 0.0, 1.00, [0.0, 0.0, 0.0])
        result = guard.apply(
            math.tan(math.radians(0.40)), 0.0, 1.05, [0.0, 0.0, 0.0]
        )

        self.assertEqual(result.derivative_rate_rad_s, 0.0)
        self.assertAlmostEqual(
            result.requested_tight_rate_rad_s,
            -30.0
            * (
                math.sin(math.radians(0.40))
                - math.sin(math.radians(0.25))
            ),
        )

    def test_filter_is_stateful_and_resets_after_a_gap(self):
        self.booster.update([0.0, 0.2, 0.2], 1.0)
        filtered = self.booster.update([0.0, 0.0, 0.0], 1.04)
        np.testing.assert_allclose(filtered, [0.0, 0.1, 0.1])

        filtered = self.booster.update([0.0, -0.3, -0.3], 1.30)
        np.testing.assert_allclose(filtered, [0.0, -0.3, -0.3])

    def test_correction_is_bounded(self):
        self.booster.update([0.0, 1.0, 1.0], 1.0)
        result = self.booster.compensate([0.0, -0.1, -0.1])
        np.testing.assert_allclose(result.correction_rad_s, [0.0, 0.5, 0.35])
        np.testing.assert_allclose(result.body_rate_rad_s, [0.0, -0.6, -0.45])

    def test_negative_gain_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "gains"):
            BodyRateReversalBooster(
                RateReversalBoostParameters(gains=(0.0, -0.1, 0.0))
            )

        with self.assertRaisesRegex(ValueError, "damping gains"):
            BodyRateReversalBooster(
                RateReversalBoostParameters(damping_gains=(0.0, 0.0, -0.1))
            )

        with self.assertRaisesRegex(ValueError, "tracking-error gains"):
            BodyRateReversalBooster(
                RateReversalBoostParameters(
                    tracking_error_gains=(0.0, 0.0, -0.1)
                )
            )

        with self.assertRaisesRegex(ValueError, "rate feedback mode"):
            BodyRateReversalBooster(
                RateReversalBoostParameters(feedback_mode="unknown")
            )

        for invalid_modes in (
            ("legacy_reversal", "tracking_error"),
            ("legacy_reversal", "tracking_error", "unknown"),
            1,
        ):
            with self.subTest(axis_feedback_modes=invalid_modes):
                with self.assertRaisesRegex(ValueError, "axis feedback modes"):
                    BodyRateReversalBooster(
                        RateReversalBoostParameters(
                            axis_feedback_modes=invalid_modes
                        )
                    )


class TestPixelAxisGuard(unittest.TestCase):
    @staticmethod
    def parameters(**overrides):
        values = {
            "acceptance_lower_normalized": -0.44,
            "acceptance_upper_normalized": 0.43,
            "activation_lower_normalized": -0.38,
            "activation_upper_normalized": 0.37,
            "rate_gain": 2.0,
            "maximum_guarded_rate_rad_s": 0.70,
        }
        values.update(overrides)
        return PixelAxisGuardParameters(**values)

    def test_positive_raw_y_commands_positive_body_pitch_restore(self):
        result = apply_pixel_axis_guard(
            0.40,
            [0.1, -0.3, 0.2],
            self.parameters(),
            body_rate_axis_index=1,
            restoring_sign=1.0,
        )

        self.assertEqual(result.active_boundary, "upper")
        self.assertTrue(result.active)
        self.assertGreater(result.requested_body_rate_rad_s, 0.0)
        self.assertGreater(result.body_rate_rad_s[1], 0.0)
        self.assertAlmostEqual(result.body_rate_rad_s[1], 0.06)
        np.testing.assert_allclose(result.body_rate_rad_s[[0, 2]], [0.1, 0.2])

    def test_negative_raw_y_commands_negative_body_pitch_restore(self):
        result = apply_pixel_axis_guard(
            -0.41,
            [0.1, 0.3, 0.2],
            self.parameters(),
            body_rate_axis_index=1,
            restoring_sign=1.0,
        )

        self.assertEqual(result.active_boundary, "lower")
        self.assertLess(result.requested_body_rate_rad_s, 0.0)
        self.assertLess(result.body_rate_rad_s[1], 0.0)

    def test_positive_raw_x_keeps_existing_negative_yaw_mapping(self):
        result = apply_pixel_axis_guard(
            0.40,
            [0.1, -0.2, 0.3],
            self.parameters(),
            body_rate_axis_index=2,
            restoring_sign=-1.0,
        )

        self.assertLess(result.requested_body_rate_rad_s, 0.0)
        self.assertLess(result.body_rate_rad_s[2], 0.0)
        np.testing.assert_allclose(result.body_rate_rad_s[:2], [0.1, -0.2])

    def test_inactive_guard_preserves_all_body_rates(self):
        result = apply_pixel_axis_guard(
            0.1,
            [0.1, -0.2, 0.3],
            self.parameters(),
            body_rate_axis_index=1,
            restoring_sign=1.0,
        )

        self.assertFalse(result.active)
        self.assertFalse(result.intervened)
        self.assertEqual(result.active_boundary, "none")
        np.testing.assert_allclose(result.body_rate_rad_s, [0.1, -0.2, 0.3])

    def test_guard_caps_excessive_restoring_rate(self):
        result = apply_pixel_axis_guard(
            0.50,
            [0.0, 1.2, 0.0],
            self.parameters(),
            body_rate_axis_index=1,
            restoring_sign=1.0,
        )

        self.assertTrue(result.limited)
        self.assertAlmostEqual(result.body_rate_rad_s[1], 0.70)
        self.assertFalse(result.within_acceptance)
        self.assertAlmostEqual(result.measured_margin_normalized, -0.07)

    def test_asymmetric_lower_and_upper_limits_are_auditable(self):
        lower = apply_pixel_axis_guard(
            -0.435,
            np.zeros(3),
            self.parameters(),
            body_rate_axis_index=1,
            restoring_sign=1.0,
        )
        upper = apply_pixel_axis_guard(
            0.425,
            np.zeros(3),
            self.parameters(),
            body_rate_axis_index=1,
            restoring_sign=1.0,
        )

        self.assertAlmostEqual(lower.measured_margin_normalized, 0.005)
        self.assertAlmostEqual(upper.measured_margin_normalized, 0.005)
        self.assertTrue(lower.within_acceptance)
        self.assertTrue(upper.within_acceptance)

    def test_derivative_leads_predicted_upper_boundary_crossing(self):
        guard = PixelAxisGuard(
            self.parameters(
                derivative_enabled=True,
                prediction_horizon_s=0.10,
                derivative_filter_tau_s=0.0,
                maximum_raw_derivative_s_inv=2.0,
            ),
            body_rate_axis_index=1,
            restoring_sign=1.0,
        )
        guard.apply(0.30, 1.00, np.zeros(3))
        result = guard.apply(0.36, 1.05, np.zeros(3))

        self.assertEqual(result.feature_sample_status, "updated")
        self.assertTrue(result.derivative_initialized)
        self.assertEqual(result.active_boundary, "upper")
        self.assertGreater(result.derivative_rate_rad_s, 0.0)
        self.assertGreater(result.body_rate_rad_s[1], 0.0)

    def test_duplicate_stamp_does_not_reintegrate_derivative(self):
        guard = PixelAxisGuard(
            self.parameters(
                derivative_enabled=True,
                derivative_filter_tau_s=0.0,
            ),
            body_rate_axis_index=1,
            restoring_sign=1.0,
        )
        guard.apply(0.20, 1.00, np.zeros(3))
        updated = guard.apply(0.30, 1.05, np.zeros(3))
        duplicate = guard.apply(0.30, 1.05, np.zeros(3))

        self.assertEqual(updated.feature_sample_status, "updated")
        self.assertEqual(duplicate.feature_sample_status, "duplicate")
        self.assertAlmostEqual(
            duplicate.filtered_coordinate_rate_s_inv,
            updated.filtered_coordinate_rate_s_inv,
        )

    def test_same_stamp_changed_vertical_feature_resets_derivative(self):
        guard = PixelAxisGuard(
            self.parameters(
                derivative_enabled=True,
                prediction_horizon_s=0.10,
                derivative_filter_tau_s=0.08,
            ),
            body_rate_axis_index=1,
            restoring_sign=1.0,
        )
        guard.apply(0.20, 1.00, np.zeros(3))
        guard.apply(0.30, 1.05, np.zeros(3))

        changed = guard.apply(0.35, 1.05, np.zeros(3))

        self.assertEqual(changed.feature_sample_status, "reset_same_stamp_changed")
        self.assertFalse(changed.derivative_initialized)
        self.assertEqual(changed.filtered_coordinate_rate_s_inv, 0.0)
        self.assertEqual(changed.derivative_rate_rad_s, 0.0)

    def test_invalid_limits_axis_and_sign_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "acceptance_lower"):
            self.parameters(activation_lower_normalized=-0.5).validate()
        with self.assertRaisesRegex(ValueError, "axis index"):
            PixelAxisGuard(self.parameters(), 3, 1.0)
        with self.assertRaisesRegex(ValueError, "restoring sign"):
            PixelAxisGuard(self.parameters(), 1, 0.0)


if __name__ == "__main__":
    unittest.main()
