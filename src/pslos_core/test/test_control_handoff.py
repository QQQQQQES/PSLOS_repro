#!/usr/bin/env python3
import math
import unittest

from pslos_core.control_handoff import (
    ControlHandoffGate,
    ControlHandoffObservation,
    ControlHandoffParameters,
    evaluate_control_handoff,
    normalized_feature_tight_coordinate,
)


def ready_observation(**overrides):
    values = {
        "estimate_valid": True,
        "estimate_source_is_dcekf": True,
        "measurement_count": 6,
        "vision_age_s": 0.05,
        "feature_detected": True,
        "feature_centered": True,
        "feature_age_s": 0.04,
        "feature_frame_valid": True,
        "debug_valid": True,
        "debug_age_s": 0.03,
        "observation_skew_s": 0.02,
        "estimate_s_tight": math.sin(math.radians(0.1)),
        "measured_s_tight": math.sin(math.radians(0.15)),
    }
    values.update(overrides)
    return ControlHandoffObservation(**values)


class TestControlHandoff(unittest.TestCase):
    def setUp(self):
        self.parameters = ControlHandoffParameters()

    def test_normalized_feature_uses_camera_ray_component(self):
        value = normalized_feature_tight_coordinate(0.2, -0.1)
        self.assertAlmostEqual(value, 0.2 / math.sqrt(1.05))

    def test_ready_observation_passes(self):
        result = evaluate_control_handoff(ready_observation(), self.parameters)
        self.assertTrue(result.ready)
        self.assertEqual(result.reason, "ready")
        self.assertAlmostEqual(result.tight_error_rad, math.radians(0.05))

    def test_each_estimator_gate_rejects(self):
        cases = (
            ({"estimate_valid": False}, "estimate_invalid"),
            ({"estimate_source_is_dcekf": False}, "estimate_source"),
            ({"measurement_count": 5}, "measurement_count"),
            ({"vision_age_s": 0.101}, "vision_age"),
            ({"feature_detected": False}, "feature_invalid"),
            ({"feature_centered": False}, "feature_center"),
            ({"feature_age_s": 0.101}, "feature_age"),
            ({"debug_valid": False}, "debug_invalid"),
            ({"debug_age_s": 0.101}, "debug_age"),
            ({"observation_skew_s": 0.031}, "observation_skew"),
            (
                {"measured_s_tight": math.sin(math.radians(0.31))},
                "tight_error",
            ),
        )
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                result = evaluate_control_handoff(
                    ready_observation(**overrides), self.parameters
                )
                self.assertFalse(result.ready)
                self.assertEqual(result.reason, reason)

    def test_gate_requires_continuous_stability(self):
        gate = ControlHandoffGate(self.parameters)
        self.assertFalse(gate.update(1.0, ready_observation())[0])
        self.assertFalse(gate.update(1.08, ready_observation())[0])
        self.assertFalse(gate.update(1.16, ready_observation())[0])
        enabled, duration, _ = gate.update(1.20, ready_observation())
        self.assertTrue(enabled)
        self.assertAlmostEqual(duration, 0.2)

    def test_gate_resets_after_a_failed_observation(self):
        gate = ControlHandoffGate(self.parameters)
        gate.update(1.0, ready_observation())
        gate.update(1.08, ready_observation())
        gate.update(1.11, ready_observation(feature_detected=False))
        self.assertEqual(gate.reset_count, 1)
        self.assertFalse(gate.update(1.30, ready_observation())[0])
        self.assertFalse(gate.update(1.38, ready_observation())[0])
        self.assertFalse(gate.update(1.46, ready_observation())[0])
        self.assertTrue(gate.update(1.50, ready_observation())[0])

    def test_gate_resets_when_new_samples_are_too_far_apart(self):
        gate = ControlHandoffGate(self.parameters)
        gate.update(1.0, ready_observation())
        gate.update(1.08, ready_observation())
        enabled, duration, _ = gate.update(1.17, ready_observation())
        self.assertFalse(enabled)
        self.assertAlmostEqual(duration, 0.0)
        self.assertEqual(gate.reset_count, 1)

    def test_gate_resets_on_duplicate_sample_stamp(self):
        gate = ControlHandoffGate(self.parameters)
        gate.update(1.0, ready_observation())
        gate.update(1.08, ready_observation())
        enabled, duration, _ = gate.update(1.08, ready_observation())
        self.assertFalse(enabled)
        self.assertAlmostEqual(duration, 0.0)
        self.assertEqual(gate.reset_count, 1)

    def test_non_physical_tight_coordinates_are_rejected(self):
        for field in ("estimate_s_tight", "measured_s_tight"):
            with self.subTest(field=field):
                result = evaluate_control_handoff(
                    ready_observation(**{field: 1.01}), self.parameters
                )
                self.assertFalse(result.ready)
                self.assertEqual(result.reason, "tight_coordinate")

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "minimum measurements"):
            ControlHandoffParameters(minimum_measurements=0).validate()
        with self.assertRaisesRegex(ValueError, "maximum tight error"):
            ControlHandoffParameters(maximum_tight_error_rad=math.inf).validate()


if __name__ == "__main__":
    unittest.main()
