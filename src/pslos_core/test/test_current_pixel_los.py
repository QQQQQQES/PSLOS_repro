#!/usr/bin/env python3
import math
from types import SimpleNamespace
import unittest

import numpy as np

from pslos_core.current_pixel_los import WorldLOSDerivative, current_pixel_los_world


class FloatStamp:
    def __init__(self, seconds):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds


def feature(**overrides):
    values = {
        "detected": True,
        "header": SimpleNamespace(
            frame_id="camera_optical",
            stamp=FloatStamp(9.95),
        ),
        "x_normalized": 0.0,
        "y_normalized": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestCurrentPixelLOS(unittest.TestCase):
    def setUp(self):
        self.rotation_body_camera = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

    def compute(self, sample=None, **overrides):
        values = {
            "feature": feature() if sample is None else sample,
            "current_time_s": 10.0,
            "maximum_feature_age_s": 0.1,
            "rotation_world_body": np.eye(3),
            "rotation_body_camera": self.rotation_body_camera,
        }
        values.update(overrides)
        return current_pixel_los_world(**values)

    def test_camera_coordinates_map_to_world_los(self):
        sample = feature(x_normalized=0.2, y_normalized=-0.3)
        los_world, age_s = self.compute(sample)
        expected = np.array([1.0, -0.2, 0.3]) / math.sqrt(1.13)

        np.testing.assert_allclose(los_world, expected, atol=1e-12)
        self.assertAlmostEqual(age_s, 0.05)
        self.assertAlmostEqual(np.linalg.norm(los_world), 1.0)

    def test_world_body_rotation_is_applied_after_camera_mapping(self):
        rotation_world_body = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        los_world, _ = self.compute(rotation_world_body=rotation_world_body)

        np.testing.assert_allclose(los_world, [0.0, 1.0, 0.0], atol=1e-12)

    def test_zero_age_and_maximum_age_boundaries_are_accepted(self):
        for stamp, expected_age in ((10.0, 0.0), (9.9, 0.1)):
            with self.subTest(stamp=stamp):
                sample = feature(
                    header=SimpleNamespace(
                        frame_id="camera_optical",
                        stamp=stamp,
                    )
                )
                _, age_s = self.compute(sample)
                self.assertAlmostEqual(age_s, expected_age)

    def test_stale_and_future_features_are_rejected(self):
        cases = ((9.899, "feature_stale"), (10.001, "feature_future"))
        for stamp, reason in cases:
            with self.subTest(reason=reason):
                sample = feature(
                    header=SimpleNamespace(
                        frame_id="camera_optical",
                        stamp=stamp,
                    )
                )
                with self.assertRaisesRegex(ValueError, "^{}$".format(reason)):
                    self.compute(sample)

    def test_zero_and_nonfinite_stamps_are_rejected(self):
        for stamp in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(stamp=stamp):
                sample = feature(
                    header=SimpleNamespace(
                        frame_id="camera_optical",
                        stamp=stamp,
                    )
                )
                with self.assertRaisesRegex(ValueError, "^feature_stamp$"):
                    self.compute(sample)

    def test_wrong_frame_and_not_detected_are_rejected(self):
        cases = (
            (
                feature(
                    header=SimpleNamespace(
                        frame_id="base_link",
                        stamp=9.95,
                    )
                ),
                "feature_frame",
            ),
            (feature(detected=False), "feature_not_detected"),
        )
        for sample, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, "^{}$".format(reason)):
                    self.compute(sample)

    def test_nonfinite_normalized_coordinates_are_rejected(self):
        for field in ("x_normalized", "y_normalized"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(field=field, value=value):
                    sample = feature(**{field: value})
                    with self.assertRaisesRegex(ValueError, "^feature_nonfinite$"):
                        self.compute(sample)

    def test_duplicate_feature_stamp_is_stateless_and_allowed(self):
        sample = feature()
        first_los, first_age = self.compute(sample)
        second_los, second_age = self.compute(sample)

        np.testing.assert_allclose(second_los, first_los, atol=0.0)
        self.assertEqual(second_age, first_age)


class TestWorldLOSDerivative(unittest.TestCase):
    def test_distinct_samples_produce_tangent_world_rate(self):
        derivative = WorldLOSDerivative(
            filter_tau_s=0.0,
            reset_gap_s=0.2,
            maximum_rate_rad_s=2.0,
        )
        first_rate, initialized, status = derivative.update([1.0, 0.0, 0.0], 1.0)
        np.testing.assert_allclose(first_rate, np.zeros(3))
        self.assertFalse(initialized)
        self.assertEqual(status, "initialized")

        los = np.array([1.0, 0.1, 0.0])
        los /= np.linalg.norm(los)
        rate, initialized, status = derivative.update(los, 1.1)

        self.assertTrue(initialized)
        self.assertEqual(status, "updated")
        self.assertGreater(rate[1], 0.0)
        self.assertAlmostEqual(float(np.dot(rate, los)), 0.0, places=12)
        self.assertLessEqual(np.linalg.norm(rate), 2.0)

    def test_duplicate_stamp_does_not_change_filtered_rate(self):
        derivative = WorldLOSDerivative(filter_tau_s=0.0)
        derivative.update([1.0, 0.0, 0.0], 1.0)
        expected, initialized, _ = derivative.update([1.0, 0.1, 0.0], 1.05)
        duplicate, duplicate_initialized, status = derivative.update(
            [1.0, 0.2, 0.0], 1.05
        )

        np.testing.assert_allclose(duplicate, expected)
        self.assertTrue(initialized)
        self.assertTrue(duplicate_initialized)
        self.assertEqual(status, "duplicate")

    def test_gap_and_nonmonotonic_stamp_reset_history(self):
        for stamp, expected_status in ((1.3, "reset_gap"), (0.9, "reset_non_monotonic")):
            with self.subTest(status=expected_status):
                derivative = WorldLOSDerivative(reset_gap_s=0.1)
                derivative.update([1.0, 0.0, 0.0], 1.0)
                rate, initialized, status = derivative.update([1.0, 0.1, 0.0], stamp)
                np.testing.assert_allclose(rate, np.zeros(3))
                self.assertFalse(initialized)
                self.assertEqual(status, expected_status)


if __name__ == "__main__":
    unittest.main()
