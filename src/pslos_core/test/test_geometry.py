#!/usr/bin/env python3
import math
import unittest

import numpy as np

from pslos_core.geometry import (
    camera_ray,
    desired_attitude_from_direction,
    los_from_relative_position,
    quaternion_xyzw_to_rotation,
    rotation_to_quaternion_xyzw,
)


class TestGeometry(unittest.TestCase):
    def test_a_f01_los_sign(self):
        position = np.array([-4.0, 2.0, -1.0])
        expected = -position / np.linalg.norm(position)
        np.testing.assert_allclose(los_from_relative_position(position), expected, atol=1e-12)

    def test_a_f02_camera_ray_signs(self):
        ray = camera_ray(0.2, -0.3)
        self.assertGreater(ray[0], 0.0)
        self.assertLess(ray[1], 0.0)
        self.assertGreater(ray[2], 0.0)

    def test_a_f07_force_direction_attitude(self):
        desired = np.array([1.0, -2.0, 3.0])
        rotation = desired_attitude_from_direction(np.eye(3), desired)
        aligned = rotation[:, 2]
        np.testing.assert_allclose(aligned, desired / np.linalg.norm(desired), atol=1e-12)

    def test_opposite_direction_alignment(self):
        rotation = desired_attitude_from_direction(np.eye(3), [0.0, 0.0, -1.0])
        np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)

    def test_quaternion_round_trip(self):
        q = np.array([0.2, -0.3, 0.1, 0.92])
        q /= np.linalg.norm(q)
        q_round_trip = rotation_to_quaternion_xyzw(quaternion_xyzw_to_rotation(q))
        self.assertLess(math.acos(np.clip(abs(np.dot(q, q_round_trip)), -1.0, 1.0)), 1e-7)


if __name__ == "__main__":
    unittest.main()
