import math

import numpy as np


EPS = 1e-12


def vector3(value):
    result = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(result)):
        raise ValueError("vector contains NaN or Inf")
    return result


def normalize(value, name="vector"):
    result = vector3(value)
    norm = float(np.linalg.norm(result))
    if norm <= EPS:
        raise ValueError("{} norm is zero".format(name))
    return result / norm


def skew(value):
    x, y, z = vector3(value)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(matrix):
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def camera_ray(x_normalized, y_normalized):
    return normalize([x_normalized, y_normalized, 1.0], "camera ray")


def los_from_relative_position(position):
    return -normalize(position, "relative position")


def quaternion_xyzw_to_rotation(quaternion):
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= EPS or not np.all(np.isfinite(q)):
        raise ValueError("invalid quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def rotation_to_quaternion_xyzw(rotation):
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
                0.25 * s,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        s = math.sqrt(
            max(
                1.0
                + rotation[index, index]
                - rotation[next_index, next_index]
                - rotation[last_index, last_index],
                0.0,
            )
        ) * 2.0
        quaternion = np.zeros(4)
        quaternion[index] = 0.25 * s
        quaternion[3] = (rotation[last_index, next_index] - rotation[next_index, last_index]) / s
        quaternion[next_index] = (rotation[next_index, index] + rotation[index, next_index]) / s
        quaternion[last_index] = (rotation[last_index, index] + rotation[index, last_index]) / s
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion / np.linalg.norm(quaternion)


def rotation_aligning(source, target):
    source = normalize(source, "source direction")
    target = normalize(target, "target direction")
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine <= EPS:
        if cosine > 0.0:
            return np.eye(3)
        seed = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.8:
            seed = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(source, seed), "opposite rotation axis")
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    cross_matrix = skew(cross)
    return np.eye(3) + cross_matrix + cross_matrix.dot(cross_matrix) * ((1.0 - cosine) / (sine * sine))


def desired_attitude_from_direction(current_rotation, desired_direction, body_axis=(0.0, 0.0, 1.0)):
    current_rotation = np.asarray(current_rotation, dtype=float).reshape(3, 3)
    current_axis = current_rotation.dot(normalize(body_axis, "body thrust axis"))
    tilt = rotation_aligning(current_axis, desired_direction)
    return tilt.dot(current_rotation)

