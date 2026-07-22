from dataclasses import dataclass
import math

import numpy as np

from .geometry import quaternion_xyzw_to_rotation


@dataclass(frozen=True)
class CommandSafetyParameters:
    hover_thrust: float = 0.5
    min_thrust: float = 0.1
    max_thrust: float = 0.75
    max_body_rate_rad_s: tuple = (1.2, 1.2, 0.8)
    max_tilt_rad: float = math.radians(35.0)

    def validate(self):
        if not 0.0 <= self.min_thrust <= self.hover_thrust <= self.max_thrust <= 1.0:
            raise ValueError("thrust limits must satisfy 0 <= min <= hover <= max <= 1")
        if min(self.max_body_rate_rad_s) <= 0.0:
            raise ValueError("body-rate limits must be positive")
        if not 0.0 < self.max_tilt_rad < 0.5 * math.pi:
            raise ValueError("max_tilt_rad must be in (0, pi/2)")


@dataclass(frozen=True)
class SafeCommand:
    valid: bool
    reason: str
    attitude_xyzw: np.ndarray
    body_rate_rad_s: np.ndarray
    thrust: float
    saturated: bool


def sanitize_command(
    attitude_xyzw,
    body_rate_rad_s,
    normalized_thrust,
    parameters,
    enforce_tilt_limit=True,
):
    parameters.validate()
    quaternion = np.asarray(attitude_xyzw, dtype=float).reshape(4)
    rates = np.asarray(body_rate_rad_s, dtype=float).reshape(3)
    values = np.concatenate((quaternion, rates, [float(normalized_thrust)]))
    if not np.all(np.isfinite(values)):
        return SafeCommand(False, "non_finite_command", quaternion, rates, 0.0, False)

    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm < 0.5:
        return SafeCommand(False, "invalid_attitude", quaternion, rates, 0.0, False)
    quaternion = quaternion / quaternion_norm
    rotation_world_body = quaternion_xyzw_to_rotation(quaternion)
    tilt = math.acos(
        float(np.clip(np.dot(rotation_world_body[:, 2], [0.0, 0.0, 1.0]), -1.0, 1.0))
    )
    if enforce_tilt_limit and tilt > parameters.max_tilt_rad:
        return SafeCommand(False, "tilt_limit", quaternion, rates, 0.0, False)

    rate_limits = np.asarray(parameters.max_body_rate_rad_s, dtype=float)
    limited_rates = np.clip(rates, -rate_limits, rate_limits)
    raw_thrust = parameters.hover_thrust * float(normalized_thrust)
    if raw_thrust <= 0.0:
        return SafeCommand(
            False, "non_positive_thrust", quaternion, limited_rates, 0.0, False
        )
    limited_thrust = float(
        np.clip(raw_thrust, parameters.min_thrust, parameters.max_thrust)
    )
    saturated = not np.allclose(limited_rates, rates) or not math.isclose(
        limited_thrust, raw_thrust
    )
    return SafeCommand(
        True,
        "saturated" if saturated else "accepted",
        quaternion,
        limited_rates,
        limited_thrust,
        saturated,
    )
