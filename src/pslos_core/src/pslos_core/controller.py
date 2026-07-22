import math
from dataclasses import dataclass

import numpy as np

from .geometry import desired_attitude_from_direction, normalize, vee, vector3


PAPER_LITERAL = "paper_literal"
SYMMETRIC_GRADIENT = "symmetric_gradient"
PAPER_EQUATION_15 = "paper_equation_15"
LYAPUNOV_CORRECTED = "lyapunov_corrected"
HARD_LONG_AXIS_BARRIER = "hard"
SOFT_LONG_AXIS_RECOVERY = "soft_recovery"


def limit_vector_norm(vector, maximum_norm):
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= maximum_norm or norm <= 1e-12:
        return vector
    return vector * (float(maximum_norm) / norm)


@dataclass(frozen=True)
class PSLOSParameters:
    alpha_lon_rad: float
    alpha_lat_rad: float
    c1: float = 0.6
    c2: float = 1.0
    c_omega: float = 1.0
    mass: float = 1.0
    tight_axis_gain: float = 1.0

    def validate(self):
        if not 0.0 < self.alpha_lon_rad < math.pi / 2.0:
            raise ValueError("alpha_lon_rad must be in (0, pi/2)")
        if not 0.0 < self.alpha_lat_rad < self.alpha_lon_rad:
            raise ValueError("alpha_lat_rad must be in (0, alpha_lon_rad)")
        if min(self.c1, self.c2, self.c_omega, self.mass, self.tight_axis_gain) <= 0.0:
            raise ValueError("controller gains and mass must be positive")


@dataclass(frozen=True)
class PSLOSEvaluation:
    los_world: np.ndarray
    axis_long_world: np.ndarray
    axis_tight_world: np.ndarray
    s_long: float
    s_tight: float
    z1: float
    z2: float
    c_long: float
    c_tight: float
    margin_long: float
    margin_tight: float
    l1: float
    l2: float
    k_h: float
    k_v: float
    tangent_projection: np.ndarray
    fov_tangent: np.ndarray
    omega_los_body: np.ndarray
    formula_mode: str


@dataclass(frozen=True)
class ControlResult:
    desired_relative_velocity: np.ndarray
    velocity_error: np.ndarray
    paper_force_world: np.ndarray
    corrected_force_world: np.ndarray
    selected_force_world: np.ndarray
    corrected_acceleration_world: np.ndarray
    desired_attitude: np.ndarray
    desired_body_rate: np.ndarray
    los_rate_guidance_acceleration_world: np.ndarray
    los_rate_guidance_active: bool
    los_rate_guidance_range_scale: float
    effective_long_axis_fov_angular_rate_weight: float
    fov: PSLOSEvaluation
    force_law_mode: str


def evaluate_pslos(
    los_world,
    range_m,
    rotation_world_body,
    rotation_body_camera,
    parameters,
    formula_mode=PAPER_LITERAL,
    long_axis_barrier_policy=HARD_LONG_AXIS_BARRIER,
    minimum_long_axis_denominator=0.01,
):
    parameters.validate()
    if range_m <= 0.0 or not math.isfinite(range_m):
        raise ValueError("range_m must be finite and positive")
    if formula_mode not in (PAPER_LITERAL, SYMMETRIC_GRADIENT):
        raise ValueError("unknown formula_mode: {}".format(formula_mode))
    if long_axis_barrier_policy not in (
        HARD_LONG_AXIS_BARRIER,
        SOFT_LONG_AXIS_RECOVERY,
    ):
        raise ValueError(
            "unknown long-axis barrier policy: {}".format(
                long_axis_barrier_policy
            )
        )
    minimum_long_axis_denominator = float(minimum_long_axis_denominator)
    if (
        not math.isfinite(minimum_long_axis_denominator)
        or minimum_long_axis_denominator <= 0.0
    ):
        raise ValueError(
            "minimum long-axis denominator must be finite and positive"
        )

    los_world = normalize(los_world, "LOS")
    rotation_world_body = np.asarray(rotation_world_body, dtype=float).reshape(3, 3)
    rotation_body_camera = np.asarray(rotation_body_camera, dtype=float).reshape(3, 3)
    rotation_world_camera = rotation_world_body.dot(rotation_body_camera)
    axis_long = rotation_world_camera.dot(np.array([0.0, 1.0, 0.0]))
    axis_tight = rotation_world_camera.dot(np.array([1.0, 0.0, 0.0]))

    s_long = float(np.dot(axis_long, los_world))
    s_tight = float(np.dot(axis_tight, los_world))
    z1 = abs(s_long)
    z2 = s_tight
    c_long = math.sin(parameters.alpha_lon_rad)
    c_tight = math.sin(parameters.alpha_lat_rad)
    raw_denominator = c_long * c_long - z1 * z1
    if (
        long_axis_barrier_policy == HARD_LONG_AXIS_BARRIER
        and raw_denominator <= 0.0
    ):
        raise ValueError("LOS is outside the long-axis barrier")
    denominator = (
        raw_denominator
        if long_axis_barrier_policy == HARD_LONG_AXIS_BARRIER
        else max(raw_denominator, minimum_long_axis_denominator)
    )

    l1 = 0.5 * math.log((c_long * c_long) / denominator)
    l2 = 0.5 * z2 * z2
    k_h = z1 / denominator
    k_v = parameters.tight_axis_gain * z2
    tangent = np.eye(3) - np.outer(los_world, los_world)

    long_axis_for_control = axis_long
    if formula_mode == SYMMETRIC_GRADIENT and s_long < 0.0:
        long_axis_for_control = -axis_long

    fov_tangent = (
        k_h * tangent.dot(long_axis_for_control) + k_v * tangent.dot(axis_tight)
    ) / range_m
    omega_los_world = parameters.c_omega * (
        k_h * np.cross(los_world, long_axis_for_control)
        + k_v * np.cross(los_world, axis_tight)
    )
    omega_los_body = rotation_world_body.T.dot(omega_los_world)

    return PSLOSEvaluation(
        los_world=los_world,
        axis_long_world=axis_long,
        axis_tight_world=axis_tight,
        s_long=s_long,
        s_tight=s_tight,
        z1=z1,
        z2=z2,
        c_long=c_long,
        c_tight=c_tight,
        margin_long=c_long - z1,
        margin_tight=c_tight - abs(z2),
        l1=l1,
        l2=l2,
        k_h=k_h,
        k_v=k_v,
        tangent_projection=tangent,
        fov_tangent=fov_tangent,
        omega_los_body=omega_los_body,
        formula_mode=formula_mode,
    )


class PSLOSController:
    def __init__(
        self,
        parameters,
        rotation_body_camera,
        formula_mode=PAPER_LITERAL,
        force_law_mode=PAPER_EQUATION_15,
        maximum_closing_speed_mps=None,
        maximum_command_acceleration_mps2=None,
        minimum_closing_range_m=0.0,
        attitude_tracking_gain=None,
        long_axis_fov_angular_rate_weight=1.0,
        long_axis_fov_recovery_start_rad=0.0,
        long_axis_fov_full_recovery_rad=0.0,
        long_axis_barrier_policy=HARD_LONG_AXIS_BARRIER,
        minimum_long_axis_denominator=0.01,
        los_rate_guidance_gain=0.0,
        maximum_los_rate_guidance_acceleration_mps2=0.0,
        los_rate_guidance_terminal_start_range_m=0.0,
        los_rate_guidance_terminal_minimum_scale=1.0,
        radial_velocity_feedback_only=False,
    ):
        parameters.validate()
        if force_law_mode not in (PAPER_EQUATION_15, LYAPUNOV_CORRECTED):
            raise ValueError("unknown force_law_mode: {}".format(force_law_mode))
        if force_law_mode == LYAPUNOV_CORRECTED and formula_mode != SYMMETRIC_GRADIENT:
            raise ValueError("lyapunov_corrected requires symmetric_gradient")
        self.parameters = parameters
        self.rotation_body_camera = np.asarray(rotation_body_camera, dtype=float).reshape(3, 3)
        self.formula_mode = formula_mode
        self.force_law_mode = force_law_mode
        limits = (maximum_closing_speed_mps, maximum_command_acceleration_mps2)
        if (limits[0] is None) != (limits[1] is None):
            raise ValueError(
                "maximum closing speed and command acceleration must be enabled together"
            )
        if any(
            value is not None
            and (not math.isfinite(float(value)) or float(value) <= 0.0)
            for value in limits
        ):
            raise ValueError("long-range limits must be finite and positive")
        self.maximum_closing_speed_mps = (
            float(maximum_closing_speed_mps)
            if maximum_closing_speed_mps is not None
            else None
        )
        self.maximum_command_acceleration_mps2 = (
            float(maximum_command_acceleration_mps2)
            if maximum_command_acceleration_mps2 is not None
            else None
        )
        self.bounded_velocity_mode = self.maximum_closing_speed_mps is not None
        self.minimum_closing_range_m = float(minimum_closing_range_m)
        if (
            not math.isfinite(self.minimum_closing_range_m)
            or self.minimum_closing_range_m < 0.0
        ):
            raise ValueError("minimum closing range must be finite and nonnegative")
        if self.minimum_closing_range_m > 0.0 and not self.bounded_velocity_mode:
            raise ValueError("minimum closing range requires bounded velocity mode")
        self.attitude_tracking_gain = (
            self.parameters.c_omega
            if attitude_tracking_gain is None
            else float(attitude_tracking_gain)
        )
        if (
            not math.isfinite(self.attitude_tracking_gain)
            or self.attitude_tracking_gain <= 0.0
        ):
            raise ValueError("attitude tracking gain must be finite and positive")
        self.long_axis_fov_angular_rate_weight = float(
            long_axis_fov_angular_rate_weight
        )
        if (
            not math.isfinite(self.long_axis_fov_angular_rate_weight)
            or not 0.0 <= self.long_axis_fov_angular_rate_weight <= 1.0
        ):
            raise ValueError(
                "long-axis FOV angular-rate weight must be finite and in [0, 1]"
            )
        self.long_axis_fov_recovery_start_rad = float(
            long_axis_fov_recovery_start_rad
        )
        self.long_axis_fov_full_recovery_rad = float(
            long_axis_fov_full_recovery_rad
        )
        recovery_angles = (
            self.long_axis_fov_recovery_start_rad,
            self.long_axis_fov_full_recovery_rad,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in recovery_angles):
            raise ValueError("long-axis FOV recovery angles must be finite and nonnegative")
        self.adaptive_long_axis_fov_weight = (
            self.long_axis_fov_full_recovery_rad > 0.0
        )
        if self.adaptive_long_axis_fov_weight:
            if not (
                self.long_axis_fov_recovery_start_rad
                < self.long_axis_fov_full_recovery_rad
                <= self.parameters.alpha_lon_rad
            ):
                raise ValueError(
                    "long-axis FOV recovery angles must satisfy "
                    "start < full <= alpha_lon"
                )
        elif self.long_axis_fov_recovery_start_rad != 0.0:
            raise ValueError(
                "long-axis FOV recovery start requires a positive full-recovery angle"
            )
        if long_axis_barrier_policy not in (
            HARD_LONG_AXIS_BARRIER,
            SOFT_LONG_AXIS_RECOVERY,
        ):
            raise ValueError(
                "unknown long-axis barrier policy: {}".format(
                    long_axis_barrier_policy
                )
            )
        minimum_long_axis_denominator = float(minimum_long_axis_denominator)
        if (
            not math.isfinite(minimum_long_axis_denominator)
            or minimum_long_axis_denominator <= 0.0
        ):
            raise ValueError(
                "minimum long-axis denominator must be finite and positive"
            )
        self.long_axis_barrier_policy = str(long_axis_barrier_policy)
        self.minimum_long_axis_denominator = minimum_long_axis_denominator
        self.los_rate_guidance_gain = float(los_rate_guidance_gain)
        self.maximum_los_rate_guidance_acceleration_mps2 = float(
            maximum_los_rate_guidance_acceleration_mps2
        )
        los_rate_limits = (
            self.los_rate_guidance_gain,
            self.maximum_los_rate_guidance_acceleration_mps2,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in los_rate_limits):
            raise ValueError("LOS-rate guidance limits must be finite and nonnegative")
        if (los_rate_limits[0] > 0.0) != (los_rate_limits[1] > 0.0):
            raise ValueError(
                "LOS-rate guidance gain and acceleration limit must be enabled together"
            )
        if self.los_rate_guidance_gain > 0.0 and not self.bounded_velocity_mode:
            raise ValueError("LOS-rate guidance requires bounded velocity mode")
        self.los_rate_guidance_terminal_start_range_m = float(
            los_rate_guidance_terminal_start_range_m
        )
        self.los_rate_guidance_terminal_minimum_scale = float(
            los_rate_guidance_terminal_minimum_scale
        )
        if (
            not math.isfinite(self.los_rate_guidance_terminal_start_range_m)
            or self.los_rate_guidance_terminal_start_range_m < 0.0
        ):
            raise ValueError("LOS-rate terminal start range must be finite and nonnegative")
        if (
            not math.isfinite(self.los_rate_guidance_terminal_minimum_scale)
            or not 0.0 <= self.los_rate_guidance_terminal_minimum_scale <= 1.0
        ):
            raise ValueError("LOS-rate terminal minimum scale must be in [0, 1]")
        if (
            self.los_rate_guidance_terminal_start_range_m > 0.0
            and self.los_rate_guidance_gain <= 0.0
        ):
            raise ValueError("LOS-rate terminal scaling requires LOS-rate guidance")
        self.radial_velocity_feedback_only = bool(radial_velocity_feedback_only)
        if self.radial_velocity_feedback_only and not self.bounded_velocity_mode:
            raise ValueError("radial velocity feedback requires bounded velocity mode")

    def compute(
        self,
        relative_position,
        relative_velocity,
        rotation_world_body,
        aerodynamic_force_world=(0.0, 0.0, 0.0),
        gravity_force_world=(0.0, 0.0, 9.80665),
        los_world=None,
        los_rate_world=None,
    ):
        relative_position = vector3(relative_position)
        relative_velocity = vector3(relative_velocity)
        rotation_world_body = np.asarray(rotation_world_body, dtype=float).reshape(3, 3)
        range_m = float(np.linalg.norm(relative_position))
        los_world = (
            -relative_position / range_m
            if los_world is None
            else normalize(los_world, "LOS")
        )
        fov = evaluate_pslos(
            los_world,
            range_m,
            rotation_world_body,
            self.rotation_body_camera,
            self.parameters,
            self.formula_mode,
            self.long_axis_barrier_policy,
            self.minimum_long_axis_denominator,
        )

        closing_position = relative_position
        if self.minimum_closing_range_m > range_m:
            closing_position = (
                relative_position * self.minimum_closing_range_m / range_m
            )
        desired_relative_velocity = -self.parameters.c1 * closing_position
        if self.bounded_velocity_mode:
            desired_relative_velocity = limit_vector_norm(
                desired_relative_velocity, self.maximum_closing_speed_mps
            )
        velocity_feedback = relative_velocity
        if self.radial_velocity_feedback_only:
            velocity_feedback = los_world * float(
                np.dot(los_world, relative_velocity)
            )
        velocity_error = velocity_feedback - desired_relative_velocity
        los_rate_guidance_acceleration = np.zeros(3)
        los_rate_guidance_active = False
        los_rate_guidance_range_scale = 1.0
        if self.los_rate_guidance_gain > 0.0 and los_rate_world is not None:
            los_rate_world = vector3(los_rate_world)
            los_rate_world = los_rate_world - los_world * float(
                np.dot(los_world, los_rate_world)
            )
            commanded_closing_speed = float(np.linalg.norm(desired_relative_velocity))
            if (
                self.los_rate_guidance_terminal_start_range_m > 0.0
                and range_m < self.los_rate_guidance_terminal_start_range_m
            ):
                blend = max(
                    0.0,
                    min(
                        1.0,
                        range_m / self.los_rate_guidance_terminal_start_range_m,
                    ),
                )
                blend = blend * blend * (3.0 - 2.0 * blend)
                los_rate_guidance_range_scale = (
                    self.los_rate_guidance_terminal_minimum_scale
                    + (1.0 - self.los_rate_guidance_terminal_minimum_scale) * blend
                )
            los_rate_guidance_acceleration = limit_vector_norm(
                self.los_rate_guidance_gain
                * commanded_closing_speed
                * los_rate_guidance_range_scale
                * los_rate_world,
                self.maximum_los_rate_guidance_acceleration_mps2,
            )
            los_rate_guidance_active = bool(
                np.linalg.norm(los_rate_guidance_acceleration) > 1e-12
            )
        if self.bounded_velocity_mode:
            # Opt-in engineering outer loop. The original paper algebra remains
            # exact when these limits are disabled.
            stable_base_acceleration = limit_vector_norm(
                -self.parameters.c2 * velocity_error
                + los_rate_guidance_acceleration,
                self.maximum_command_acceleration_mps2,
            )
        else:
            stable_base_acceleration = (
                -self.parameters.c1 * relative_velocity
                - self.parameters.c2 * velocity_error
                - relative_position
            )
        equation_15_bracket = stable_base_acceleration - fov.fov_tangent
        paper_force = (
            -vector3(aerodynamic_force_world)
            - vector3(gravity_force_world)
            - self.parameters.mass * equation_15_bracket
        )
        corrected_acceleration = stable_base_acceleration + fov.fov_tangent
        if self.bounded_velocity_mode:
            corrected_acceleration = limit_vector_norm(
                corrected_acceleration, self.maximum_command_acceleration_mps2
            )
        corrected_force = (
            -vector3(aerodynamic_force_world)
            - vector3(gravity_force_world)
            + self.parameters.mass * corrected_acceleration
        )
        selected_force = (
            paper_force
            if self.force_law_mode == PAPER_EQUATION_15
            else corrected_force
        )

        desired_attitude = desired_attitude_from_direction(rotation_world_body, selected_force)
        attitude_error = 0.5 * vee(
            desired_attitude.T.dot(rotation_world_body)
            - rotation_world_body.T.dot(desired_attitude)
        )
        omega_attitude = -self.attitude_tracking_gain * attitude_error
        # Engineering-only long-axis decoupling.  Keep the narrow-axis FOV
        # correction at full paper strength: reducing the combined vector can
        # let a target immediately escape the +/-1 degree region.  The default
        # long-axis weight of one preserves the controller exactly.  A smaller
        # explicit weight uses the much wider vertical FOV to avoid cancelling
        # the pitch rate needed for high-speed acceleration; the downstream
        # vertical visibility guard retains the physical image-boundary gate.
        long_axis_for_control = fov.axis_long_world
        if self.formula_mode == SYMMETRIC_GRADIENT and fov.s_long < 0.0:
            long_axis_for_control = -long_axis_for_control
        omega_long_world = self.parameters.c_omega * fov.k_h * np.cross(
            fov.los_world, long_axis_for_control
        )
        omega_long_body = rotation_world_body.T.dot(omega_long_world)
        omega_tight_body = fov.omega_los_body - omega_long_body
        effective_long_axis_weight = self.long_axis_fov_angular_rate_weight
        if self.adaptive_long_axis_fov_weight:
            long_axis_angle = math.asin(min(1.0, fov.z1))
            blend = (
                (long_axis_angle - self.long_axis_fov_recovery_start_rad)
                / (
                    self.long_axis_fov_full_recovery_rad
                    - self.long_axis_fov_recovery_start_rad
                )
            )
            blend = min(1.0, max(0.0, blend))
            # Smoothstep avoids a pitch-rate discontinuity at either edge of
            # the speed-priority/recovery transition.
            blend = blend * blend * (3.0 - 2.0 * blend)
            effective_long_axis_weight += (
                1.0 - self.long_axis_fov_angular_rate_weight
            ) * blend
        desired_body_rate = (
            effective_long_axis_weight * omega_long_body
            + omega_tight_body
            + omega_attitude
        )
        return ControlResult(
            desired_relative_velocity=desired_relative_velocity,
            velocity_error=velocity_error,
            paper_force_world=paper_force,
            corrected_force_world=corrected_force,
            selected_force_world=selected_force,
            corrected_acceleration_world=corrected_acceleration,
            desired_attitude=desired_attitude,
            desired_body_rate=desired_body_rate,
            los_rate_guidance_acceleration_world=los_rate_guidance_acceleration,
            los_rate_guidance_active=los_rate_guidance_active,
            los_rate_guidance_range_scale=los_rate_guidance_range_scale,
            effective_long_axis_fov_angular_rate_weight=(
                effective_long_axis_weight
            ),
            fov=fov,
            force_law_mode=self.force_law_mode,
        )
