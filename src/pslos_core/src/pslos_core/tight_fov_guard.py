from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TightFOVGuardParameters:
    activation_rad: float = math.radians(0.5)
    rate_gain: float = 30.0
    maximum_guarded_rate_rad_s: float = 0.25
    maximum_feature_age_s: float = 0.12
    derivative_enabled: bool = False
    prediction_horizon_s: float = 0.05
    derivative_filter_tau_s: float = 0.08
    derivative_reset_gap_s: float = 0.12
    maximum_raw_derivative_s_inv: float = 0.25
    maximum_derivative_rate_rad_s: float = 0.20
    los_rate_feedforward_enabled: bool = False
    los_rate_feedforward_gain: float = 1.0
    maximum_los_rate_feedforward_rad_s: float = 0.70
    body_rate_prediction_enabled: bool = False
    body_rate_prediction_horizon_s: float = 0.15
    body_rate_deadband_rad_s: float = 0.03

    def validate(self, acceptance_limit_rad):
        acceptance_limit_rad = float(acceptance_limit_rad)
        if not math.isfinite(acceptance_limit_rad) or not (
            0.0 < acceptance_limit_rad < 0.5 * math.pi
        ):
            raise ValueError("tight FOV acceptance limit must be finite and in (0, pi/2)")
        if not 0.0 < self.activation_rad < acceptance_limit_rad:
            raise ValueError("tight FOV guard activation must be inside the acceptance limit")
        if self.rate_gain <= 0.0 or not math.isfinite(self.rate_gain):
            raise ValueError("tight FOV guard rate gain must be finite and positive")
        if self.maximum_guarded_rate_rad_s <= 0.0 or not math.isfinite(
            self.maximum_guarded_rate_rad_s
        ):
            raise ValueError("tight FOV guarded-rate limit must be finite and positive")
        if self.maximum_feature_age_s <= 0.0 or not math.isfinite(
            self.maximum_feature_age_s
        ):
            raise ValueError("tight FOV guard feature age must be finite and positive")
        if self.prediction_horizon_s < 0.0 or not math.isfinite(
            self.prediction_horizon_s
        ):
            raise ValueError("tight FOV prediction horizon must be finite and nonnegative")
        if self.derivative_filter_tau_s < 0.0 or not math.isfinite(
            self.derivative_filter_tau_s
        ):
            raise ValueError("tight FOV derivative filter tau must be finite and nonnegative")
        if self.derivative_reset_gap_s <= 0.0 or not math.isfinite(
            self.derivative_reset_gap_s
        ):
            raise ValueError("tight FOV derivative reset gap must be finite and positive")
        if self.maximum_raw_derivative_s_inv <= 0.0 or not math.isfinite(
            self.maximum_raw_derivative_s_inv
        ):
            raise ValueError("tight FOV raw derivative limit must be finite and positive")
        if self.maximum_derivative_rate_rad_s <= 0.0 or not math.isfinite(
            self.maximum_derivative_rate_rad_s
        ):
            raise ValueError("tight FOV derivative-rate limit must be finite and positive")
        if self.los_rate_feedforward_gain < 0.0 or not math.isfinite(
            self.los_rate_feedforward_gain
        ):
            raise ValueError(
                "tight FOV LOS-rate feedforward gain must be finite and nonnegative"
            )
        if self.maximum_los_rate_feedforward_rad_s <= 0.0 or not math.isfinite(
            self.maximum_los_rate_feedforward_rad_s
        ):
            raise ValueError(
                "tight FOV LOS-rate feedforward limit must be finite and positive"
            )
        if self.body_rate_prediction_horizon_s < 0.0 or not math.isfinite(
            self.body_rate_prediction_horizon_s
        ):
            raise ValueError(
                "tight FOV body-rate prediction horizon must be finite and nonnegative"
            )
        if self.body_rate_deadband_rad_s < 0.0 or not math.isfinite(
            self.body_rate_deadband_rad_s
        ):
            raise ValueError(
                "tight FOV body-rate deadband must be finite and nonnegative"
            )


@dataclass(frozen=True)
class TightFOVGuardResult:
    body_rate_rad_s: np.ndarray
    intervention_body_rate_rad_s: np.ndarray
    requested_tight_rate_rad_s: float
    measured_s_tight: float
    measured_margin_tight: float
    filtered_s_tight_rate_s_inv: float
    outward_s_tight_rate_s_inv: float
    derivative_rate_rad_s: float
    derivative_initialized: bool
    feature_sample_status: str
    los_rate_feedforward_initialized: bool
    external_s_tight_rate_s_inv: float
    los_rate_feedforward_yaw_rate_rad_s: float
    body_rate_prediction_initialized: bool
    body_rate_prediction_active: bool
    measured_body_yaw_rate_rad_s: float
    rotational_s_tight_rate_s_inv: float
    body_rate_prediction_interval_s: float
    body_rate_predicted_s_tight: float
    body_rate_prediction_rate_rad_s: float
    active: bool
    limited: bool


def normalized_tight_coordinate(x_normalized, y_normalized):
    x_normalized = float(x_normalized)
    y_normalized = float(y_normalized)
    if not all(math.isfinite(value) for value in (x_normalized, y_normalized)):
        raise ValueError("tight FOV guard feature must be finite")
    return x_normalized / math.sqrt(
        1.0 + x_normalized * x_normalized + y_normalized * y_normalized
    )


def apply_tight_fov_guard(
    measured_x_normalized,
    measured_y_normalized,
    acceptance_limit_rad,
    commanded_body_rate_rad_s,
    parameters,
    filtered_s_tight_rate_s_inv=0.0,
    derivative_initialized=False,
    feature_sample_status="stateless",
    raw_s_tight_rate_s_inv=None,
    measured_body_yaw_rate_rad_s=None,
    feature_age_s=0.0,
):
    parameters.validate(acceptance_limit_rad)
    measured_s_tight = normalized_tight_coordinate(
        measured_x_normalized, measured_y_normalized
    )
    commanded_rate = np.asarray(commanded_body_rate_rad_s, dtype=float).reshape(3)
    if not np.all(np.isfinite(commanded_rate)):
        raise ValueError("tight FOV guard body-rate command must be finite")
    filtered_rate = float(filtered_s_tight_rate_s_inv)
    if not math.isfinite(filtered_rate):
        raise ValueError("tight FOV guard derivative must be finite")
    raw_rate = filtered_rate if raw_s_tight_rate_s_inv is None else float(
        raw_s_tight_rate_s_inv
    )
    if not math.isfinite(raw_rate):
        raise ValueError("tight FOV guard raw derivative must be finite")
    feature_age = float(feature_age_s)
    if not math.isfinite(feature_age) or feature_age < 0.0:
        raise ValueError("tight FOV guard feature age must be finite and nonnegative")

    # The PS-LOS tight coordinate is the normalized camera ray x component.
    acceptance_limit = math.sin(float(acceptance_limit_rad))
    activation_limit = math.sin(parameters.activation_rad)
    measured_margin = acceptance_limit - abs(measured_s_tight)
    excess = max(0.0, abs(measured_s_tight) - activation_limit)
    tight_direction = float(np.sign(measured_s_tight))
    proportional_excess = excess
    predicted_excess_growth = 0.0
    phase_rate = filtered_rate
    if parameters.derivative_enabled and derivative_initialized:
        predicted_s_tight_filtered = (
            measured_s_tight + parameters.prediction_horizon_s * filtered_rate
        )
        predicted_s_tight_raw = (
            measured_s_tight + parameters.prediction_horizon_s * raw_rate
        )
        predicted_raw_direction = float(np.sign(predicted_s_tight_raw))
        predicted_raw_excess = max(
            0.0, abs(predicted_s_tight_raw) - activation_limit
        )
        crosses_opposite_activation = bool(
            excess > 0.0
            and measured_margin > 0.0
            and predicted_raw_excess > 0.0
            and predicted_raw_direction != 0.0
            and predicted_raw_direction != tight_direction
        )
        if crosses_opposite_activation:
            # Brake for the boundary the feature will reach, not the side it is
            # leaving. Current-side proportional error cannot be transferred.
            tight_direction = predicted_raw_direction
            proportional_excess = 0.0
            predicted_excess_growth = predicted_raw_excess
            phase_rate = raw_rate
        else:
            predicted_filtered_direction = float(
                np.sign(predicted_s_tight_filtered)
            )
            predicted_filtered_excess = max(
                0.0, abs(predicted_s_tight_filtered) - activation_limit
            )
            if predicted_filtered_direction == tight_direction:
                predicted_excess_growth = max(
                    0.0, predicted_filtered_excess - excess
                )

    outward_rate = max(0.0, tight_direction * phase_rate)
    derivative_rate = 0.0
    derivative_rate_was_capped = False
    if parameters.derivative_enabled and derivative_initialized:
        unbounded_derivative_rate = (
            parameters.rate_gain * predicted_excess_growth
        )
        derivative_rate = min(
            parameters.maximum_derivative_rate_rad_s,
            unbounded_derivative_rate,
        )
        derivative_rate_was_capped = not math.isclose(
            derivative_rate, unbounded_derivative_rate
        )
    restoring_rate = parameters.rate_gain * proportional_excess + derivative_rate
    requested_tight_rate = -tight_direction * restoring_rate
    bounded_requested_rate = float(
        np.clip(
            requested_tight_rate,
            -parameters.maximum_guarded_rate_rad_s,
            parameters.maximum_guarded_rate_rad_s,
        )
    )
    guarded_rate = commanded_rate.copy()
    los_rate_feedforward_initialized = False
    external_s_tight_rate = math.nan
    los_rate_feedforward_yaw_rate = 0.0
    los_rate_feedforward_limited = False
    ray_z = 1.0 / math.sqrt(
        1.0
        + float(measured_x_normalized) ** 2
        + float(measured_y_normalized) ** 2
    )
    if (
        parameters.los_rate_feedforward_enabled
        and derivative_initialized
        and measured_body_yaw_rate_rad_s is not None
    ):
        measured_yaw_rate = float(measured_body_yaw_rate_rad_s)
        if not math.isfinite(measured_yaw_rate):
            raise ValueError("tight FOV measured body yaw rate must be finite")
        # The measured image rate contains both target/translation motion and
        # camera rotation.  Remove the measured rotational component, then
        # command the body yaw rate that cancels the remaining apparent LOS
        # motion.  This is an opt-in iris/PX4 engineering feedforward; it uses
        # only timestamped pixels and IMU body rate, never target truth/range.
        rotational_rate = ray_z * measured_yaw_rate
        external_s_tight_rate = filtered_rate - rotational_rate
        unbounded_feedforward = (
            -parameters.los_rate_feedforward_gain
            * external_s_tight_rate
            / max(ray_z, 1e-9)
        )
        los_rate_feedforward_yaw_rate = float(
            np.clip(
                unbounded_feedforward,
                -parameters.maximum_los_rate_feedforward_rad_s,
                parameters.maximum_los_rate_feedforward_rad_s,
            )
        )
        los_rate_feedforward_limited = not math.isclose(
            los_rate_feedforward_yaw_rate, unbounded_feedforward
        )
        guarded_rate[2] += los_rate_feedforward_yaw_rate
        los_rate_feedforward_initialized = True
    pre_proportional_guard_yaw_rate = float(guarded_rate[2])
    guard_active = restoring_rate > 0.0
    if guard_active:
        # Keep useful restoring commands, but reject outward, weak, or excessive rates.
        if tight_direction > 0.0:
            guarded_rate[2] = np.clip(
                pre_proportional_guard_yaw_rate,
                -parameters.maximum_guarded_rate_rad_s,
                bounded_requested_rate,
            )
        else:
            guarded_rate[2] = np.clip(
                pre_proportional_guard_yaw_rate,
                bounded_requested_rate,
                parameters.maximum_guarded_rate_rad_s,
            )

    body_rate_prediction_initialized = False
    body_rate_prediction_active = False
    measured_body_yaw_rate = math.nan
    rotational_s_tight_rate = math.nan
    body_rate_prediction_interval = math.nan
    body_rate_predicted_s_tight = math.nan
    body_rate_prediction_rate = 0.0
    body_rate_prediction_limited = False
    if (
        parameters.body_rate_prediction_enabled
        and measured_body_yaw_rate_rad_s is not None
    ):
        measured_body_yaw_rate = float(measured_body_yaw_rate_rad_s)
        if not math.isfinite(measured_body_yaw_rate):
            raise ValueError("tight FOV measured body yaw rate must be finite")
        body_rate_prediction_initialized = True
        effective_yaw_rate = (
            measured_body_yaw_rate
            if abs(measured_body_yaw_rate)
            >= parameters.body_rate_deadband_rad_s
            else 0.0
        )
        # For this camera mounting, positive body yaw moves optical ray x positive.
        rotational_s_tight_rate = ray_z * effective_yaw_rate
        body_rate_prediction_interval = (
            feature_age + parameters.body_rate_prediction_horizon_s
        )
        body_rate_predicted_s_tight = (
            measured_s_tight
            + body_rate_prediction_interval * rotational_s_tight_rate
        )
        predicted_direction = float(np.sign(body_rate_predicted_s_tight))
        predicted_excess = max(
            0.0, abs(body_rate_predicted_s_tight) - activation_limit
        )
        moving_outward_at_prediction = (
            predicted_direction != 0.0
            and predicted_direction * rotational_s_tight_rate > 0.0
        )
        crosses_opposite_side = bool(
            measured_s_tight * body_rate_predicted_s_tight < 0.0
        )
        observed_still_moving_outward_on_current_side = bool(
            derivative_initialized
            and measured_s_tight * filtered_rate > 0.0
        )
        body_rate_prediction_active = bool(
            predicted_excess > 0.0
            and moving_outward_at_prediction
            # At high target angular rate, body rotation alone can predict a
            # center crossing while the observed feature is still escaping on
            # its current side.  Braking the restoring command in that phase
            # loses the target; wait until the image trend confirms the cross.
            and not (
                crosses_opposite_side
                and observed_still_moving_outward_on_current_side
            )
        )
        if body_rate_prediction_active:
            # Project onto commands that brake the predicted outward rotation.
            # A prediction may reject acceleration, but never creates a reverse-rate floor.
            input_guarded_yaw_rate = float(guarded_rate[2])
            if predicted_direction > 0.0:
                guarded_rate[2] = np.clip(
                    input_guarded_yaw_rate,
                    -parameters.maximum_guarded_rate_rad_s,
                    0.0,
                )
                body_rate_prediction_limited = (
                    input_guarded_yaw_rate
                    < -parameters.maximum_guarded_rate_rad_s
                )
            else:
                guarded_rate[2] = np.clip(
                    input_guarded_yaw_rate,
                    0.0,
                    parameters.maximum_guarded_rate_rad_s,
                )
                body_rate_prediction_limited = (
                    input_guarded_yaw_rate
                    > parameters.maximum_guarded_rate_rad_s
                )
            body_rate_prediction_rate = float(
                guarded_rate[2] - input_guarded_yaw_rate
            )
    base_rate_was_capped = (
        guard_active
        and pre_proportional_guard_yaw_rate * tight_direction < 0.0
        and abs(pre_proportional_guard_yaw_rate)
        > parameters.maximum_guarded_rate_rad_s
    )
    return TightFOVGuardResult(
        body_rate_rad_s=guarded_rate,
        intervention_body_rate_rad_s=guarded_rate - commanded_rate,
        requested_tight_rate_rad_s=requested_tight_rate,
        measured_s_tight=measured_s_tight,
        measured_margin_tight=measured_margin,
        filtered_s_tight_rate_s_inv=filtered_rate,
        outward_s_tight_rate_s_inv=outward_rate,
        derivative_rate_rad_s=derivative_rate,
        derivative_initialized=bool(derivative_initialized),
        feature_sample_status=str(feature_sample_status),
        los_rate_feedforward_initialized=los_rate_feedforward_initialized,
        external_s_tight_rate_s_inv=external_s_tight_rate,
        los_rate_feedforward_yaw_rate_rad_s=los_rate_feedforward_yaw_rate,
        body_rate_prediction_initialized=body_rate_prediction_initialized,
        body_rate_prediction_active=body_rate_prediction_active,
        measured_body_yaw_rate_rad_s=measured_body_yaw_rate,
        rotational_s_tight_rate_s_inv=rotational_s_tight_rate,
        body_rate_prediction_interval_s=body_rate_prediction_interval,
        body_rate_predicted_s_tight=body_rate_predicted_s_tight,
        body_rate_prediction_rate_rad_s=body_rate_prediction_rate,
        active=(
            guard_active
            or body_rate_prediction_active
            or (
                los_rate_feedforward_initialized
                and not math.isclose(los_rate_feedforward_yaw_rate, 0.0)
            )
        ),
        limited=(
            guard_active
            and (
                base_rate_was_capped
                or not math.isclose(bounded_requested_rate, requested_tight_rate)
                or derivative_rate_was_capped
            )
        )
        or body_rate_prediction_limited
        or los_rate_feedforward_limited,
    )


class TightFOVGuard:
    """Adds phase lead from distinct, timestamped feature observations."""

    def __init__(self, acceptance_limit_rad, parameters):
        parameters.validate(acceptance_limit_rad)
        self.acceptance_limit_rad = float(acceptance_limit_rad)
        self.parameters = parameters
        self.reset()

    def reset(self):
        self.last_feature_stamp_s = None
        self.last_measured_s_tight = None
        self.filtered_s_tight_rate_s_inv = 0.0
        self.raw_s_tight_rate_s_inv = 0.0
        self.derivative_initialized = False

    def apply(
        self,
        measured_x_normalized,
        measured_y_normalized,
        feature_stamp_s,
        commanded_body_rate_rad_s,
        measured_body_yaw_rate_rad_s=None,
        feature_age_s=0.0,
    ):
        feature_stamp_s = float(feature_stamp_s)
        if not math.isfinite(feature_stamp_s):
            raise ValueError("tight FOV guard feature stamp must be finite")
        measured_s_tight = normalized_tight_coordinate(
            measured_x_normalized, measured_y_normalized
        )
        sample_status = "duplicate"
        if self.last_feature_stamp_s is None:
            sample_status = "initialized"
            self.last_feature_stamp_s = feature_stamp_s
            self.last_measured_s_tight = measured_s_tight
        else:
            dt = feature_stamp_s - self.last_feature_stamp_s
            if dt < 0.0:
                sample_status = "reset_non_monotonic"
                self.reset()
                self.last_feature_stamp_s = feature_stamp_s
                self.last_measured_s_tight = measured_s_tight
            elif dt > self.parameters.derivative_reset_gap_s:
                sample_status = "reset_gap"
                self.reset()
                self.last_feature_stamp_s = feature_stamp_s
                self.last_measured_s_tight = measured_s_tight
            elif dt == 0.0 and not math.isclose(
                measured_s_tight,
                self.last_measured_s_tight,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                sample_status = "reset_same_stamp_changed"
                self.filtered_s_tight_rate_s_inv = 0.0
                self.raw_s_tight_rate_s_inv = 0.0
                self.derivative_initialized = False
                self.last_measured_s_tight = measured_s_tight
            elif dt > 0.0:
                sample_status = "updated"
                raw_rate = np.clip(
                    (measured_s_tight - self.last_measured_s_tight) / dt,
                    -self.parameters.maximum_raw_derivative_s_inv,
                    self.parameters.maximum_raw_derivative_s_inv,
                )
                self.raw_s_tight_rate_s_inv = float(raw_rate)
                if self.parameters.derivative_filter_tau_s == 0.0:
                    self.filtered_s_tight_rate_s_inv = float(raw_rate)
                else:
                    alpha = dt / (self.parameters.derivative_filter_tau_s + dt)
                    self.filtered_s_tight_rate_s_inv += alpha * (
                        float(raw_rate) - self.filtered_s_tight_rate_s_inv
                    )
                self.derivative_initialized = True
                self.last_feature_stamp_s = feature_stamp_s
                self.last_measured_s_tight = measured_s_tight

        return apply_tight_fov_guard(
            measured_x_normalized,
            measured_y_normalized,
            self.acceptance_limit_rad,
            commanded_body_rate_rad_s,
            self.parameters,
            filtered_s_tight_rate_s_inv=self.filtered_s_tight_rate_s_inv,
            raw_s_tight_rate_s_inv=self.raw_s_tight_rate_s_inv,
            derivative_initialized=self.derivative_initialized,
            feature_sample_status=sample_status,
            measured_body_yaw_rate_rad_s=measured_body_yaw_rate_rad_s,
            feature_age_s=feature_age_s,
        )


@dataclass(frozen=True)
class PixelAxisGuardParameters:
    """Limits for one raw normalized image coordinate."""

    acceptance_lower_normalized: float
    acceptance_upper_normalized: float
    activation_lower_normalized: float
    activation_upper_normalized: float
    rate_gain: float = 2.0
    maximum_guarded_rate_rad_s: float = 0.70
    derivative_enabled: bool = False
    prediction_horizon_s: float = 0.05
    derivative_filter_tau_s: float = 0.08
    derivative_reset_gap_s: float = 0.12
    maximum_raw_derivative_s_inv: float = 2.0
    maximum_derivative_rate_rad_s: float = 0.20

    def validate(self):
        limits = (
            self.acceptance_lower_normalized,
            self.activation_lower_normalized,
            self.activation_upper_normalized,
            self.acceptance_upper_normalized,
        )
        if not all(math.isfinite(float(value)) for value in limits):
            raise ValueError("pixel-axis guard limits must be finite")
        if not all(left < right for left, right in zip(limits, limits[1:])):
            raise ValueError(
                "pixel-axis guard limits must satisfy acceptance_lower < "
                "activation_lower < activation_upper < acceptance_upper"
            )
        if self.rate_gain <= 0.0 or not math.isfinite(self.rate_gain):
            raise ValueError("pixel-axis guard rate gain must be finite and positive")
        if self.maximum_guarded_rate_rad_s <= 0.0 or not math.isfinite(
            self.maximum_guarded_rate_rad_s
        ):
            raise ValueError(
                "pixel-axis guarded-rate limit must be finite and positive"
            )
        if self.prediction_horizon_s < 0.0 or not math.isfinite(
            self.prediction_horizon_s
        ):
            raise ValueError(
                "pixel-axis prediction horizon must be finite and nonnegative"
            )
        if self.derivative_filter_tau_s < 0.0 or not math.isfinite(
            self.derivative_filter_tau_s
        ):
            raise ValueError(
                "pixel-axis derivative filter tau must be finite and nonnegative"
            )
        if self.derivative_reset_gap_s <= 0.0 or not math.isfinite(
            self.derivative_reset_gap_s
        ):
            raise ValueError(
                "pixel-axis derivative reset gap must be finite and positive"
            )
        if self.maximum_raw_derivative_s_inv <= 0.0 or not math.isfinite(
            self.maximum_raw_derivative_s_inv
        ):
            raise ValueError(
                "pixel-axis raw derivative limit must be finite and positive"
            )
        if self.maximum_derivative_rate_rad_s <= 0.0 or not math.isfinite(
            self.maximum_derivative_rate_rad_s
        ):
            raise ValueError(
                "pixel-axis derivative-rate limit must be finite and positive"
            )


@dataclass(frozen=True)
class PixelAxisGuardResult:
    body_rate_rad_s: np.ndarray
    intervention_body_rate_rad_s: np.ndarray
    requested_body_rate_rad_s: float
    measured_coordinate_normalized: float
    measured_margin_normalized: float
    filtered_coordinate_rate_s_inv: float
    outward_coordinate_rate_s_inv: float
    derivative_rate_rad_s: float
    derivative_initialized: bool
    feature_sample_status: str
    active_boundary: str
    body_rate_axis_index: int
    restoring_sign: float
    within_acceptance: bool
    active: bool
    intervened: bool
    limited: bool


def _validate_pixel_axis_mapping(body_rate_axis_index, restoring_sign):
    if isinstance(body_rate_axis_index, bool) or body_rate_axis_index not in (0, 1, 2):
        raise ValueError("pixel-axis guard body-rate axis index must be 0, 1, or 2")
    restoring_sign = float(restoring_sign)
    if restoring_sign not in (-1.0, 1.0):
        raise ValueError("pixel-axis guard restoring sign must be -1 or +1")
    return int(body_rate_axis_index), restoring_sign


def apply_pixel_axis_guard(
    measured_coordinate_normalized,
    commanded_body_rate_rad_s,
    parameters,
    body_rate_axis_index,
    restoring_sign,
    filtered_coordinate_rate_s_inv=0.0,
    derivative_initialized=False,
    feature_sample_status="stateless",
):
    """Guard one raw normalized pixel coordinate with a bounded body rate.

    ``restoring_sign`` maps an upper-image-boundary violation to the body-rate
    sign that moves the feature back toward the image center. For the optical
    camera convention x-right/y-down/z-forward used here, x-to-yaw uses -1 and
    y-to-pitch uses +1.
    """

    parameters.validate()
    body_rate_axis_index, restoring_sign = _validate_pixel_axis_mapping(
        body_rate_axis_index, restoring_sign
    )
    measured_coordinate = float(measured_coordinate_normalized)
    if not math.isfinite(measured_coordinate):
        raise ValueError("pixel-axis guard coordinate must be finite")
    commanded_rate = np.asarray(commanded_body_rate_rad_s, dtype=float).reshape(3)
    if not np.all(np.isfinite(commanded_rate)):
        raise ValueError("pixel-axis guard body-rate command must be finite")
    filtered_rate = float(filtered_coordinate_rate_s_inv)
    if not math.isfinite(filtered_rate):
        raise ValueError("pixel-axis guard derivative must be finite")

    upper_excess = max(
        0.0, measured_coordinate - parameters.activation_upper_normalized
    )
    lower_excess = max(
        0.0, parameters.activation_lower_normalized - measured_coordinate
    )
    boundary_direction = 0.0
    active_boundary = "none"
    excess = 0.0
    if upper_excess > 0.0:
        boundary_direction = 1.0
        active_boundary = "upper"
        excess = upper_excess
    elif lower_excess > 0.0:
        boundary_direction = -1.0
        active_boundary = "lower"
        excess = lower_excess

    predicted_excess = excess
    if parameters.derivative_enabled and derivative_initialized:
        predicted_coordinate = (
            measured_coordinate + parameters.prediction_horizon_s * filtered_rate
        )
        predicted_upper_excess = max(
            0.0, predicted_coordinate - parameters.activation_upper_normalized
        )
        predicted_lower_excess = max(
            0.0, parameters.activation_lower_normalized - predicted_coordinate
        )
        upper_growth = predicted_upper_excess - upper_excess
        lower_growth = predicted_lower_excess - lower_excess
        if boundary_direction == 0.0 and max(upper_growth, lower_growth) > 0.0:
            if upper_growth >= lower_growth:
                boundary_direction = 1.0
                active_boundary = "upper"
                predicted_excess = predicted_upper_excess
            else:
                boundary_direction = -1.0
                active_boundary = "lower"
                predicted_excess = predicted_lower_excess
        elif boundary_direction > 0.0:
            predicted_excess = predicted_upper_excess
        elif boundary_direction < 0.0:
            predicted_excess = predicted_lower_excess

    outward_rate = max(0.0, boundary_direction * filtered_rate)
    derivative_rate = 0.0
    derivative_rate_was_capped = False
    if parameters.derivative_enabled and derivative_initialized:
        unbounded_derivative_rate = parameters.rate_gain * max(
            0.0, predicted_excess - excess
        )
        derivative_rate = min(
            parameters.maximum_derivative_rate_rad_s,
            unbounded_derivative_rate,
        )
        derivative_rate_was_capped = not math.isclose(
            derivative_rate, unbounded_derivative_rate
        )

    restoring_rate = parameters.rate_gain * excess + derivative_rate
    restoring_direction = restoring_sign * boundary_direction
    requested_rate = restoring_direction * restoring_rate
    bounded_restoring_rate = min(
        restoring_rate, parameters.maximum_guarded_rate_rad_s
    )
    guarded_rate = commanded_rate.copy()
    guard_active = restoring_rate > 0.0
    base_rate_was_capped = False
    if guard_active:
        aligned_command = restoring_direction * commanded_rate[body_rate_axis_index]
        bounded_aligned_command = float(
            np.clip(
                aligned_command,
                bounded_restoring_rate,
                parameters.maximum_guarded_rate_rad_s,
            )
        )
        guarded_rate[body_rate_axis_index] = (
            restoring_direction * bounded_aligned_command
        )
        base_rate_was_capped = (
            aligned_command > parameters.maximum_guarded_rate_rad_s
        )

    intervention = guarded_rate - commanded_rate
    measured_margin = min(
        measured_coordinate - parameters.acceptance_lower_normalized,
        parameters.acceptance_upper_normalized - measured_coordinate,
    )
    return PixelAxisGuardResult(
        body_rate_rad_s=guarded_rate,
        intervention_body_rate_rad_s=intervention,
        requested_body_rate_rad_s=requested_rate,
        measured_coordinate_normalized=measured_coordinate,
        measured_margin_normalized=measured_margin,
        filtered_coordinate_rate_s_inv=filtered_rate,
        outward_coordinate_rate_s_inv=outward_rate,
        derivative_rate_rad_s=derivative_rate,
        derivative_initialized=bool(derivative_initialized),
        feature_sample_status=str(feature_sample_status),
        active_boundary=active_boundary,
        body_rate_axis_index=body_rate_axis_index,
        restoring_sign=restoring_sign,
        within_acceptance=measured_margin >= 0.0,
        active=guard_active,
        intervened=not np.array_equal(intervention, np.zeros(3)),
        limited=guard_active
        and (
            base_rate_was_capped
            or restoring_rate > parameters.maximum_guarded_rate_rad_s
            or derivative_rate_was_capped
        ),
    )


class PixelAxisGuard:
    """Stateful derivative lead for one raw normalized pixel coordinate.

    Feature freshness is deliberately a caller responsibility. A caller must
    reject stale observations and call ``reset`` before reusing this guard.
    """

    def __init__(self, parameters, body_rate_axis_index, restoring_sign):
        parameters.validate()
        body_rate_axis_index, restoring_sign = _validate_pixel_axis_mapping(
            body_rate_axis_index, restoring_sign
        )
        self.parameters = parameters
        self.body_rate_axis_index = body_rate_axis_index
        self.restoring_sign = restoring_sign
        self.reset()

    def reset(self):
        self.last_feature_stamp_s = None
        self.last_measured_coordinate_normalized = None
        self.filtered_coordinate_rate_s_inv = 0.0
        self.derivative_initialized = False

    def apply(
        self,
        measured_coordinate_normalized,
        feature_stamp_s,
        commanded_body_rate_rad_s,
    ):
        feature_stamp_s = float(feature_stamp_s)
        if not math.isfinite(feature_stamp_s):
            raise ValueError("pixel-axis guard feature stamp must be finite")
        measured_coordinate = float(measured_coordinate_normalized)
        if not math.isfinite(measured_coordinate):
            raise ValueError("pixel-axis guard coordinate must be finite")

        sample_status = "duplicate"
        if self.last_feature_stamp_s is None:
            sample_status = "initialized"
            self.last_feature_stamp_s = feature_stamp_s
            self.last_measured_coordinate_normalized = measured_coordinate
        else:
            dt = feature_stamp_s - self.last_feature_stamp_s
            if dt < 0.0:
                sample_status = "reset_non_monotonic"
                self.reset()
                self.last_feature_stamp_s = feature_stamp_s
                self.last_measured_coordinate_normalized = measured_coordinate
            elif dt > self.parameters.derivative_reset_gap_s:
                sample_status = "reset_gap"
                self.reset()
                self.last_feature_stamp_s = feature_stamp_s
                self.last_measured_coordinate_normalized = measured_coordinate
            elif dt == 0.0 and not math.isclose(
                measured_coordinate,
                self.last_measured_coordinate_normalized,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                sample_status = "reset_same_stamp_changed"
                self.filtered_coordinate_rate_s_inv = 0.0
                self.derivative_initialized = False
                self.last_measured_coordinate_normalized = measured_coordinate
            elif dt > 0.0:
                sample_status = "updated"
                raw_rate = np.clip(
                    (
                        measured_coordinate
                        - self.last_measured_coordinate_normalized
                    )
                    / dt,
                    -self.parameters.maximum_raw_derivative_s_inv,
                    self.parameters.maximum_raw_derivative_s_inv,
                )
                if self.parameters.derivative_filter_tau_s == 0.0:
                    self.filtered_coordinate_rate_s_inv = float(raw_rate)
                else:
                    alpha = dt / (self.parameters.derivative_filter_tau_s + dt)
                    self.filtered_coordinate_rate_s_inv += alpha * (
                        float(raw_rate) - self.filtered_coordinate_rate_s_inv
                    )
                self.derivative_initialized = True
                self.last_feature_stamp_s = feature_stamp_s
                self.last_measured_coordinate_normalized = measured_coordinate

        return apply_pixel_axis_guard(
            measured_coordinate,
            commanded_body_rate_rad_s,
            self.parameters,
            self.body_rate_axis_index,
            self.restoring_sign,
            filtered_coordinate_rate_s_inv=self.filtered_coordinate_rate_s_inv,
            derivative_initialized=self.derivative_initialized,
            feature_sample_status=sample_status,
        )
