from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RateReversalBoostParameters:
    gains: tuple = (0.0, 0.0, 0.0)
    damping_gains: tuple = (0.0, 0.0, 0.0)
    filter_tau_s: float = 0.04
    max_correction_rad_s: tuple = (0.35, 0.35, 0.35)
    reset_gap_s: float = 0.20
    feedback_mode: str = "legacy_reversal"
    tracking_error_gains: tuple = (0.0, 0.0, 0.0)
    axis_feedback_modes: tuple = None

    def resolved_axis_feedback_modes(self):
        if self.axis_feedback_modes is None:
            return (self.feedback_mode,) * 3
        try:
            modes = tuple(self.axis_feedback_modes)
        except TypeError as error:
            raise ValueError(
                "axis feedback modes must contain exactly three values"
            ) from error
        if len(modes) != 3:
            raise ValueError("axis feedback modes must contain exactly three values")
        return modes

    def validate(self):
        if self.feedback_mode not in ("legacy_reversal", "tracking_error"):
            raise ValueError(
                "rate feedback mode must be legacy_reversal or tracking_error"
            )
        axis_feedback_modes = self.resolved_axis_feedback_modes()
        if any(
            mode not in ("legacy_reversal", "tracking_error")
            for mode in axis_feedback_modes
        ):
            raise ValueError(
                "axis feedback modes must be legacy_reversal or tracking_error"
            )
        gains = np.asarray(self.gains, dtype=float).reshape(3)
        damping_gains = np.asarray(self.damping_gains, dtype=float).reshape(3)
        tracking_error_gains = np.asarray(
            self.tracking_error_gains, dtype=float
        ).reshape(3)
        limits = np.asarray(self.max_correction_rad_s, dtype=float).reshape(3)
        if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
            raise ValueError("reversal gains must be finite and nonnegative")
        if not np.all(np.isfinite(damping_gains)) or np.any(damping_gains < 0.0):
            raise ValueError("damping gains must be finite and nonnegative")
        if not np.all(np.isfinite(tracking_error_gains)) or np.any(
            tracking_error_gains < 0.0
        ):
            raise ValueError("tracking-error gains must be finite and nonnegative")
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("correction limits must be finite and positive")
        if self.filter_tau_s < 0.0 or not math.isfinite(self.filter_tau_s):
            raise ValueError("filter_tau_s must be finite and nonnegative")
        if self.reset_gap_s <= 0.0 or not math.isfinite(self.reset_gap_s):
            raise ValueError("reset_gap_s must be finite and positive")


@dataclass(frozen=True)
class RateReversalBoostResult:
    body_rate_rad_s: np.ndarray
    filtered_rate_rad_s: np.ndarray
    correction_rad_s: np.ndarray
    damping_correction_rad_s: np.ndarray
    reversal_correction_rad_s: np.ndarray
    tracking_error_rad_s: np.ndarray
    tracking_error_correction_rad_s: np.ndarray


class BodyRateReversalBooster:
    """Damp measured rates and add braking while a commanded reversal is pending."""

    def __init__(self, parameters):
        parameters.validate()
        self.parameters = parameters
        self.filtered_rate_rad_s = None
        self.last_stamp_s = None

    @property
    def initialized(self):
        return self.filtered_rate_rad_s is not None

    def reset(self):
        self.filtered_rate_rad_s = None
        self.last_stamp_s = None

    def update(self, measured_body_rate_rad_s, stamp_s):
        rates = np.asarray(measured_body_rate_rad_s, dtype=float).reshape(3)
        stamp_s = float(stamp_s)
        if not np.all(np.isfinite(rates)) or not math.isfinite(stamp_s):
            raise ValueError("measured body rate and stamp must be finite")

        if self.last_stamp_s is None:
            self.filtered_rate_rad_s = rates.copy()
        else:
            dt = stamp_s - self.last_stamp_s
            if dt <= 0.0 or dt > self.parameters.reset_gap_s:
                self.filtered_rate_rad_s = rates.copy()
            elif self.parameters.filter_tau_s == 0.0:
                self.filtered_rate_rad_s = rates.copy()
            else:
                alpha = dt / (self.parameters.filter_tau_s + dt)
                self.filtered_rate_rad_s += alpha * (
                    rates - self.filtered_rate_rad_s
                )
        self.last_stamp_s = stamp_s
        return self.filtered_rate_rad_s.copy()

    def compensate(self, commanded_body_rate_rad_s):
        command = np.asarray(commanded_body_rate_rad_s, dtype=float).reshape(3).copy()
        if not np.all(np.isfinite(command)):
            raise ValueError("commanded body rate must be finite")
        if not self.initialized:
            zeros = np.zeros(3)
            return RateReversalBoostResult(
                command, zeros, zeros, zeros, zeros, zeros, zeros
            )

        gains = np.asarray(self.parameters.gains, dtype=float)
        damping_gains = np.asarray(self.parameters.damping_gains, dtype=float)
        tracking_error_gains = np.asarray(
            self.parameters.tracking_error_gains, dtype=float
        )
        limits = np.asarray(self.parameters.max_correction_rad_s, dtype=float)
        tracking_error = command - self.filtered_rate_rad_s
        axis_feedback_modes = np.asarray(
            self.parameters.resolved_axis_feedback_modes(), dtype=object
        )
        legacy_axes = axis_feedback_modes == "legacy_reversal"
        tracking_axes = axis_feedback_modes == "tracking_error"

        damping_correction = damping_gains * self.filtered_rate_rad_s
        reversal_correction = gains * self.filtered_rate_rad_s
        reversal_correction[command * self.filtered_rate_rad_s > 0.0] = 0.0
        damping_correction[~legacy_axes] = 0.0
        reversal_correction[~legacy_axes] = 0.0
        legacy_correction = np.clip(
            damping_correction + reversal_correction,
            -limits,
            limits,
        )
        tracking_correction = np.clip(
            tracking_error_gains * tracking_error,
            -limits,
            limits,
        )
        tracking_correction[~tracking_axes] = 0.0
        correction = legacy_correction - tracking_correction
        return RateReversalBoostResult(
            command - correction,
            self.filtered_rate_rad_s.copy(),
            correction,
            damping_correction,
            reversal_correction,
            tracking_error,
            tracking_correction,
        )
