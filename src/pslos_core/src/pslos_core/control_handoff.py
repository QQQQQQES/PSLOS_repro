from dataclasses import dataclass
import math


def normalized_feature_tight_coordinate(x_normalized, y_normalized):
    x_normalized = float(x_normalized)
    y_normalized = float(y_normalized)
    if not all(math.isfinite(value) for value in (x_normalized, y_normalized)):
        raise ValueError("handoff feature must be finite")
    return x_normalized / math.sqrt(
        1.0 + x_normalized * x_normalized + y_normalized * y_normalized
    )


@dataclass(frozen=True)
class ControlHandoffParameters:
    minimum_measurements: int = 6
    maximum_vision_age_s: float = 0.1
    maximum_feature_age_s: float = 0.1
    maximum_debug_age_s: float = 0.1
    maximum_observation_skew_s: float = 0.03
    maximum_tight_error_rad: float = math.radians(0.2)
    stable_duration_s: float = 0.2
    maximum_sample_gap_s: float = 0.08

    def validate(self):
        if self.minimum_measurements < 1:
            raise ValueError("handoff minimum measurements must be positive")
        positive_finite = (
            ("maximum vision age", self.maximum_vision_age_s),
            ("maximum feature age", self.maximum_feature_age_s),
            ("maximum debug age", self.maximum_debug_age_s),
            ("maximum observation skew", self.maximum_observation_skew_s),
            ("maximum tight error", self.maximum_tight_error_rad),
            ("stable duration", self.stable_duration_s),
            ("maximum sample gap", self.maximum_sample_gap_s),
        )
        for name, value in positive_finite:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("handoff {} must be finite and positive".format(name))
        if self.maximum_tight_error_rad >= 0.5 * math.pi:
            raise ValueError("handoff maximum tight error must be below pi/2")


@dataclass(frozen=True)
class ControlHandoffObservation:
    estimate_valid: bool
    estimate_source_is_dcekf: bool
    measurement_count: int
    vision_age_s: float
    feature_detected: bool
    feature_centered: bool
    feature_age_s: float
    feature_frame_valid: bool
    debug_valid: bool
    debug_age_s: float
    observation_skew_s: float
    estimate_s_tight: float
    measured_s_tight: float


@dataclass(frozen=True)
class ControlHandoffEvaluation:
    ready: bool
    reason: str
    tight_error_rad: float


def evaluate_control_handoff(observation, parameters):
    parameters.validate()
    if not observation.estimate_source_is_dcekf:
        return ControlHandoffEvaluation(False, "estimate_source", math.inf)
    if not observation.estimate_valid:
        return ControlHandoffEvaluation(False, "estimate_invalid", math.inf)
    if int(observation.measurement_count) < parameters.minimum_measurements:
        return ControlHandoffEvaluation(False, "measurement_count", math.inf)
    if not math.isfinite(observation.vision_age_s) or not (
        0.0 <= observation.vision_age_s <= parameters.maximum_vision_age_s
    ):
        return ControlHandoffEvaluation(False, "vision_age", math.inf)
    if not observation.feature_detected or not observation.feature_frame_valid:
        return ControlHandoffEvaluation(False, "feature_invalid", math.inf)
    if not observation.feature_centered:
        return ControlHandoffEvaluation(False, "feature_center", math.inf)
    if not math.isfinite(observation.feature_age_s) or not (
        0.0 <= observation.feature_age_s <= parameters.maximum_feature_age_s
    ):
        return ControlHandoffEvaluation(False, "feature_age", math.inf)
    if not observation.debug_valid:
        return ControlHandoffEvaluation(False, "debug_invalid", math.inf)
    if not math.isfinite(observation.debug_age_s) or not (
        0.0 <= observation.debug_age_s <= parameters.maximum_debug_age_s
    ):
        return ControlHandoffEvaluation(False, "debug_age", math.inf)
    if not math.isfinite(observation.observation_skew_s) or not (
        0.0
        <= observation.observation_skew_s
        <= parameters.maximum_observation_skew_s
    ):
        return ControlHandoffEvaluation(False, "observation_skew", math.inf)
    if not all(
        math.isfinite(value)
        for value in (observation.estimate_s_tight, observation.measured_s_tight)
    ):
        return ControlHandoffEvaluation(False, "tight_coordinate", math.inf)

    if (
        abs(observation.estimate_s_tight) > 1.0 + 1e-12
        or abs(observation.measured_s_tight) > 1.0 + 1e-12
    ):
        return ControlHandoffEvaluation(False, "tight_coordinate", math.inf)
    estimate_angle = math.asin(max(-1.0, min(1.0, observation.estimate_s_tight)))
    measured_angle = math.asin(max(-1.0, min(1.0, observation.measured_s_tight)))
    tight_error = abs(estimate_angle - measured_angle)
    if tight_error > parameters.maximum_tight_error_rad:
        return ControlHandoffEvaluation(False, "tight_error", tight_error)
    return ControlHandoffEvaluation(True, "ready", tight_error)


class ControlHandoffGate:
    def __init__(self, parameters):
        parameters.validate()
        self.parameters = parameters
        self.stable_start_s = None
        self.last_sample_stamp_s = None
        self.maximum_stable_duration_s = 0.0
        self.reset_count = 0
        self.last_reason = "not_evaluated"

    def update(self, stamp_s, observation):
        stamp_s = float(stamp_s)
        if not math.isfinite(stamp_s):
            raise ValueError("handoff timestamp must be finite")
        evaluation = evaluate_control_handoff(observation, self.parameters)
        if not evaluation.ready:
            if self.stable_start_s is not None:
                self.reset_count += 1
            self.stable_start_s = None
            self.last_sample_stamp_s = stamp_s
            self.last_reason = evaluation.reason
            return False, 0.0, evaluation

        sample_gap_s = None
        if self.last_sample_stamp_s is not None:
            sample_gap_s = stamp_s - self.last_sample_stamp_s
        sample_gap_invalid = (
            sample_gap_s is not None
            and (
                sample_gap_s <= 0.0
                or sample_gap_s
                > self.parameters.maximum_sample_gap_s + 1e-12
            )
        )
        if self.stable_start_s is None or sample_gap_invalid:
            if self.stable_start_s is not None:
                self.reset_count += 1
            self.stable_start_s = stamp_s
        self.last_sample_stamp_s = stamp_s
        stable_duration = max(0.0, stamp_s - self.stable_start_s)
        self.maximum_stable_duration_s = max(
            self.maximum_stable_duration_s, stable_duration
        )
        self.last_reason = evaluation.reason
        return (
            stable_duration + 1e-12 >= self.parameters.stable_duration_s,
            stable_duration,
            evaluation,
        )
