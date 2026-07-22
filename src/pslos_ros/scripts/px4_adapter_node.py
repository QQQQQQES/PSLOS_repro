#!/usr/bin/env python3
import math
import threading
import time

import numpy as np
import rospy
from mavros_msgs.msg import AttitudeTarget
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from pslos_core.command_safety import CommandSafetyParameters, sanitize_command
from pslos_core.geometry import quaternion_xyzw_to_rotation
from pslos_core.rate_damping import (
    BodyRateReversalBooster,
    RateReversalBoostParameters,
)
from pslos_core.tight_fov_guard import (
    PixelAxisGuard,
    PixelAxisGuardParameters,
    TightFOVGuard,
    TightFOVGuardParameters,
)
from pslos_msgs.msg import (
    AdapterDebug,
    ControlReference,
    TargetFeature,
    VisibilityGuardDebug,
)


def vector3_message(value):
    from geometry_msgs.msg import Vector3

    message = Vector3()
    message.x, message.y, message.z = (float(item) for item in value)
    return message


def vertical_visibility_bounds(image_height, fy, cy, activation_margin_px):
    """Return raw normalized y bounds for the calibrated half-open image."""
    image_height = int(image_height)
    fy = float(fy)
    cy = float(cy)
    activation_margin_px = float(activation_margin_px)
    if image_height <= 0:
        raise ValueError("visibility guard image height must be positive")
    if not all(math.isfinite(value) for value in (fy, cy, activation_margin_px)):
        raise ValueError("visibility guard calibration must be finite")
    if fy <= 0.0:
        raise ValueError("visibility guard fy must be positive")
    if not 0.0 < cy < float(image_height):
        raise ValueError("visibility guard cy must lie inside the image")
    if not 0.0 < activation_margin_px < 0.5 * float(image_height):
        raise ValueError(
            "visibility guard activation margin must be in (0, image_height/2)"
        )
    return (
        -cy / fy,
        (float(image_height) - cy) / fy,
        (activation_margin_px - cy) / fy,
        (float(image_height) - activation_margin_px - cy) / fy,
    )


def visibility_recovery_thrust_scale(
    guard_result, parameters, minimum_thrust_scale
):
    """Blend thrust down only as an active vertical recovery nears the image edge."""
    minimum_thrust_scale = float(minimum_thrust_scale)
    if (
        not math.isfinite(minimum_thrust_scale)
        or minimum_thrust_scale <= 0.0
        or minimum_thrust_scale > 1.0
    ):
        raise ValueError(
            "visibility recovery minimum thrust scale must be in (0, 1]"
        )
    if guard_result is None or not guard_result.active:
        return 1.0

    coordinate = float(guard_result.measured_coordinate_normalized)
    if guard_result.active_boundary == "upper":
        activation = float(parameters.activation_upper_normalized)
        acceptance = float(parameters.acceptance_upper_normalized)
        progress = (coordinate - activation) / (acceptance - activation)
    elif guard_result.active_boundary == "lower":
        activation = float(parameters.activation_lower_normalized)
        acceptance = float(parameters.acceptance_lower_normalized)
        progress = (activation - coordinate) / (activation - acceptance)
    else:
        return 1.0
    progress = float(np.clip(progress, 0.0, 1.0))
    return 1.0 - progress * (1.0 - minimum_thrust_scale)


def attitude_alignment_thrust_scale(
    desired_attitude_xyzw, measured_attitude_xyzw, minimum_scale
):
    """Preserve the requested world-up thrust while attitude is still lagging.

    The controller publishes the desired force magnitude.  In body-rate mode the
    vehicle attitude needs finite time to align with that force, so applying the
    full magnitude immediately can turn horizontal-acceleration demand into a
    vertical impulse.  Scale by desired/measured body-z world-up components so
    the actual axis cannot produce more upward thrust than the requested axis.
    The limiter deliberately never boosts thrust when the measured vehicle is
    tilted farther than requested.
    """
    minimum_scale = float(minimum_scale)
    if (
        not math.isfinite(minimum_scale)
        or minimum_scale <= 0.0
        or minimum_scale > 1.0
    ):
        raise ValueError("attitude thrust projection minimum scale must be in (0, 1]")
    desired_rotation = quaternion_xyzw_to_rotation(desired_attitude_xyzw)
    measured_rotation = quaternion_xyzw_to_rotation(measured_attitude_xyzw)
    desired_up_component = float(desired_rotation[2, 2])
    measured_up_component = float(measured_rotation[2, 2])
    if (
        not math.isfinite(desired_up_component)
        or not math.isfinite(measured_up_component)
    ):
        raise ValueError("attitude thrust alignment is non-finite")
    axis_alignment_scale = float(
        np.dot(desired_rotation[:, 2], measured_rotation[:, 2])
    )
    if measured_up_component <= 1e-6:
        return minimum_scale
    vertical_component_scale = desired_up_component / measured_up_component
    return float(
        np.clip(
            min(axis_alignment_scale, vertical_component_scale),
            minimum_scale,
            1.0,
        )
    )


class ControlEnableLease:
    """Tracks a renewable Bool authorization with ROS and monotonic clocks."""

    def __init__(self, required, timeout_s, wall_timeout_s=None):
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("control enable timeout must be finite and positive")
        if wall_timeout_s is None:
            wall_timeout_s = timeout_s
        if not math.isfinite(wall_timeout_s) or wall_timeout_s <= 0.0:
            raise ValueError(
                "control enable wall timeout must be finite and positive"
            )
        self.required = bool(required)
        self.timeout_s = float(timeout_s)
        self.wall_timeout_s = float(wall_timeout_s)
        self._asserted = not self.required
        self._ros_stamp_s = None
        self._wall_stamp_s = None
        self._generation = 0

    @property
    def generation(self):
        return self._generation

    def update(self, enabled, ros_time_s, wall_time_s):
        enabled = bool(enabled)
        if enabled:
            if not self._asserted:
                self._generation += 1
            self._asserted = True
            self._ros_stamp_s = float(ros_time_s)
            self._wall_stamp_s = float(wall_time_s)
        else:
            self._generation += 1
            self._asserted = False
            self._ros_stamp_s = None
            self._wall_stamp_s = None
        return self._generation

    def authorize(self, ros_time_s, wall_time_s):
        """Returns (permitted, generation, reason, newly_timed_out)."""
        if not self.required:
            return True, self._generation, "control_enable_not_required", False
        if not self._asserted:
            return False, self._generation, "control_disabled", False

        ros_age_s = float(ros_time_s) - self._ros_stamp_s
        wall_age_s = float(wall_time_s) - self._wall_stamp_s
        fresh = bool(
            math.isfinite(ros_age_s)
            and math.isfinite(wall_age_s)
            and 0.0 <= ros_age_s <= self.timeout_s
            and 0.0 <= wall_age_s <= self.wall_timeout_s
        )
        if fresh:
            return True, self._generation, "control_enabled", False

        # Revoking here makes a timeout edge observable exactly once per grant.
        self._asserted = False
        self._generation += 1
        return False, self._generation, "control_enable_timeout", True


class PX4AdapterNode:
    def __init__(self):
        self.enabled = bool(rospy.get_param("~enable_px4_output", False))
        self.require_control_enable = bool(
            rospy.get_param("~require_control_enable", False)
        )
        self.control_enable_timeout_s = float(
            rospy.get_param("~control_enable_timeout_s", 0.25)
        )
        self.control_enable_wall_timeout_s = float(
            rospy.get_param(
                "~control_enable_wall_timeout_s",
                self.control_enable_timeout_s,
            )
        )
        self.control_enable_lease = ControlEnableLease(
            self.require_control_enable,
            self.control_enable_timeout_s,
            self.control_enable_wall_timeout_s,
        )
        self.lock = threading.Lock()
        self.reference = None
        self.feature = None
        self.imu_stamp = None
        self.imu_attitude_xyzw = None
        self.latest_measured_body_yaw_rate_rad_s = None
        self.publisher = None
        self.audit_publisher = None
        self.visibility_audit_publisher = None
        self.timer = None
        self.subscriber = None
        self.control_enable_subscriber = None
        self.imu_subscriber = None
        self.feature_subscriber = None
        self.command_timeout_s = float(rospy.get_param("~command_timeout_s", 0.10))
        self.parameters = CommandSafetyParameters(
            hover_thrust=float(rospy.get_param("~hover_thrust", 0.5)),
            min_thrust=float(rospy.get_param("~min_thrust", 0.1)),
            max_thrust=float(rospy.get_param("~max_thrust", 0.75)),
            max_body_rate_rad_s=tuple(
                rospy.get_param("~max_body_rate_rad_s", [1.2, 1.2, 0.8])
            ),
            max_tilt_rad=math.radians(float(rospy.get_param("~max_tilt_deg", 35.0))),
        )
        self.control_mode = str(rospy.get_param("~control_mode", "attitude")).strip().lower()
        if self.control_mode == "attitude":
            self.attitude_target_type_mask = (
                AttitudeTarget.IGNORE_ROLL_RATE
                | AttitudeTarget.IGNORE_PITCH_RATE
                | AttitudeTarget.IGNORE_YAW_RATE
            )
        elif self.control_mode == "body_rate":
            self.attitude_target_type_mask = AttitudeTarget.IGNORE_ATTITUDE
        else:
            raise ValueError("control_mode must be one of: attitude, body_rate")
        self.rate_reversal_boost_enabled = bool(
            rospy.get_param("~rate_reversal_boost/enabled", False)
        )
        axis_feedback_modes = rospy.get_param(
            "~rate_reversal_boost/axis_feedback_modes", None
        )
        boost_parameters = RateReversalBoostParameters(
            feedback_mode=str(
                rospy.get_param(
                    "~rate_reversal_boost/feedback_mode", "legacy_reversal"
                )
            ),
            gains=tuple(rospy.get_param("~rate_reversal_boost/gains", [0.0, 0.0, 0.0])),
            damping_gains=tuple(
                rospy.get_param(
                    "~rate_reversal_boost/damping_gains", [0.0, 0.0, 0.0]
                )
            ),
            tracking_error_gains=tuple(
                rospy.get_param(
                    "~rate_reversal_boost/tracking_error_gains", [0.0, 0.0, 0.0]
                )
            ),
            axis_feedback_modes=(
                tuple(axis_feedback_modes)
                if axis_feedback_modes is not None
                else None
            ),
            filter_tau_s=float(
                rospy.get_param("~rate_reversal_boost/filter_tau_s", 0.04)
            ),
            max_correction_rad_s=tuple(
                rospy.get_param(
                    "~rate_reversal_boost/max_correction_rad_s",
                    [0.35, 0.35, 0.35],
                )
            ),
            reset_gap_s=float(
                rospy.get_param("~rate_reversal_boost/reset_gap_s", 0.20)
            ),
        )
        self.rate_reversal_booster = BodyRateReversalBooster(boost_parameters)
        self.max_imu_age_s = float(
            rospy.get_param("~rate_reversal_boost/max_imu_age_s", 0.10)
        )
        self.thrust_attitude_projection_enabled = bool(
            rospy.get_param("~thrust_attitude_projection/enabled", False)
        )
        self.thrust_attitude_projection_minimum_scale = float(
            rospy.get_param("~thrust_attitude_projection/minimum_scale", 0.5)
        )
        self.thrust_attitude_projection_max_imu_age_s = float(
            rospy.get_param("~thrust_attitude_projection/max_imu_age_s", 0.10)
        )
        if (
            not math.isfinite(self.thrust_attitude_projection_minimum_scale)
            or self.thrust_attitude_projection_minimum_scale <= 0.0
            or self.thrust_attitude_projection_minimum_scale > 1.0
        ):
            raise ValueError(
                "attitude thrust projection minimum scale must be in (0, 1]"
            )
        if (
            not math.isfinite(self.thrust_attitude_projection_max_imu_age_s)
            or self.thrust_attitude_projection_max_imu_age_s <= 0.0
        ):
            raise ValueError(
                "attitude thrust projection IMU age must be finite and positive"
            )
        if self.thrust_attitude_projection_enabled and self.control_mode != "body_rate":
            raise ValueError("attitude thrust projection requires body_rate control mode")
        if self.rate_reversal_boost_enabled and self.control_mode != "body_rate":
            raise ValueError("rate reversal boost requires body_rate control mode")
        self.tight_fov_guard_enabled = bool(
            rospy.get_param("~tight_fov_pixel_guard/enabled", False)
        )
        self.tight_fov_acceptance_rad = math.radians(
            float(rospy.get_param("~tight_fov_pixel_guard/acceptance_limit_deg", 1.0))
        )
        self.tight_fov_guard_parameters = TightFOVGuardParameters(
            activation_rad=math.radians(
                float(rospy.get_param("~tight_fov_pixel_guard/activation_deg", 0.5))
            ),
            rate_gain=float(rospy.get_param("~tight_fov_pixel_guard/rate_gain", 30.0)),
            maximum_guarded_rate_rad_s=float(
                rospy.get_param(
                    "~tight_fov_pixel_guard/max_guarded_rate_rad_s", 0.25
                )
            ),
            maximum_feature_age_s=float(
                rospy.get_param("~tight_fov_pixel_guard/max_feature_age_s", 0.12)
            ),
            derivative_enabled=bool(
                rospy.get_param("~tight_fov_pixel_guard/derivative_enabled", False)
            ),
            prediction_horizon_s=float(
                rospy.get_param("~tight_fov_pixel_guard/prediction_horizon_s", 0.05)
            ),
            derivative_filter_tau_s=float(
                rospy.get_param("~tight_fov_pixel_guard/derivative_filter_tau_s", 0.08)
            ),
            derivative_reset_gap_s=float(
                rospy.get_param("~tight_fov_pixel_guard/derivative_reset_gap_s", 0.12)
            ),
            maximum_raw_derivative_s_inv=float(
                rospy.get_param(
                    "~tight_fov_pixel_guard/max_raw_derivative_s_inv", 0.25
                )
            ),
            maximum_derivative_rate_rad_s=float(
                rospy.get_param(
                    "~tight_fov_pixel_guard/max_derivative_rate_rad_s", 0.20
                )
            ),
            los_rate_feedforward_enabled=bool(
                rospy.get_param(
                    "~tight_fov_pixel_guard/los_rate_feedforward_enabled", False
                )
            ),
            los_rate_feedforward_gain=float(
                rospy.get_param(
                    "~tight_fov_pixel_guard/los_rate_feedforward_gain", 1.0
                )
            ),
            maximum_los_rate_feedforward_rad_s=float(
                rospy.get_param(
                    "~tight_fov_pixel_guard/max_los_rate_feedforward_rad_s", 0.70
                )
            ),
            body_rate_prediction_enabled=bool(
                rospy.get_param(
                    "~tight_fov_pixel_guard/body_rate_prediction_enabled", False
                )
            ),
            body_rate_prediction_horizon_s=float(
                rospy.get_param(
                    "~tight_fov_pixel_guard/body_rate_prediction_horizon_s", 0.15
                )
            ),
            body_rate_deadband_rad_s=float(
                rospy.get_param(
                    "~tight_fov_pixel_guard/body_rate_deadband_rad_s", 0.03
                )
            ),
        )
        self.tight_fov_guard = None
        if self.tight_fov_guard_enabled:
            if self.control_mode != "body_rate":
                raise ValueError("tight FOV pixel guard requires body_rate control mode")
            if (
                (
                    self.tight_fov_guard_parameters.body_rate_prediction_enabled
                    or self.tight_fov_guard_parameters.los_rate_feedforward_enabled
                )
                and not self.rate_reversal_boost_enabled
            ):
                raise ValueError(
                    "tight FOV body-rate prediction/feedforward requires rate feedback IMU input"
                )
            self.tight_fov_guard = TightFOVGuard(
                self.tight_fov_acceptance_rad,
                self.tight_fov_guard_parameters,
            )
        self.visibility_guard_enabled = bool(
            rospy.get_param("~vertical_visibility_guard/enabled", False)
        )
        self.visibility_guard_max_feature_age_s = float(
            rospy.get_param("~vertical_visibility_guard/max_feature_age_s", 0.12)
        )
        if (
            not math.isfinite(self.visibility_guard_max_feature_age_s)
            or self.visibility_guard_max_feature_age_s <= 0.0
        ):
            raise ValueError(
                "vertical visibility guard feature age must be finite and positive"
            )
        self.visibility_guard_image_height = int(
            rospy.get_param("~image_height", 480)
        )
        self.visibility_guard_fy = float(rospy.get_param("~fy", 554.254691191187))
        self.visibility_guard_cy = float(rospy.get_param("~cy", 240.5))
        self.visibility_guard_activation_margin_px = float(
            rospy.get_param("~vertical_visibility_guard/activation_margin_px", 80.0)
        )
        self.visibility_guard_restoring_sign = float(
            rospy.get_param("~vertical_visibility_guard/restoring_sign", 1.0)
        )
        self.visibility_guard_minimum_thrust_scale = float(
            rospy.get_param(
                "~vertical_visibility_guard/minimum_thrust_scale", 1.0
            )
        )
        if (
            not math.isfinite(self.visibility_guard_minimum_thrust_scale)
            or self.visibility_guard_minimum_thrust_scale <= 0.0
            or self.visibility_guard_minimum_thrust_scale > 1.0
        ):
            raise ValueError(
                "vertical visibility guard minimum thrust scale must be in (0, 1]"
            )
        (
            visibility_acceptance_lower,
            visibility_acceptance_upper,
            visibility_activation_lower,
            visibility_activation_upper,
        ) = vertical_visibility_bounds(
            self.visibility_guard_image_height,
            self.visibility_guard_fy,
            self.visibility_guard_cy,
            self.visibility_guard_activation_margin_px,
        )
        self.visibility_guard_parameters = PixelAxisGuardParameters(
            acceptance_lower_normalized=visibility_acceptance_lower,
            acceptance_upper_normalized=visibility_acceptance_upper,
            activation_lower_normalized=visibility_activation_lower,
            activation_upper_normalized=visibility_activation_upper,
            rate_gain=float(
                rospy.get_param("~vertical_visibility_guard/rate_gain", 4.0)
            ),
            maximum_guarded_rate_rad_s=float(
                rospy.get_param(
                    "~vertical_visibility_guard/max_guarded_rate_rad_s", 1.0
                )
            ),
            derivative_enabled=bool(
                rospy.get_param("~vertical_visibility_guard/derivative_enabled", True)
            ),
            prediction_horizon_s=float(
                rospy.get_param(
                    "~vertical_visibility_guard/prediction_horizon_s", 0.08
                )
            ),
            derivative_filter_tau_s=float(
                rospy.get_param(
                    "~vertical_visibility_guard/derivative_filter_tau_s", 0.08
                )
            ),
            derivative_reset_gap_s=float(
                rospy.get_param(
                    "~vertical_visibility_guard/derivative_reset_gap_s", 0.12
                )
            ),
            maximum_raw_derivative_s_inv=float(
                rospy.get_param(
                    "~vertical_visibility_guard/max_raw_derivative_s_inv", 2.0
                )
            ),
            maximum_derivative_rate_rad_s=float(
                rospy.get_param(
                    "~vertical_visibility_guard/max_derivative_rate_rad_s", 0.25
                )
            ),
        )
        self.visibility_guard = None
        if self.visibility_guard_enabled:
            if self.control_mode != "body_rate":
                raise ValueError(
                    "vertical visibility guard requires body_rate control mode"
                )
            self.visibility_guard = PixelAxisGuard(
                self.visibility_guard_parameters,
                body_rate_axis_index=1,
                restoring_sign=self.visibility_guard_restoring_sign,
            )
        self.parameters.validate()
        if self.enabled:
            self.publisher = rospy.Publisher(
                "attitude_target", AttitudeTarget, queue_size=20
            )
            self.audit_publisher = rospy.Publisher(
                "adapter_debug", AdapterDebug, queue_size=20
            )
            if self.visibility_guard_enabled:
                self.visibility_audit_publisher = rospy.Publisher(
                    "visibility_guard_debug", VisibilityGuardDebug, queue_size=20
                )

        # Register callbacks only after publishers and all callback state exist.
        self.subscriber = rospy.Subscriber(
            "reference", ControlReference, self.reference_callback, queue_size=1
        )
        if self.require_control_enable:
            self.control_enable_subscriber = rospy.Subscriber(
                "control_enable", Bool, self.control_enable_callback, queue_size=1
            )
        if (
            self.rate_reversal_boost_enabled
            or self.thrust_attitude_projection_enabled
        ):
            self.imu_subscriber = rospy.Subscriber(
                "imu", Imu, self.imu_callback, queue_size=1
            )
        if self.tight_fov_guard_enabled or self.visibility_guard_enabled:
            self.feature_subscriber = rospy.Subscriber(
                "feature", TargetFeature, self.feature_callback, queue_size=1
            )

        if not self.enabled:
            rospy.logwarn(
                "PX4 output disabled: no setpoint publisher and no mode/arming services created"
            )
            return

        publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 80.0))
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / publish_rate_hz), self.timer_callback
        )
        axis_feedback_modes = boost_parameters.resolved_axis_feedback_modes()
        rospy.logwarn(
            "PX4 attitude output ENABLED; control_mode=%s; rate_reversal_boost=%s; "
            "control_enable_required=%s; control_enable_timeout=%.3f s; "
            "control_enable_wall_timeout=%.3f s; "
            "rate_feedback_mode=%s; rate_axis_feedback_modes=%s,%s,%s; "
            "reversal_gains=%s; damping_gains=%s; "
            "tracking_error_gains=%s; "
            "this node never changes mode and never arms",
            self.control_mode,
            self.rate_reversal_boost_enabled,
            self.require_control_enable,
            self.control_enable_timeout_s,
            self.control_enable_wall_timeout_s,
            boost_parameters.feedback_mode,
            axis_feedback_modes[0],
            axis_feedback_modes[1],
            axis_feedback_modes[2],
            list(boost_parameters.gains),
            list(boost_parameters.damping_gains),
            list(boost_parameters.tracking_error_gains),
        )
        if self.thrust_attitude_projection_enabled:
            rospy.logwarn(
                "Attitude thrust projection ENABLED: minimum_scale=%.3f "
                "max_imu_age=%.3f s; thrust is reduced to preserve the requested "
                "vertical component and avoid amplifying attitude error",
                self.thrust_attitude_projection_minimum_scale,
                self.thrust_attitude_projection_max_imu_age_s,
            )
        if self.tight_fov_guard_enabled:
            rospy.logwarn(
                "Tight FOV pixel guard ENABLED: activation=%.3f deg acceptance=%.3f deg "
                "gain=%.3f max_guarded_rate=%.3f rad/s max_feature_age=%.3f s; "
                "derivative=%s horizon=%.3f s tau=%.3f s reset_gap=%.3f s "
                "raw_derivative_limit=%.3f 1/s derivative_rate_limit=%.3f rad/s; "
                "los_rate_feedforward=%s feedforward_gain=%.3f "
                "feedforward_limit=%.3f rad/s; "
                "body_rate_prediction=%s body_rate_horizon=%.3f s "
                "body_rate_deadband=%.3f rad/s; "
                "FOV LOS uses the latest normalized pixel; "
                "this is an R4 platform safety layer, not the paper K_v term",
                math.degrees(self.tight_fov_guard_parameters.activation_rad),
                math.degrees(self.tight_fov_acceptance_rad),
                self.tight_fov_guard_parameters.rate_gain,
                self.tight_fov_guard_parameters.maximum_guarded_rate_rad_s,
                self.tight_fov_guard_parameters.maximum_feature_age_s,
                self.tight_fov_guard_parameters.derivative_enabled,
                self.tight_fov_guard_parameters.prediction_horizon_s,
                self.tight_fov_guard_parameters.derivative_filter_tau_s,
                self.tight_fov_guard_parameters.derivative_reset_gap_s,
                self.tight_fov_guard_parameters.maximum_raw_derivative_s_inv,
                self.tight_fov_guard_parameters.maximum_derivative_rate_rad_s,
                self.tight_fov_guard_parameters.los_rate_feedforward_enabled,
                self.tight_fov_guard_parameters.los_rate_feedforward_gain,
                self.tight_fov_guard_parameters.maximum_los_rate_feedforward_rad_s,
                self.tight_fov_guard_parameters.body_rate_prediction_enabled,
                self.tight_fov_guard_parameters.body_rate_prediction_horizon_s,
                self.tight_fov_guard_parameters.body_rate_deadband_rad_s,
            )
        if self.visibility_guard_enabled:
            rospy.logwarn(
                "Vertical pixel visibility guard ENABLED: image_height=%d fy=%.3f "
                "cy=%.3f activation_margin=%.1f px acceptance=[%.6f, %.6f] "
                "activation=[%.6f, %.6f] gain=%.3f max_guarded_rate=%.3f rad/s "
                "max_feature_age=%.3f s derivative=%s; upper raw-y boundary restores "
                "with body-pitch sign %.0f; minimum recovery thrust scale=%.3f; "
                "this is an R4 platform safety layer, not the "
                "paper alpha_lon barrier",
                self.visibility_guard_image_height,
                self.visibility_guard_fy,
                self.visibility_guard_cy,
                self.visibility_guard_activation_margin_px,
                self.visibility_guard_parameters.acceptance_lower_normalized,
                self.visibility_guard_parameters.acceptance_upper_normalized,
                self.visibility_guard_parameters.activation_lower_normalized,
                self.visibility_guard_parameters.activation_upper_normalized,
                self.visibility_guard_parameters.rate_gain,
                self.visibility_guard_parameters.maximum_guarded_rate_rad_s,
                self.visibility_guard_max_feature_age_s,
                self.visibility_guard_parameters.derivative_enabled,
                self.visibility_guard_restoring_sign,
                self.visibility_guard_minimum_thrust_scale,
            )

    def reference_callback(self, message):
        with self.lock:
            self.reference = message

    def control_enable_callback(self, message):
        with self.lock:
            now = rospy.Time.now()
            wall_now_s = time.monotonic()
            self.control_enable_lease.update(
                message.data, now.to_sec(), wall_now_s
            )
            if not message.data:
                self._reset_pixel_guards_locked()
                self.publish_blocked_audit(now, "control_disabled")

    def _reset_pixel_guards_locked(self):
        tight_fov_guard = getattr(self, "tight_fov_guard", None)
        if tight_fov_guard is not None:
            tight_fov_guard.reset()
        visibility_guard = getattr(self, "visibility_guard", None)
        if visibility_guard is not None:
            visibility_guard.reset()

    def _reset_tight_fov_guard_locked(self):
        """Compatibility alias for tests and older callers."""
        self._reset_pixel_guards_locked()

    def imu_callback(self, message):
        stamp = message.header.stamp
        if stamp == rospy.Time():
            stamp = rospy.Time.now()
        angular_velocity = message.angular_velocity
        orientation = message.orientation
        try:
            with self.lock:
                if self.rate_reversal_boost_enabled:
                    self.rate_reversal_booster.update(
                        [angular_velocity.x, angular_velocity.y, angular_velocity.z],
                        stamp.to_sec(),
                    )
                self.latest_measured_body_yaw_rate_rad_s = float(
                    angular_velocity.z
                )
                measured_attitude = np.array(
                    [orientation.x, orientation.y, orientation.z, orientation.w],
                    dtype=float,
                )
                quaternion_xyzw_to_rotation(measured_attitude)
                self.imu_attitude_xyzw = measured_attitude
                self.imu_stamp = stamp
        except ValueError as error:
            rospy.logwarn_throttle(1.0, "PX4 IMU sample rejected: %s", error)

    def feature_callback(self, message):
        with self.lock:
            self.feature = message

    def populate_rate_feedback_configuration(self, audit):
        parameters = self.rate_reversal_booster.parameters
        audit.rate_feedback_enabled = self.rate_reversal_boost_enabled
        audit.rate_axis_feedback_modes = list(
            parameters.resolved_axis_feedback_modes()
        )
        audit.rate_reversal_gains = vector3_message(parameters.gains)
        audit.rate_damping_gains = vector3_message(parameters.damping_gains)
        audit.rate_tracking_error_gains = vector3_message(
            parameters.tracking_error_gains
        )
        audit.rate_filter_tau_s = parameters.filter_tau_s
        audit.rate_max_correction_rad_s = vector3_message(
            parameters.max_correction_rad_s
        )
        audit.rate_reset_gap_s = parameters.reset_gap_s
        audit.rate_max_imu_age_s = self.max_imu_age_s

    def populate_tight_guard_configuration(self, audit):
        parameters = self.tight_fov_guard_parameters
        audit.guard_body_rate_prediction_enabled = (
            getattr(parameters, "body_rate_prediction_enabled", False)
        )
        audit.guard_body_rate_prediction_horizon_s = (
            getattr(parameters, "body_rate_prediction_horizon_s", 0.15)
        )
        audit.guard_body_rate_deadband_rad_s = getattr(
            parameters, "body_rate_deadband_rad_s", 0.03
        )

    @staticmethod
    def populate_tight_guard_body_rate_evidence(audit, guard_result):
        audit.guard_body_rate_prediction_initialized = bool(
            guard_result
            and getattr(guard_result, "body_rate_prediction_initialized", False)
        )
        audit.guard_body_rate_prediction_active = bool(
            guard_result
            and getattr(guard_result, "body_rate_prediction_active", False)
        )
        audit.guard_measured_body_yaw_rate_rad_s = (
            getattr(guard_result, "measured_body_yaw_rate_rad_s", math.nan)
            if guard_result
            else math.nan
        )
        audit.guard_rotational_s_tight_rate = (
            getattr(guard_result, "rotational_s_tight_rate_s_inv", math.nan)
            if guard_result
            else math.nan
        )
        audit.guard_body_rate_prediction_interval_s = (
            getattr(guard_result, "body_rate_prediction_interval_s", math.nan)
            if guard_result
            else math.nan
        )
        audit.guard_body_rate_predicted_s_tight = (
            getattr(guard_result, "body_rate_predicted_s_tight", math.nan)
            if guard_result
            else math.nan
        )
        audit.guard_body_rate_prediction_rate_rad_s = (
            getattr(guard_result, "body_rate_prediction_rate_rad_s", 0.0)
            if guard_result
            else 0.0
        )

    def publish_blocked_audit(
        self, stamp, reason, control_enabled=False, feature_age=math.inf
    ):
        if self.audit_publisher is None:
            return
        audit = AdapterDebug()
        audit.header.stamp = stamp
        audit.header.frame_id = "base_link"
        audit.control_enable_required = self.require_control_enable
        audit.control_enabled = bool(control_enabled)
        audit.output_permitted = False
        audit.guard_enabled = self.tight_fov_guard_enabled
        audit.guard_derivative_enabled = bool(
            self.tight_fov_guard_parameters.derivative_enabled
        )
        audit.guard_derivative_initialized = False
        audit.feature_valid = False
        audit.feature_age = float(feature_age)
        audit.measured_s_tight = math.nan
        audit.measured_margin_tight = math.nan
        audit.filtered_s_tight_rate = math.nan
        audit.outward_s_tight_rate = math.nan
        audit.guard_derivative_rate = 0.0
        audit.reference_body_rate = vector3_message(np.zeros(3))
        audit.damping_correction_body_rate = vector3_message(np.zeros(3))
        self.populate_rate_feedback_configuration(audit)
        self.populate_tight_guard_configuration(audit)
        self.populate_tight_guard_body_rate_evidence(audit, None)
        audit.guard_correction_body_rate = vector3_message(np.zeros(3))
        audit.pre_safety_body_rate = vector3_message(np.zeros(3))
        audit.sent_body_rate = vector3_message(np.zeros(3))
        audit.rate_saturated = False
        audit.thrust_saturated = False
        audit.reason = reason
        try:
            self.audit_publisher.publish(audit)
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise

    def publish_audit(
        self,
        stamp,
        reference_rate,
        feedback_correction,
        guard_result,
        feature_valid,
        feature_age,
        pre_safety_rate,
        reference_thrust,
        command,
        control_enabled=True,
        output_permitted=True,
    ):
        audit = AdapterDebug()
        audit.header.stamp = stamp
        audit.header.frame_id = "base_link"
        audit.control_enable_required = self.require_control_enable
        audit.control_enabled = bool(control_enabled)
        audit.output_permitted = bool(output_permitted)
        audit.guard_enabled = self.tight_fov_guard_enabled
        audit.guard_active = bool(guard_result and guard_result.active)
        audit.guard_limited = bool(guard_result and guard_result.limited)
        audit.guard_derivative_enabled = bool(
            self.tight_fov_guard_parameters.derivative_enabled
        )
        audit.guard_derivative_initialized = bool(
            guard_result and guard_result.derivative_initialized
        )
        audit.feature_valid = feature_valid
        audit.feature_age = feature_age
        audit.measured_s_tight = (
            guard_result.measured_s_tight if guard_result else math.nan
        )
        audit.measured_margin_tight = (
            guard_result.measured_margin_tight if guard_result else math.nan
        )
        audit.filtered_s_tight_rate = (
            guard_result.filtered_s_tight_rate_s_inv if guard_result else math.nan
        )
        audit.outward_s_tight_rate = (
            guard_result.outward_s_tight_rate_s_inv if guard_result else math.nan
        )
        audit.guard_derivative_rate = (
            guard_result.derivative_rate_rad_s if guard_result else 0.0
        )
        audit.reference_body_rate = vector3_message(reference_rate)
        audit.damping_correction_body_rate = vector3_message(feedback_correction)
        self.populate_rate_feedback_configuration(audit)
        self.populate_tight_guard_configuration(audit)
        self.populate_tight_guard_body_rate_evidence(audit, guard_result)
        audit.guard_correction_body_rate = vector3_message(
            guard_result.intervention_body_rate_rad_s
            if guard_result
            else np.zeros(3)
        )
        audit.pre_safety_body_rate = vector3_message(pre_safety_rate)
        audit.sent_body_rate = vector3_message(
            command.body_rate_rad_s if command.valid else np.zeros(3)
        )
        audit.reference_normalized_thrust = reference_thrust
        audit.sent_thrust = command.thrust if command.valid else 0.0
        raw_thrust = self.parameters.hover_thrust * reference_thrust
        audit.rate_saturated = bool(
            command.valid
            and not np.allclose(command.body_rate_rad_s, pre_safety_rate)
        )
        audit.thrust_saturated = bool(
            command.valid and not math.isclose(command.thrust, raw_thrust)
        )
        audit.reason = command.reason
        try:
            self.audit_publisher.publish(audit)
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise

    def publish_visibility_audit(
        self,
        stamp,
        feature,
        feature_valid,
        feature_age,
        input_pitch_rate,
        guard_result,
        guard_error=None,
        sent_pitch_rate=None,
    ):
        if self.visibility_audit_publisher is None:
            return
        audit = VisibilityGuardDebug()
        audit.header.stamp = stamp
        audit.header.frame_id = "camera_optical"
        audit.enabled = self.visibility_guard_enabled
        audit.active = bool(guard_result and guard_result.active)
        audit.intervened = bool(guard_result and guard_result.intervened)
        audit.limited = bool(guard_result and guard_result.limited)
        audit.within_acceptance = bool(
            guard_result and guard_result.within_acceptance
        )
        audit.feature_valid = bool(feature_valid)
        # The authorization callback can publish this audit slightly after the
        # guard calculation. Pair the feature stamp with the actual sent/audit
        # stamp so the recorded age describes the emitted command.
        audit.feature_age = (
            float((stamp - feature.header.stamp).to_sec())
            if feature is not None
            else float(feature_age)
        )
        audit.feature_stamp = (
            feature.header.stamp if feature is not None else rospy.Time()
        )
        audit.v = float(feature.v) if feature is not None else math.nan
        audit.y_normalized = (
            float(feature.y_normalized) if feature is not None else math.nan
        )
        parameters = self.visibility_guard_parameters
        audit.acceptance_lower = parameters.acceptance_lower_normalized
        audit.acceptance_upper = parameters.acceptance_upper_normalized
        audit.activation_lower = parameters.activation_lower_normalized
        audit.activation_upper = parameters.activation_upper_normalized
        audit.measured_margin = (
            guard_result.measured_margin_normalized if guard_result else math.nan
        )
        audit.filtered_coordinate_rate = (
            guard_result.filtered_coordinate_rate_s_inv
            if guard_result
            else math.nan
        )
        audit.outward_coordinate_rate = (
            guard_result.outward_coordinate_rate_s_inv
            if guard_result
            else math.nan
        )
        audit.derivative_rate = (
            guard_result.derivative_rate_rad_s if guard_result else 0.0
        )
        audit.derivative_enabled = parameters.derivative_enabled
        audit.derivative_initialized = bool(
            guard_result and guard_result.derivative_initialized
        )
        audit.feature_sample_status = (
            guard_result.feature_sample_status if guard_result else "unavailable"
        )
        audit.active_boundary = (
            guard_result.active_boundary if guard_result else "none"
        )
        audit.requested_pitch_rate = (
            guard_result.requested_body_rate_rad_s if guard_result else math.nan
        )
        audit.input_pitch_rate = float(input_pitch_rate)
        audit.output_pitch_rate = (
            float(sent_pitch_rate)
            if sent_pitch_rate is not None
            else (
                float(guard_result.body_rate_rad_s[1])
                if guard_result
                else float(input_pitch_rate)
            )
        )
        if guard_error is not None:
            audit.reason = "guard_error: {}".format(guard_error)
        elif not feature_valid:
            audit.reason = "feature_invalid"
        elif guard_result is None:
            audit.reason = "guard_unavailable"
        else:
            audit.reason = "accepted"
        try:
            self.visibility_audit_publisher.publish(audit)
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise

    def _pixel_guard_feature_age_limit(self):
        age_limits = []
        if self.tight_fov_guard_enabled:
            age_limits.append(
                float(self.tight_fov_guard_parameters.maximum_feature_age_s)
            )
        if self.visibility_guard_enabled:
            age_limits.append(float(self.visibility_guard_max_feature_age_s))
        return min(age_limits) if age_limits else None

    @staticmethod
    def _required_feature_status(feature, now, maximum_feature_age_s):
        if feature is None:
            return False, math.inf, "feature_missing"
        try:
            detected = bool(feature.detected)
            stamp = feature.header.stamp
            frame_id = str(feature.header.frame_id)
            x_normalized = float(feature.x_normalized)
            y_normalized = float(feature.y_normalized)
            feature_age = (now - stamp).to_sec()
        except (AttributeError, TypeError, ValueError):
            return False, math.inf, "feature_schema"
        if not detected:
            return False, feature_age, "feature_not_detected"
        if stamp.is_zero():
            return False, feature_age, "feature_stamp"
        if frame_id != "camera_optical":
            return False, feature_age, "feature_frame"
        if not all(
            math.isfinite(value)
            for value in (x_normalized, y_normalized, feature_age)
        ):
            return False, feature_age, "feature_nonfinite"
        if feature_age < 0.0:
            return False, feature_age, "feature_future"
        if feature_age > maximum_feature_age_s:
            return False, feature_age, "feature_stale"
        return True, feature_age, "accepted"

    def _publish_if_authorized(
        self,
        reference,
        expected_generation,
        publish_action,
        feature_required=False,
        feature_snapshot=None,
        blocked_reason=None,
    ):
        """Runs one output action only while the snapshotted lease is valid."""
        with self.lock:
            final_now = rospy.Time.now()
            (
                final_permitted,
                final_generation,
                final_reason,
                newly_timed_out,
            ) = self.control_enable_lease.authorize(
                final_now.to_sec(), time.monotonic()
            )
            if newly_timed_out:
                self._reset_pixel_guards_locked()
                self.publish_blocked_audit(final_now, final_reason)
            reference_age_s = (final_now - reference.header.stamp).to_sec()
            if (
                not final_permitted
                or final_generation != expected_generation
                or not math.isfinite(reference_age_s)
                or reference_age_s < 0.0
                or reference_age_s > self.command_timeout_s
            ):
                return False

            feature_age = math.inf
            if feature_required:
                feature_valid, feature_age, feature_reason = (
                    self._required_feature_status(
                        feature_snapshot,
                        final_now,
                        self._pixel_guard_feature_age_limit(),
                    )
                )
                if not feature_valid:
                    self._reset_pixel_guards_locked()
                    self.publish_blocked_audit(
                        final_now,
                        feature_reason,
                        control_enabled=final_permitted,
                        feature_age=feature_age,
                    )
                    return False
                # A newer subscriber sample does not invalidate the fresh,
                # leased snapshot that this cycle already used. Publishing
                # that exact snapshot keeps the setpoint/audit pair atomic
                # without dropping one timer cycle at every image boundary.
            if blocked_reason is not None:
                self._reset_pixel_guards_locked()
                self.publish_blocked_audit(
                    final_now,
                    blocked_reason,
                    control_enabled=final_permitted,
                    feature_age=feature_age,
                )
                return False
            publish_action(final_now, final_permitted)
            return True

    def _apply_pixel_guards_for_generation(
        self,
        now,
        expected_generation,
        commanded_body_rate,
        measured_body_yaw_rate_rad_s=None,
    ):
        """Atomically applies both guards to one leased feature snapshot."""
        feature_valid = False
        feature_age = math.inf
        feature = None
        tight_result = None
        visibility_result = None
        tight_error = None
        visibility_error = None
        tight_enabled = bool(
            getattr(
                self,
                "tight_fov_guard_enabled",
                getattr(self, "tight_fov_guard", None) is not None,
            )
        )
        visibility_enabled = bool(
            getattr(
                self,
                "visibility_guard_enabled",
                getattr(self, "visibility_guard", None) is not None,
            )
        )
        maximum_feature_age_s = self._pixel_guard_feature_age_limit()
        with self.lock:
            (
                guard_permitted,
                guard_generation,
                guard_reason,
                newly_timed_out,
            ) = self.control_enable_lease.authorize(
                now.to_sec(), time.monotonic()
            )
            if newly_timed_out:
                self._reset_pixel_guards_locked()
                self.publish_blocked_audit(now, guard_reason)
            if not guard_permitted or guard_generation != expected_generation:
                return (
                    False,
                    tight_result,
                    visibility_result,
                    feature_valid,
                    feature_age,
                    feature,
                    tight_error,
                    visibility_error,
                )

            feature = self.feature
            feature_valid, feature_age, _feature_reason = (
                self._required_feature_status(
                    feature, now, maximum_feature_age_s
                )
            )
            if feature_valid:
                guarded_rate = np.asarray(commanded_body_rate, dtype=float).reshape(3)
                if tight_enabled:
                    try:
                        tight_arguments = (
                            feature.x_normalized,
                            feature.y_normalized,
                            feature.header.stamp.to_sec(),
                            guarded_rate,
                        )
                        if getattr(
                            self.tight_fov_guard_parameters,
                            "body_rate_prediction_enabled",
                            False,
                        ):
                            tight_result = self.tight_fov_guard.apply(
                                *tight_arguments,
                                measured_body_yaw_rate_rad_s=(
                                    measured_body_yaw_rate_rad_s
                                ),
                                feature_age_s=feature_age,
                            )
                        else:
                            tight_result = self.tight_fov_guard.apply(*tight_arguments)
                        guarded_rate = tight_result.body_rate_rad_s
                    except ValueError as error:
                        tight_error = error
                        self.tight_fov_guard.reset()
                if visibility_enabled:
                    try:
                        visibility_result = self.visibility_guard.apply(
                            feature.y_normalized,
                            feature.header.stamp.to_sec(),
                            guarded_rate,
                        )
                    except ValueError as error:
                        visibility_error = error
                        self.visibility_guard.reset()
            else:
                self._reset_pixel_guards_locked()

        return (
            True,
            tight_result,
            visibility_result,
            feature_valid,
            feature_age,
            feature,
            tight_error,
            visibility_error,
        )

    def _apply_tight_fov_guard_for_generation(
        self,
        now,
        expected_generation,
        commanded_body_rate,
        measured_body_yaw_rate_rad_s=None,
    ):
        """Compatibility wrapper around the atomic dual-guard stage."""
        (
            authorized,
            tight_result,
            _visibility_result,
            feature_valid,
            feature_age,
            _feature,
            tight_error,
            _visibility_error,
        ) = self._apply_pixel_guards_for_generation(
            now,
            expected_generation,
            commanded_body_rate,
            measured_body_yaw_rate_rad_s=measured_body_yaw_rate_rad_s,
        )
        return authorized, tight_result, feature_valid, feature_age, tight_error

    def timer_callback(self, _event):
        if rospy.is_shutdown():
            return
        with self.lock:
            now = rospy.Time.now()
            wall_now_s = time.monotonic()
            reference = self.reference
            (
                output_permitted,
                control_enable_generation,
                control_enable_reason,
                newly_timed_out,
            ) = self.control_enable_lease.authorize(now.to_sec(), wall_now_s)
            if newly_timed_out:
                self._reset_pixel_guards_locked()
                self.publish_blocked_audit(now, control_enable_reason)
        if reference is None:
            return
        if not output_permitted:
            return
        age = (now - reference.header.stamp).to_sec()
        if (
            not reference.valid
            or reference.mode != ControlReference.MODE_TRACK
            or age < 0.0
            or age > self.command_timeout_s
        ):
            rospy.logwarn_throttle(1.0, "PX4 reference rejected: invalid or stale")
            return

        attitude = reference.desired_attitude
        body_rate = reference.desired_body_rate
        reference_body_rate = np.array(
            [body_rate.x, body_rate.y, body_rate.z], dtype=float
        )
        commanded_body_rate = reference_body_rate.copy()
        feedback_result = None
        measured_body_yaw_rate_rad_s = None
        if self.rate_reversal_boost_enabled:
            with self.lock:
                imu_stamp = self.imu_stamp
                if imu_stamp is not None and 0.0 <= (now - imu_stamp).to_sec() <= self.max_imu_age_s:
                    feedback_result = self.rate_reversal_booster.compensate(commanded_body_rate)
            if feedback_result is None:
                rospy.logwarn_throttle(
                    1.0, "PX4 rate reversal boost bypassed: no fresh IMU"
                )
                with self.lock:
                    imu_stamp = self.imu_stamp
                    held_yaw_rate = getattr(
                        self, "latest_measured_body_yaw_rate_rad_s", None
                    )
                    if (
                        imu_stamp is not None
                        and held_yaw_rate is not None
                        and math.isfinite(held_yaw_rate)
                        and 0.0
                        <= (now - imu_stamp).to_sec()
                        <= 2.0 * self.max_imu_age_s
                    ):
                        measured_body_yaw_rate_rad_s = float(held_yaw_rate)
            else:
                commanded_body_rate = feedback_result.body_rate_rad_s.copy()
                measured_body_yaw_rate_rad_s = float(
                    feedback_result.filtered_rate_rad_s[2]
                )
        feedback_correction = commanded_body_rate - reference_body_rate

        if (
            self.tight_fov_guard_enabled
            and (
                getattr(
                    self.tight_fov_guard_parameters,
                    "body_rate_prediction_enabled",
                    False,
                )
                or getattr(
                    self.tight_fov_guard_parameters,
                    "los_rate_feedforward_enabled",
                    False,
                )
            )
            and measured_body_yaw_rate_rad_s is None
        ):
            # Do not emit an unaudited startup command while the configured
            # narrow-FOV predictor still lacks its first fresh IMU sample.
            # The supervisor's attitude-stream watchdog bounds this wait.
            return

        feature_valid = False
        feature_age = math.inf
        guard_result = None
        visibility_result = None
        feature_snapshot = None
        visibility_input_pitch_rate = float(commanded_body_rate[1])
        visibility_error = None
        if self.tight_fov_guard_enabled or self.visibility_guard_enabled:
            (
                guard_authorized,
                guard_result,
                visibility_result,
                feature_valid,
                feature_age,
                feature_snapshot,
                guard_error,
                visibility_error,
            ) = self._apply_pixel_guards_for_generation(
                now,
                control_enable_generation,
                commanded_body_rate,
                measured_body_yaw_rate_rad_s=measured_body_yaw_rate_rad_s,
            )
            if not guard_authorized:
                return
            if guard_result is not None:
                commanded_body_rate = guard_result.body_rate_rad_s.copy()
            visibility_input_pitch_rate = float(commanded_body_rate[1])
            if visibility_result is not None:
                commanded_body_rate = visibility_result.body_rate_rad_s.copy()
            if guard_error is not None:
                rospy.logwarn_throttle(
                    1.0, "Tight FOV pixel guard bypassed: %s", guard_error
                )
            elif self.tight_fov_guard_enabled and not feature_valid:
                rospy.logwarn_throttle(
                    1.0, "Tight FOV pixel guard bypassed: no fresh detected feature"
                )
            if visibility_error is not None:
                rospy.logwarn_throttle(
                    1.0,
                    "Vertical visibility guard bypassed: %s",
                    visibility_error,
                )
            elif self.visibility_guard_enabled and not feature_valid:
                rospy.logwarn_throttle(
                    1.0,
                    "Vertical visibility guard bypassed: no fresh detected feature",
                )
            blocked_reason = None
            if not feature_valid:
                _, _, blocked_reason = self._required_feature_status(
                    feature_snapshot,
                    now,
                    self._pixel_guard_feature_age_limit(),
                )
            elif guard_error is not None:
                blocked_reason = "tight_fov_guard_error"
            elif visibility_error is not None:
                blocked_reason = "visibility_guard_error"
            if blocked_reason is not None:
                try:
                    self._publish_if_authorized(
                        reference,
                        control_enable_generation,
                        lambda *_args: None,
                        feature_required=True,
                        feature_snapshot=feature_snapshot,
                        blocked_reason=blocked_reason,
                    )
                except rospy.ROSException:
                    if not rospy.is_shutdown():
                        raise
                return
        pre_safety_body_rate = commanded_body_rate.copy()
        visibility_thrust_scale = 1.0
        if visibility_result is not None:
            visibility_thrust_scale = visibility_recovery_thrust_scale(
                visibility_result,
                self.visibility_guard_parameters,
                getattr(self, "visibility_guard_minimum_thrust_scale", 1.0),
            )
        platform_normalized_thrust = (
            reference.normalized_thrust * visibility_thrust_scale
        )
        if getattr(self, "thrust_attitude_projection_enabled", False):
            measured_attitude = None
            with self.lock:
                imu_stamp = self.imu_stamp
                if (
                    imu_stamp is not None
                    and 0.0
                    <= (now - imu_stamp).to_sec()
                    <= self.thrust_attitude_projection_max_imu_age_s
                    and self.imu_attitude_xyzw is not None
                ):
                    measured_attitude = self.imu_attitude_xyzw.copy()
            if measured_attitude is None:
                rospy.logwarn_throttle(
                    1.0,
                    "Attitude thrust projection bypassed: no fresh IMU attitude",
                )
            else:
                platform_normalized_thrust *= attitude_alignment_thrust_scale(
                    [attitude.x, attitude.y, attitude.z, attitude.w],
                    measured_attitude,
                    self.thrust_attitude_projection_minimum_scale,
                )
        command = sanitize_command(
            [attitude.x, attitude.y, attitude.z, attitude.w],
            commanded_body_rate,
            platform_normalized_thrust,
            self.parameters,
            enforce_tilt_limit=self.control_mode == "attitude",
        )
        if not command.valid:
            def publish_invalid_audit(final_now, final_permitted):
                self.publish_audit(
                    final_now,
                    reference_body_rate,
                    feedback_correction,
                    guard_result,
                    feature_valid,
                    feature_age,
                    pre_safety_body_rate,
                    platform_normalized_thrust,
                    command,
                    control_enabled=final_permitted,
                    output_permitted=True,
                )
            try:
                self._publish_if_authorized(
                    reference,
                    control_enable_generation,
                    publish_invalid_audit,
                    feature_required=bool(
                        self.tight_fov_guard_enabled
                        or self.visibility_guard_enabled
                    ),
                    feature_snapshot=feature_snapshot,
                )
            except rospy.ROSException:
                if not rospy.is_shutdown():
                    raise
            rospy.logwarn_throttle(1.0, "PX4 reference rejected: %s", command.reason)
            return

        target = AttitudeTarget()
        target.header.stamp = now
        target.header.frame_id = "base_link"
        target.type_mask = self.attitude_target_type_mask
        target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = (
            float(value) for value in command.attitude_xyzw
        )
        target.body_rate.x, target.body_rate.y, target.body_rate.z = (
            float(value) for value in command.body_rate_rad_s
        )
        target.thrust = command.thrust

        def publish_setpoint_and_audit(final_now, final_permitted):
            target.header.stamp = final_now
            self.publisher.publish(target)
            self.publish_audit(
                final_now,
                reference_body_rate,
                feedback_correction,
                guard_result,
                feature_valid,
                feature_age,
                pre_safety_body_rate,
                platform_normalized_thrust,
                command,
                control_enabled=final_permitted,
                output_permitted=True,
            )
            self.publish_visibility_audit(
                final_now,
                feature_snapshot,
                feature_valid,
                feature_age,
                visibility_input_pitch_rate,
                visibility_result,
                visibility_error,
                sent_pitch_rate=command.body_rate_rad_s[1],
            )

        try:
            self._publish_if_authorized(
                reference,
                control_enable_generation,
                publish_setpoint_and_audit,
                feature_required=bool(
                    self.tight_fov_guard_enabled
                    or self.visibility_guard_enabled
                ),
                feature_snapshot=feature_snapshot,
            )
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise


def main():
    rospy.init_node("pslos_px4_adapter")
    PX4AdapterNode()
    rospy.spin()


if __name__ == "__main__":
    main()
