#!/usr/bin/env python3
import math

import numpy as np
import rospy

from pslos_core.controller import (
    HARD_LONG_AXIS_BARRIER,
    PSLOSController,
    PSLOSParameters,
    SOFT_LONG_AXIS_RECOVERY,
)
from pslos_core.current_pixel_los import WorldLOSDerivative, current_pixel_los_world
from pslos_core.geometry import quaternion_xyzw_to_rotation, rotation_to_quaternion_xyzw
from pslos_msgs.msg import (
    ControlReference,
    CurrentPixelFOVDebug,
    PSLOSDebug,
    RelativeState,
    TargetFeature,
)


def array_from_vector(message):
    return np.array([message.x, message.y, message.z], dtype=float)


def vector3_message(value):
    from geometry_msgs.msg import Vector3

    message = Vector3()
    message.x, message.y, message.z = (float(item) for item in value)
    return message


def quaternion_message(value):
    from geometry_msgs.msg import Quaternion

    message = Quaternion()
    message.x, message.y, message.z, message.w = (float(item) for item in value)
    return message


class ReferenceNode:
    def __init__(self):
        paper = rospy.get_param("~paper", {})
        camera = rospy.get_param("~camera", {})
        platform = rospy.get_param("~platform_adapter", {})
        safety = rospy.get_param("~safety", {})
        parameters = PSLOSParameters(
            alpha_lon_rad=math.radians(float(paper.get("alpha_lon_deg", 55.0))),
            alpha_lat_rad=math.radians(float(paper.get("alpha_lat_deg", 1.0))),
            c1=float(paper.get("c1", 0.6)),
            c2=float(paper.get("c2", 1.0)),
            c_omega=float(paper.get("c_omega", 1.0)),
            mass=float(platform.get("mass_kg", 1.0)),
            tight_axis_gain=float(paper.get("tight_axis_gain", 1.0)),
        )
        rotation_body_camera = camera.get(
            "rotation_body_camera",
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        )
        maximum_closing_speed_mps = float(
            platform.get("maximum_closing_speed_mps", 0.0)
        )
        maximum_command_acceleration_mps2 = float(
            platform.get("maximum_command_acceleration_mps2", 0.0)
        )
        if (maximum_closing_speed_mps > 0.0) != (
            maximum_command_acceleration_mps2 > 0.0
        ):
            raise ValueError(
                "long-range maximum closing speed and acceleration must be enabled together"
            )
        bounded_velocity_mode = maximum_closing_speed_mps > 0.0
        attitude_tracking_gain = float(
            platform.get("attitude_tracking_gain", 0.0)
        )
        long_axis_fov_angular_rate_weight = float(
            platform.get("long_axis_fov_angular_rate_weight", 1.0)
        )
        long_axis_fov_recovery_start_deg = float(
            platform.get("long_axis_fov_recovery_start_deg", 0.0)
        )
        long_axis_fov_full_recovery_deg = float(
            platform.get("long_axis_fov_full_recovery_deg", 0.0)
        )
        self.controller = PSLOSController(
            parameters,
            rotation_body_camera,
            str(paper.get("formula_mode", "paper_literal")),
            str(paper.get("force_law_mode", "paper_equation_15")),
            maximum_closing_speed_mps=(
                maximum_closing_speed_mps if bounded_velocity_mode else None
            ),
            maximum_command_acceleration_mps2=(
                maximum_command_acceleration_mps2
                if bounded_velocity_mode
                else None
            ),
            minimum_closing_range_m=float(
                platform.get("minimum_closing_range_m", 0.0)
            ),
            attitude_tracking_gain=(
                attitude_tracking_gain if attitude_tracking_gain > 0.0 else None
            ),
            long_axis_fov_angular_rate_weight=(
                long_axis_fov_angular_rate_weight
            ),
            long_axis_fov_recovery_start_rad=math.radians(
                long_axis_fov_recovery_start_deg
            ),
            long_axis_fov_full_recovery_rad=math.radians(
                long_axis_fov_full_recovery_deg
            ),
            long_axis_barrier_policy=str(
                platform.get(
                    "long_axis_barrier_policy", HARD_LONG_AXIS_BARRIER
                )
            ),
            minimum_long_axis_denominator=float(
                platform.get("minimum_long_axis_denominator", 0.01)
            ),
            los_rate_guidance_gain=float(
                platform.get("current_los_rate_guidance_gain", 0.0)
            ),
            maximum_los_rate_guidance_acceleration_mps2=float(
                platform.get(
                    "maximum_current_los_rate_guidance_acceleration_mps2", 0.0
                )
            ),
            los_rate_guidance_terminal_start_range_m=float(
                platform.get("current_los_rate_guidance_terminal_start_range_m", 0.0)
            ),
            los_rate_guidance_terminal_minimum_scale=float(
                platform.get("current_los_rate_guidance_terminal_minimum_scale", 1.0)
            ),
            radial_velocity_feedback_only=bool(
                platform.get("radial_velocity_feedback_only", False)
            ),
        )
        self.gravity_force = np.asarray(
            platform.get("gravity_force_world", [0.0, 0.0, -9.80665]), dtype=float
        )
        self.aerodynamic_force = np.asarray(
            platform.get("aerodynamic_force_world", [0.0, 0.0, 0.0]), dtype=float
        )
        self.capture_radius = float(safety.get("capture_radius_m", 0.7))
        self.max_state_age = float(safety.get("max_state_age_s", 0.1))
        self.require_current_pixel_los = bool(
            safety.get("require_current_pixel_los", False)
        )
        self.use_current_los_for_position_direction = bool(
            platform.get("use_current_los_for_position_direction", False)
        )
        self.current_los_rate_guidance_enabled = bool(
            platform.get("current_los_rate_guidance_enabled", False)
        )
        if self.use_current_los_for_position_direction and not self.require_current_pixel_los:
            raise ValueError(
                "current-LOS position direction requires safety/require_current_pixel_los=true"
            )
        if self.current_los_rate_guidance_enabled and not self.require_current_pixel_los:
            raise ValueError(
                "current-LOS rate guidance requires safety/require_current_pixel_los=true"
            )
        self.current_los_rate_filter = (
            WorldLOSDerivative(
                filter_tau_s=float(
                    platform.get("current_los_rate_filter_tau_s", 0.12)
                ),
                reset_gap_s=float(
                    platform.get("current_los_rate_reset_gap_s", 0.16)
                ),
                maximum_rate_rad_s=float(
                    platform.get("maximum_current_los_rate_rad_s", 1.5)
                ),
            )
            if self.current_los_rate_guidance_enabled
            else None
        )
        self.max_feature_age = float(safety.get("max_feature_age_s", 0.12))
        if self.max_feature_age <= 0.0 or not math.isfinite(self.max_feature_age):
            raise ValueError("maximum feature age must be finite and positive")
        self.state = None
        self.feature = None
        self.reference_publisher = rospy.Publisher("reference", ControlReference, queue_size=10)
        self.debug_publisher = rospy.Publisher("debug", PSLOSDebug, queue_size=10)
        self.current_pixel_fov_debug_publisher = rospy.Publisher(
            "current_pixel_fov_debug", CurrentPixelFOVDebug, queue_size=10
        )
        self.subscriber = rospy.Subscriber(
            "relative_state", RelativeState, self.state_callback, queue_size=1
        )
        self.feature_subscriber = None
        if self.require_current_pixel_los:
            self.feature_subscriber = rospy.Subscriber(
                "feature", TargetFeature, self.feature_callback, queue_size=1
            )
        self.timer = rospy.Timer(rospy.Duration(0.01), self.timer_callback)
        rospy.loginfo(
            "PS-LOS reference only: formula=%s force_law=%s, alpha_lon=%.3f deg, alpha_lat=%.3f deg",
            self.controller.formula_mode,
            self.controller.force_law_mode,
            math.degrees(parameters.alpha_lon_rad),
            math.degrees(parameters.alpha_lat_rad),
        )
        if self.controller.bounded_velocity_mode:
            rospy.logwarn(
                "R4 long-range engineering outer loop ENABLED: "
                "maximum_closing_speed=%.3f m/s maximum_command_acceleration=%.3f m/s^2; "
                "this opt-in platform limit is not the paper outer-loop algebra",
                self.controller.maximum_closing_speed_mps,
                self.controller.maximum_command_acceleration_mps2,
            )
            if self.controller.minimum_closing_range_m > 0.0:
                rospy.logwarn(
                    "R4 estimated-range closing floor ENABLED: %.3f m; "
                    "FOV geometry still uses the unmodified estimator range",
                    self.controller.minimum_closing_range_m,
                )
            if self.controller.attitude_tracking_gain != parameters.c_omega:
                rospy.logwarn(
                    "R4 engineering attitude tracking gain ENABLED: %.3f "
                    "(paper-coupled gain %.3f); FOV feedback remains unchanged",
                    self.controller.attitude_tracking_gain,
                    parameters.c_omega,
                )
            if self.controller.long_axis_fov_angular_rate_weight != 1.0:
                rospy.logwarn(
                    "R4 engineering long-axis FOV angular-rate weight ENABLED: "
                    "%.3f; narrow-axis paper feedback remains full strength and "
                    "the downstream vertical image guard remains active",
                    self.controller.long_axis_fov_angular_rate_weight,
                )
            if self.controller.adaptive_long_axis_fov_weight:
                rospy.logwarn(
                    "R4 engineering long-axis speed-priority region ENABLED: "
                    "minimum weight %.3f through %.1f deg, smooth full recovery "
                    "by %.1f deg",
                    self.controller.long_axis_fov_angular_rate_weight,
                    math.degrees(
                        self.controller.long_axis_fov_recovery_start_rad
                    ),
                    math.degrees(
                        self.controller.long_axis_fov_full_recovery_rad
                    ),
                )
        if (
            self.controller.long_axis_barrier_policy
            == SOFT_LONG_AXIS_RECOVERY
        ):
            rospy.logwarn(
                "R4 long-axis SOFT RECOVERY enabled: minimum denominator=%.6f; "
                "the configured angular barrier is a recovery threshold, while "
                "physical image visibility remains the hard gate",
                self.controller.minimum_long_axis_denominator,
            )
        if self.require_current_pixel_los:
            rospy.logwarn(
                "FOV LOS uses the latest normalized pixel; "
                "PS-LOS FOV terms require frame=camera_optical "
                "max_feature_age=%.3f s; "
                "DC-EKF still supplies range, position and velocity",
                self.max_feature_age,
            )
        if self.use_current_los_for_position_direction:
            rospy.logwarn(
                "R4 current-pixel pursuit direction ENABLED: relative-position "
                "direction follows the fresh pixel LOS while DC-EKF supplies range; "
                "this is an engineering adaptation, not paper algebra"
            )
        if self.current_los_rate_guidance_enabled:
            rospy.logwarn(
                "R4 current-pixel world-LOS-rate guidance ENABLED: gain=%.3f "
                "acceleration_limit=%.3f m/s^2; pixel/attitude/EKF-range only, "
                "no truth or depth",
                self.controller.los_rate_guidance_gain,
                self.controller.maximum_los_rate_guidance_acceleration_mps2,
            )

    def state_callback(self, message):
        self.state = message

    def feature_callback(self, message):
        self.feature = message

    @staticmethod
    def publish(publisher, message):
        try:
            publisher.publish(message)
        except rospy.ROSException as error:
            if not rospy.is_shutdown() and "closed topic" not in str(error):
                raise
            return False
        return True

    def publish_invalid(self, stamp, mode, reason, state_age=math.inf):
        reference = ControlReference()
        reference.header.stamp = stamp
        reference.header.frame_id = "world_enu"
        reference.mode = mode
        reference.valid = False
        reference.reason = reason
        if not self.publish(self.reference_publisher, reference):
            return

        debug = PSLOSDebug()
        debug.header = reference.header
        debug.state_age = state_age
        debug.formula_mode = self.controller.formula_mode
        debug.force_law_mode = self.controller.force_law_mode
        debug.mode = reason
        debug.valid = False
        debug.reason = reason
        self.publish(self.debug_publisher, debug)

    def publish_current_pixel_audit(
        self,
        stamp,
        feature,
        feature_age=math.nan,
        s_long=math.nan,
        s_tight=math.nan,
        feature_valid=False,
        reason="feature_schema",
    ):
        audit = CurrentPixelFOVDebug()
        audit.header.stamp = stamp
        audit.header.frame_id = "world_enu"
        audit.feature_age = float(feature_age)
        audit.s_long = float(s_long)
        audit.s_tight = float(s_tight)
        audit.feature_valid = bool(feature_valid)
        audit.reason = str(reason)
        try:
            audit.feature_stamp = feature.header.stamp
            audit.feature_frame_id = str(feature.header.frame_id)
            audit.x_normalized = float(feature.x_normalized)
            audit.y_normalized = float(feature.y_normalized)
        except (AttributeError, TypeError, ValueError, OverflowError):
            audit.x_normalized = math.nan
            audit.y_normalized = math.nan
        self.publish(self.current_pixel_fov_debug_publisher, audit)

    @staticmethod
    def feature_age_for_audit(feature, stamp):
        try:
            return (stamp - feature.header.stamp).to_sec()
        except (AttributeError, TypeError, ValueError, OverflowError):
            return math.nan

    def timer_callback(self, _event):
        if rospy.is_shutdown():
            return
        stamp = rospy.Time.now()
        state = self.state
        if state is None:
            self.publish_invalid(stamp, ControlReference.MODE_INVALID, "no_state")
            return
        state_age = max(0.0, (stamp - state.header.stamp).to_sec())
        if not state.valid or state_age > self.max_state_age:
            self.publish_invalid(
                stamp, ControlReference.MODE_STALE_STATE, "stale_state", state_age
            )
            return
        if state.range <= self.capture_radius:
            self.publish_invalid(stamp, ControlReference.MODE_CAPTURE, "capture", state_age)
            return

        quaternion = state.interceptor_attitude
        rotation_world_body = quaternion_xyzw_to_rotation(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        )
        los_world = array_from_vector(state.los)
        feature_age = math.nan
        feature = None
        los_rate_world = None
        if self.require_current_pixel_los:
            # Freeze the exact subscriber sample used by this controller cycle.
            feature = self.feature
            try:
                los_world, feature_age = current_pixel_los_world(
                    feature,
                    stamp.to_sec(),
                    self.max_feature_age,
                    rotation_world_body,
                    self.controller.rotation_body_camera,
                )
            except ValueError as error:
                if getattr(self, "current_los_rate_filter", None) is not None:
                    self.current_los_rate_filter.reset()
                self.publish_current_pixel_audit(
                    stamp,
                    feature,
                    feature_age=self.feature_age_for_audit(feature, stamp),
                    reason=str(error),
                )
                self.publish_invalid(
                    stamp,
                    ControlReference.MODE_STALE_STATE,
                    "current_pixel_{}".format(error),
                    state_age,
                )
                return
            if getattr(self, "current_los_rate_guidance_enabled", False):
                los_rate_world, los_rate_initialized, _ = (
                    self.current_los_rate_filter.update(
                        los_world, feature.header.stamp.to_sec()
                    )
                )
                if not los_rate_initialized:
                    los_rate_world = None
        relative_position = array_from_vector(state.position)
        if self.use_current_los_for_position_direction:
            relative_position = -float(state.range) * los_world
        try:
            result = self.controller.compute(
                relative_position,
                array_from_vector(state.velocity),
                rotation_world_body,
                self.aerodynamic_force,
                self.gravity_force,
                los_world=los_world,
                los_rate_world=los_rate_world,
            )
        except ValueError as error:
            self.publish_invalid(
                stamp, ControlReference.MODE_OUTSIDE_SECTOR, str(error), state_age
            )
            return

        if result.fov.margin_long < 0.0:
            reason = "long_recovery"
        elif result.fov.margin_tight < 0.0:
            reason = "tight_recovery"
        else:
            reason = "track"
        reference = ControlReference()
        reference.header.stamp = stamp
        reference.header.frame_id = "world_enu"
        reference.desired_force_world = vector3_message(result.selected_force_world)
        reference.desired_attitude = quaternion_message(
            rotation_to_quaternion_xyzw(result.desired_attitude)
        )
        reference.desired_body_rate = vector3_message(result.desired_body_rate)
        reference.normalized_thrust = float(
            np.linalg.norm(result.selected_force_world)
            / (self.controller.parameters.mass * 9.80665)
        )
        reference.mode = ControlReference.MODE_TRACK
        reference.valid = True
        reference.reason = reason
        if not self.publish(self.reference_publisher, reference):
            return

        debug = PSLOSDebug()
        debug.header = reference.header
        debug.range = state.range
        debug.s_long = result.fov.s_long
        debug.s_tight = result.fov.s_tight
        debug.z1 = result.fov.z1
        debug.z2 = result.fov.z2
        debug.c_long = result.fov.c_long
        debug.c_tight = result.fov.c_tight
        debug.margin_long = result.fov.margin_long
        debug.margin_tight = result.fov.margin_tight
        debug.l1 = result.fov.l1
        debug.l2 = result.fov.l2
        debug.k_h = result.fov.k_h
        debug.k_v = result.fov.k_v
        debug.fov_tangent = vector3_message(result.fov.fov_tangent)
        debug.omega_los_body = vector3_message(result.fov.omega_los_body)
        debug.velocity_error = vector3_message(result.velocity_error)
        debug.state_age = state_age
        debug.vision_age = (
            feature_age if self.require_current_pixel_los else state.vision_age
        )
        debug.formula_mode = result.fov.formula_mode
        debug.force_law_mode = result.force_law_mode
        debug.mode = reason
        debug.valid = True
        debug.reason = reason
        if not self.publish(self.debug_publisher, debug):
            return
        if self.require_current_pixel_los:
            self.publish_current_pixel_audit(
                stamp,
                feature,
                feature_age=feature_age,
                s_long=result.fov.s_long,
                s_tight=result.fov.s_tight,
                feature_valid=True,
                reason="accepted",
            )


def main():
    rospy.init_node("pslos_reference")
    ReferenceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
