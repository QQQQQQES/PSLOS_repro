#!/usr/bin/env python3
from collections import deque
import math
import threading

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from pslos_core.dcekf import (
    B_GYR,
    DelayCompensatedEKF,
    EKFParameters,
    FEATURE,
    P_R,
    Q,
    V_R,
)
from pslos_core.geometry import (
    camera_ray,
    normalize,
    quaternion_xyzw_to_rotation,
    rotation_aligning,
    rotation_to_quaternion_xyzw,
)
from pslos_msgs.msg import RelativeState, TargetFeature


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


def stationary_gravity_world_alignment(messages, maximum_correction_rad):
    """Return a fixed world-frame correction from stationary IMU gravity.

    PX4 attitude and the visual estimator must agree on which direction is
    vertical.  A small fixed roll/pitch frame error becomes a metre-scale
    lateral position error when a bearing-only target is initialized at long
    range.  During the existing stationary bias window, specific force gives
    an independent measurement of world up.  The returned left-multiplying
    rotation maps the reported mean up direction onto ENU +Z.
    """
    maximum_correction_rad = float(maximum_correction_rad)
    if (
        not math.isfinite(maximum_correction_rad)
        or maximum_correction_rad <= 0.0
        or maximum_correction_rad > math.pi
    ):
        raise ValueError("gravity attitude alignment limit must be in (0, pi]")
    reported_up_world = []
    for message in messages:
        acceleration = np.array(
            [
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ],
            dtype=float,
        )
        acceleration_norm = float(np.linalg.norm(acceleration))
        if not math.isfinite(acceleration_norm) or acceleration_norm <= 1e-6:
            continue
        quaternion = message.orientation
        rotation_world_body = quaternion_xyzw_to_rotation(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        )
        reported_up_world.append(
            rotation_world_body.dot(acceleration / acceleration_norm)
        )
    if not reported_up_world:
        raise ValueError("gravity attitude alignment has no valid IMU samples")
    mean_up_world = normalize(np.mean(reported_up_world, axis=0), "mean IMU up")
    correction_angle_rad = math.acos(
        float(np.clip(mean_up_world[2], -1.0, 1.0))
    )
    if correction_angle_rad > maximum_correction_rad:
        raise ValueError(
            "gravity attitude alignment {:.3f} deg exceeds {:.3f} deg limit".format(
                math.degrees(correction_angle_rad),
                math.degrees(maximum_correction_rad),
            )
        )
    return (
        rotation_aligning(mean_up_world, np.array([0.0, 0.0, 1.0])),
        correction_angle_rad,
    )


class DCEKFNode:
    def __init__(self):
        camera = rospy.get_param("~camera", {})
        initialization = rospy.get_param("~initialization", {})
        noise = rospy.get_param("~noise", {})
        delay = rospy.get_param("~delay", {})
        validity = rospy.get_param("~validity", {})
        attitude_aiding = rospy.get_param("~attitude_aiding", {})
        velocity_aiding = rospy.get_param("~velocity_aiding", {})
        parameters = EKFParameters(
            gravity_world=tuple(rospy.get_param("~gravity_world", [0.0, 0.0, -9.80665])),
            gyro_noise_std=float(noise.get("gyro_std", 0.015)),
            accel_noise_std=float(noise.get("accel_std", 0.20)),
            target_accel_process_std=float(
                noise.get("target_accel_process_std", 0.0)
            ),
            gyro_bias_rw_std=float(noise.get("gyro_bias_rw_std", 0.0005)),
            accel_bias_rw_std=float(noise.get("accel_bias_rw_std", 0.01)),
            feature_process_noise_std=float(noise.get("feature_process_std", 0.002)),
            feature_measurement_noise_std=float(
                noise.get("feature_measurement_std", 0.006)
            ),
            initial_attitude_std=float(
                initialization.get("initial_attitude_std_rad", 0.03)
            ),
            initial_range_std_m=float(initialization.get("initial_range_std_m", 30.0)),
            initial_lateral_std_m=float(initialization.get("initial_lateral_std_m", 2.0)),
            initial_velocity_std_mps=float(
                initialization.get("initial_velocity_std_mps", 5.0)
            ),
            initial_gyro_bias_std=float(
                initialization.get("initial_gyro_bias_std_rps", 0.03)
            ),
            initial_accel_bias_std=float(
                initialization.get("initial_accel_bias_std_mps2", 0.3)
            ),
            max_history_s=float(delay.get("max_history_s", 0.5)),
            max_step_s=float(delay.get("max_imu_step_s", 0.03)),
            history_sync_tolerance_s=float(
                delay.get("history_sync_tolerance_s", 0.008)
            ),
            innovation_gate_chi2=float(delay.get("innovation_gate_chi2", 13.82)),
        )
        rotation_body_camera = camera.get(
            "rotation_body_camera",
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        )
        self.estimator = DelayCompensatedEKF(parameters, rotation_body_camera)
        self.estimator_lock = threading.RLock()
        self.initial_range_m = float(initialization.get("initial_range_m", 30.0))
        self.pending_runtime_range_m = None
        self.pending_runtime_measurement_count = 0
        self.feature_loss_reset_s = float(
            initialization.get("feature_loss_reset_s", 0.0)
        )
        if not math.isfinite(self.feature_loss_reset_s) or self.feature_loss_reset_s < 0.0:
            raise ValueError("initialization/feature_loss_reset_s must be nonnegative")
        self.bias_calibration_s = float(
            initialization.get("bias_calibration_s", 2.0)
        )
        self.minimum_bias_samples = int(
            initialization.get("minimum_bias_samples", 100)
        )
        self.require_vision_ready = bool(
            initialization.get("require_vision_ready", False)
        )
        self.vision_ready = not self.require_vision_ready
        self.vision_ready_cutoff_stamp_s = None
        self.frozen_initial_gyro_bias = None
        self.frozen_initial_accel_bias = None
        self.gravity_align_attitude_enabled = bool(
            initialization.get("gravity_align_attitude", False)
        )
        self.gravity_align_attitude_max_correction_rad = math.radians(
            float(initialization.get("gravity_align_attitude_max_correction_deg", 5.0))
        )
        if (
            not math.isfinite(self.gravity_align_attitude_max_correction_rad)
            or self.gravity_align_attitude_max_correction_rad <= 0.0
            or self.gravity_align_attitude_max_correction_rad > math.pi
        ):
            raise ValueError(
                "initialization/gravity_align_attitude_max_correction_deg "
                "must be in (0, 180]"
            )
        self.attitude_world_alignment_rotation = np.eye(3)
        self.minimum_measurements = int(validity.get("minimum_measurements", 3))
        self.max_position_std_m = float(validity.get("max_position_std_m", 50.0))
        self.max_vision_age_s = float(validity.get("max_vision_age_s", 0.5))
        self.attitude_aiding_enabled = bool(attitude_aiding.get("enabled", False))
        self.attitude_measurement_std_rad = float(
            attitude_aiding.get("measurement_std_rad", 0.02)
        )
        self.attitude_enforce_innovation_gate = bool(
            attitude_aiding.get("enforce_innovation_gate", True)
        )
        self.attitude_max_residual_rad = math.radians(
            float(attitude_aiding.get("max_residual_deg", 20.0))
        )
        if self.attitude_measurement_std_rad <= 0.0:
            raise ValueError("attitude_aiding/measurement_std_rad must be positive")
        if self.attitude_max_residual_rad <= 0.0:
            raise ValueError("attitude_aiding/max_residual_deg must be positive")
        self.velocity_aiding_enabled = bool(velocity_aiding.get("enabled", False))
        self.stationary_target = bool(velocity_aiding.get("stationary_target", False))
        self.velocity_radial_only = bool(velocity_aiding.get("radial_only", False))
        transverse_measurement_std = float(
            velocity_aiding.get("transverse_measurement_std_mps", 0.0)
        )
        self.velocity_transverse_measurement_std_mps = (
            math.inf
            if transverse_measurement_std == 0.0
            else transverse_measurement_std
        )
        self.velocity_transverse_activation_range_m = float(
            velocity_aiding.get("transverse_activation_range_m", 0.0)
        )
        self.velocity_measurement_std_mps = float(
            velocity_aiding.get("measurement_std_mps", 0.1)
        )
        self.velocity_enforce_innovation_gate = bool(
            velocity_aiding.get("enforce_innovation_gate", True)
        )
        self.velocity_max_residual_mps = float(
            velocity_aiding.get("max_residual_mps", 3.0)
        )
        self.velocity_max_future_lead_s = float(
            velocity_aiding.get("max_future_lead_s", 0.02)
        )
        if (
            self.velocity_aiding_enabled
            and not self.velocity_radial_only
            and not self.stationary_target
        ):
            raise ValueError(
                "full velocity aiding requires velocity_aiding/stationary_target=true"
            )
        if min(self.velocity_measurement_std_mps, self.velocity_max_residual_mps) <= 0.0:
            raise ValueError("velocity aiding limits must be positive")
        if self.velocity_transverse_measurement_std_mps <= 0.0 or math.isnan(
            self.velocity_transverse_measurement_std_mps
        ):
            raise ValueError(
                "velocity_aiding/transverse_measurement_std_mps must be nonnegative"
            )
        if (
            not math.isfinite(self.velocity_transverse_activation_range_m)
            or self.velocity_transverse_activation_range_m < 0.0
        ):
            raise ValueError(
                "velocity_aiding/transverse_activation_range_m must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.velocity_max_future_lead_s)
            or self.velocity_max_future_lead_s < 0.0
        ):
            raise ValueError("velocity aiding future lead must be finite and nonnegative")
        self.preinit_imu = deque()
        self.latest_body_rate = np.zeros(3)
        self.publisher = rospy.Publisher("relative_state", RelativeState, queue_size=10)
        self.imu_subscriber = rospy.Subscriber("imu", Imu, self.imu_callback, queue_size=200)
        self.velocity_subscriber = (
            rospy.Subscriber(
                "platform_velocity",
                TwistStamped,
                self.velocity_callback,
                queue_size=100,
            )
            if self.velocity_aiding_enabled
            else None
        )
        self.feature_subscriber = rospy.Subscriber(
            "feature", TargetFeature, self.feature_callback, queue_size=20
        )
        self.vision_ready_subscriber = (
            rospy.Subscriber(
                "vision_ready", Bool, self.vision_ready_callback, queue_size=10
            )
            if self.require_vision_ready
            else None
        )
        rospy.loginfo(
            "PS-LOS 18-state DC-EKF: range prior %.1f +/- %.1f m, history %.2f s, mode=%s",
            self.initial_range_m,
            parameters.initial_range_std_m,
            parameters.max_history_s,
            "platform_attitude_aided"
            if self.attitude_aiding_enabled
            else "paper_imu_only",
        )
        if self.require_vision_ready:
            rospy.logwarn(
                "DC-EKF vision-ready gate enabled; freezing the first %.2f s "
                "stationary IMU bias prior before accepting target features",
                self.bias_calibration_s,
            )
        if self.gravity_align_attitude_enabled:
            rospy.logwarn(
                "DC-EKF stationary gravity attitude alignment enabled: "
                "maximum correction %.2f deg",
                math.degrees(self.gravity_align_attitude_max_correction_rad),
            )
        if self.velocity_aiding_enabled:
            rospy.logwarn(
                "DC-EKF %s velocity aiding enabled: sigma=%.3f m/s "
                "transverse_sigma=%s activation_range=%s "
                "maximum future lead=%.3f s",
                "radial platform-velocity"
                if self.velocity_radial_only
                else "stationary-target full",
                self.velocity_measurement_std_mps,
                (
                    "disabled"
                    if math.isinf(self.velocity_transverse_measurement_std_mps)
                    else "{:.3f} m/s".format(
                        self.velocity_transverse_measurement_std_mps
                    )
                ),
                (
                    "all ranges"
                    if self.velocity_transverse_activation_range_m == 0.0
                    else "{:.1f} m".format(
                        self.velocity_transverse_activation_range_m
                    )
                ),
                self.velocity_max_future_lead_s,
            )
        rospy.loginfo(
            "DC-EKF covariance: initial_attitude_std_rad=%.9g "
            "initial_gyro_bias_std_rps=%.9g initial_accel_bias_std_mps2=%.9g "
            "gyro_std=%.9g accel_std=%.9g gyro_bias_rw_std=%.9g "
            "accel_bias_rw_std=%.9g mode=%s",
            parameters.initial_attitude_std,
            parameters.initial_gyro_bias_std,
            parameters.initial_accel_bias_std,
            parameters.gyro_noise_std,
            parameters.accel_noise_std,
            parameters.gyro_bias_rw_std,
            parameters.accel_bias_rw_std,
            "platform_attitude_aided"
            if self.attitude_aiding_enabled
            else "paper_imu_only",
        )

    @staticmethod
    def imu_values(message):
        angular_velocity = np.array(
            [message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z]
        )
        acceleration = np.array(
            [
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ]
        )
        return angular_velocity, acceleration

    @staticmethod
    def valid_orientation(message):
        quaternion = message.orientation
        norm = math.sqrt(
            quaternion.x ** 2 + quaternion.y ** 2 + quaternion.z ** 2 + quaternion.w ** 2
        )
        return message.orientation_covariance[0] >= 0.0 and norm > 0.5

    def trim_preinit_buffer(self, newest_stamp):
        retention_s = max(
            self.estimator.parameters.max_history_s,
            self.bias_calibration_s + self.estimator.parameters.max_history_s,
        )
        cutoff = newest_stamp - retention_s
        while self.preinit_imu and self.preinit_imu[0].header.stamp.to_sec() < cutoff:
            self.preinit_imu.popleft()

    def aligned_attitude_xyzw(self, quaternion):
        rotation_world_body = quaternion_xyzw_to_rotation(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        )
        alignment = getattr(self, "attitude_world_alignment_rotation", np.eye(3))
        return rotation_to_quaternion_xyzw(alignment.dot(rotation_world_body))

    def bias_prior_from_samples(self, calibration_samples):
        gyro_samples = []
        accel_bias_samples = []
        gravity_world = np.asarray(self.estimator.parameters.gravity_world, dtype=float)
        for message in calibration_samples:
            angular_velocity, measured_specific_force = self.imu_values(message)
            rotation_world_body = quaternion_xyzw_to_rotation(
                self.aligned_attitude_xyzw(message.orientation)
            )
            expected_specific_force = rotation_world_body.T.dot(-gravity_world)
            gyro_samples.append(angular_velocity)
            accel_bias_samples.append(measured_specific_force - expected_specific_force)
        return np.mean(gyro_samples, axis=0), np.mean(accel_bias_samples, axis=0)

    def freeze_initial_bias_prior_if_ready(self):
        if (
            not self.require_vision_ready
            or self.frozen_initial_gyro_bias is not None
        ):
            return False
        candidates = [
            message for message in self.preinit_imu if self.valid_orientation(message)
        ]
        span_s = (
            candidates[-1].header.stamp.to_sec()
            - candidates[0].header.stamp.to_sec()
            if candidates
            else 0.0
        )
        if len(candidates) < self.minimum_bias_samples or span_s < self.bias_calibration_s:
            return False
        if getattr(self, "gravity_align_attitude_enabled", False):
            try:
                (
                    self.attitude_world_alignment_rotation,
                    correction_angle_rad,
                ) = stationary_gravity_world_alignment(
                    candidates,
                    self.gravity_align_attitude_max_correction_rad,
                )
            except ValueError as error:
                rospy.logerr_throttle(
                    1.0, "DC-EKF gravity attitude alignment rejected: %s", error
                )
                return False
            rospy.logwarn(
                "DC-EKF applied %.3f deg stationary gravity attitude alignment",
                math.degrees(correction_angle_rad),
            )
        gyro_bias, accel_bias = self.bias_prior_from_samples(candidates)
        if not np.all(np.isfinite(gyro_bias)) or not np.all(np.isfinite(accel_bias)):
            rospy.logerr("DC-EKF refused a non-finite frozen IMU bias prior")
            return False
        self.frozen_initial_gyro_bias = gyro_bias.copy()
        self.frozen_initial_accel_bias = accel_bias.copy()
        rospy.logwarn(
            "DC-EKF froze preflight bias prior from %d samples over %.3f s: "
            "gyro=%s accel=%s",
            len(candidates),
            span_s,
            np.array2string(gyro_bias, precision=4),
            np.array2string(accel_bias, precision=4),
        )
        return True

    def vision_ready_callback(self, message):
        if not self.require_vision_ready:
            return
        with self.estimator_lock:
            if not bool(message.data):
                was_ready = self.vision_ready
                self.vision_ready = False
                self.vision_ready_cutoff_stamp_s = None
                self.clear_runtime_reinitialization_prior()
                if self.estimator.initialized:
                    self.estimator.reset()
                    rospy.logwarn("DC-EKF vision-ready gate revoked; estimator reset")
                elif was_ready:
                    rospy.logwarn("DC-EKF vision-ready gate revoked")
                return
            if self.vision_ready:
                return
            if self.frozen_initial_gyro_bias is None:
                rospy.logerr_throttle(
                    1.0,
                    "DC-EKF vision-ready request rejected: preflight bias prior is incomplete",
                )
                return
            cutoff_stamp_s = rospy.Time.now().to_sec()
            if not math.isfinite(cutoff_stamp_s) or cutoff_stamp_s <= 0.0:
                rospy.logerr_throttle(
                    1.0,
                    "DC-EKF vision-ready request rejected: ROS time is unavailable",
                )
                return
            self.vision_ready_cutoff_stamp_s = cutoff_stamp_s
            self.vision_ready = True
            rospy.logwarn(
                "DC-EKF vision-ready gate opened at %.6f; accepting only newer features",
                cutoff_stamp_s,
            )

    def imu_callback(self, message):
        with self.estimator_lock:
            stamp = message.header.stamp.to_sec()
            if stamp <= 0.0:
                return
            angular_velocity, acceleration = self.imu_values(message)
            self.latest_body_rate = angular_velocity
            if not self.preinit_imu or stamp > self.preinit_imu[-1].header.stamp.to_sec():
                self.preinit_imu.append(message)
                self.trim_preinit_buffer(stamp)
                self.freeze_initial_bias_prior_if_ready()
            if not self.estimator.initialized:
                return
            if stamp <= self.estimator.stamp:
                return
            try:
                self.estimator.propagate(stamp, angular_velocity, acceleration)
            except ValueError as error:
                rospy.logwarn_throttle(1.0, "DC-EKF IMU propagation rejected: %s", error)
                if str(error) in (
                    "estimated target depth is non-positive",
                    "IMU gap exceeds retained history",
                ):
                    preserved_range_m = self.cache_runtime_reinitialization_prior()
                    self.estimator.reset()
                    if preserved_range_m is None:
                        rospy.logwarn(
                            "DC-EKF reset; waiting for a fresh delayed feature"
                        )
                    else:
                        rospy.logwarn(
                            "DC-EKF runtime reset preserved %.3f m range and %d "
                            "accepted measurements for fresh-LOS reinitialization",
                            preserved_range_m,
                            self.pending_runtime_measurement_count,
                        )
                return
            self.update_attitude(message)
            self.publish_state(False)

    def update_attitude(self, message):
        if not self.attitude_aiding_enabled or not self.valid_orientation(message):
            return
        quaternion = self.aligned_attitude_xyzw(message.orientation)
        result = self.estimator.update_attitude(
            message.header.stamp.to_sec(),
            quaternion,
            self.attitude_measurement_std_rad,
            enforce_innovation_gate=self.attitude_enforce_innovation_gate,
            max_residual_rad=self.attitude_max_residual_rad,
        )
        if not result.accepted:
            rospy.logwarn_throttle(
                1.0,
                "DC-EKF attitude observation rejected: %s, mahalanobis=%.3f",
                result.reason,
                result.innovation_mahalanobis,
            )

    def velocity_callback(self, message):
        if not self.velocity_aiding_enabled:
            return
        with self.estimator_lock:
            stamp = message.header.stamp.to_sec()
            if not self.estimator.initialized or stamp <= 0.0:
                return
            effective_stamp = stamp
            latest_estimator_stamp = self.estimator.stamp
            if stamp > latest_estimator_stamp:
                future_lead_s = stamp - latest_estimator_stamp
                if future_lead_s > getattr(
                    self, "velocity_max_future_lead_s", 0.02
                ):
                    rospy.logwarn_throttle(
                        1.0,
                        "DC-EKF velocity observation leads IMU history by %.3f s",
                        future_lead_s,
                    )
                    return
                # The velocity and IMU samples share a PX4 timestamp, but ROS
                # can deliver velocity first. Fuse at the latest retained IMU
                # state when the lead is bounded to one sensor frame.
                effective_stamp = latest_estimator_stamp
            velocity = message.twist.linear
            transverse_measurement_std = (
                self.velocity_transverse_measurement_std_mps
            )
            if (
                math.isfinite(transverse_measurement_std)
                and self.velocity_transverse_activation_range_m > 0.0
                and np.linalg.norm(self.estimator.state[P_R])
                > self.velocity_transverse_activation_range_m
            ):
                transverse_measurement_std = math.inf
            try:
                result = self.estimator.update_velocity(
                    effective_stamp,
                    [velocity.x, velocity.y, velocity.z],
                    self.velocity_measurement_std_mps,
                    enforce_innovation_gate=self.velocity_enforce_innovation_gate,
                    max_residual_mps=self.velocity_max_residual_mps,
                    radial_only=self.velocity_radial_only,
                    transverse_measurement_std=(
                        transverse_measurement_std
                    ),
                )
            except ValueError as error:
                rospy.logwarn_throttle(1.0, "DC-EKF velocity observation rejected: %s", error)
                return
            if not result.accepted:
                rospy.logwarn_throttle(
                    1.0,
                    "DC-EKF velocity observation rejected: %s, mahalanobis=%.3f",
                    result.reason,
                    result.innovation_mahalanobis,
                )
                return
            self.publish_state(False)

    def clear_runtime_reinitialization_prior(self):
        self.pending_runtime_range_m = None
        self.pending_runtime_measurement_count = 0

    def cache_runtime_reinitialization_prior(self):
        """Retain active-track range across a numerical geometry reset.

        A monocular runtime reset must not silently reuse the original launch
        range: near interception that can turn an 8 m target back into a 100 m
        target.  The fresh pixel supplies direction on reinitialization while
        this scalar prior preserves range continuity.  A cold start and a
        feature-loss reacquisition still use the configured launch prior.
        """
        self.clear_runtime_reinitialization_prior()
        if not self.estimator.initialized:
            return None
        try:
            range_m = float(np.linalg.norm(self.estimator.state[P_R]))
        except (RuntimeError, ValueError, TypeError):
            return None
        minimum_range_m = float(
            getattr(self.estimator.parameters, "min_depth_m", 1e-3)
        )
        if not math.isfinite(range_m) or range_m <= minimum_range_m:
            return None
        # Interception range should not grow beyond the cold-start prior.  This
        # also prevents a corrupt state norm from becoming the recovery prior.
        range_m = min(range_m, self.initial_range_m)
        self.pending_runtime_range_m = range_m
        self.pending_runtime_measurement_count = max(
            1, int(getattr(self.estimator, "measurement_count", 1))
        )
        return range_m

    def initialize_from_feature(self, feature):
        feature_stamp = feature.header.stamp.to_sec()
        candidates = [message for message in self.preinit_imu if self.valid_orientation(message)]
        if not candidates:
            rospy.logwarn_throttle(1.0, "DC-EKF waiting for IMU orientation")
            return False
        nearest = min(candidates, key=lambda message: abs(message.header.stamp.to_sec() - feature_stamp))
        if abs(nearest.header.stamp.to_sec() - feature_stamp) > 0.03:
            rospy.logwarn_throttle(1.0, "No IMU attitude close to first feature timestamp")
            return False
        quaternion = self.aligned_attitude_xyzw(nearest.orientation)
        if self.require_vision_ready:
            if self.frozen_initial_gyro_bias is None:
                rospy.logwarn_throttle(1.0, "DC-EKF waiting for frozen bias prior")
                return False
            initial_gyro_bias = self.frozen_initial_gyro_bias.copy()
            initial_accel_bias = self.frozen_initial_accel_bias.copy()
        else:
            calibration_samples = [
                message
                for message in candidates
                if message.header.stamp.to_sec() <= feature_stamp
            ]
            calibration_span = (
                calibration_samples[-1].header.stamp.to_sec()
                - calibration_samples[0].header.stamp.to_sec()
                if calibration_samples
                else 0.0
            )
            if (
                len(calibration_samples) < self.minimum_bias_samples
                or calibration_span < self.bias_calibration_s
            ):
                rospy.logwarn_throttle(
                    1.0,
                    "DC-EKF collecting stationary bias prior: %d/%d samples, %.2f/%.2f s",
                    len(calibration_samples),
                    self.minimum_bias_samples,
                    calibration_span,
                    self.bias_calibration_s,
                )
                return False
            initial_gyro_bias, initial_accel_bias = self.bias_prior_from_samples(
                calibration_samples
            )
        pending_runtime_range_m = getattr(
            self, "pending_runtime_range_m", None
        )
        initialization_range_m = (
            pending_runtime_range_m
            if pending_runtime_range_m is not None
            else self.initial_range_m
        )
        preserved_measurement_count = getattr(
            self, "pending_runtime_measurement_count", 0
        )
        self.estimator.initialize(
            feature_stamp,
            quaternion,
            [feature.x_normalized, feature.y_normalized],
            initialization_range_m,
            initial_gyro_bias,
            initial_accel_bias,
        )
        if pending_runtime_range_m is not None:
            self.estimator.measurement_count = max(
                self.estimator.measurement_count,
                preserved_measurement_count,
                self.minimum_measurements,
            )
            self.clear_runtime_reinitialization_prior()
        buffered = [
            message
            for message in self.preinit_imu
            if message.header.stamp.to_sec() > feature_stamp
        ]
        for message in buffered:
            angular_velocity, acceleration = self.imu_values(message)
            try:
                self.estimator.propagate(
                    message.header.stamp.to_sec(), angular_velocity, acceleration
                )
                self.update_attitude(message)
            except ValueError as error:
                rospy.logwarn("DC-EKF initialization replay stopped: %s", error)
                break
        rospy.loginfo(
            "DC-EKF initialized from bearing only at %.3f m; bias prior gyro=%s "
            "accel=%s",
            initialization_range_m,
            np.array2string(initial_gyro_bias, precision=4),
            np.array2string(initial_accel_bias, precision=4),
        )
        return True

    def feature_callback(self, feature):
        with self.estimator_lock:
            if not feature.detected or feature.header.stamp.is_zero():
                return
            feature_stamp = feature.header.stamp.to_sec()
            if self.require_vision_ready and (
                not self.vision_ready
                or self.vision_ready_cutoff_stamp_s is None
                or feature_stamp <= self.vision_ready_cutoff_stamp_s
            ):
                return
            if (
                self.estimator.initialized
                and self.feature_loss_reset_s > 0.0
                and self.estimator.last_feature_stamp is not None
                and feature_stamp - self.estimator.last_feature_stamp
                > self.feature_loss_reset_s
            ):
                gap_s = feature_stamp - self.estimator.last_feature_stamp
                self.estimator.reset()
                self.clear_runtime_reinitialization_prior()
                rospy.logwarn(
                    "DC-EKF reset after %.3f s feature loss; waiting for a fresh target prior",
                    gap_s,
                )
            if not self.estimator.initialized:
                if self.initialize_from_feature(feature):
                    self.publish_state(True)
                return
            confidence = max(min(float(feature.confidence), 1.0), 0.05)
            measurement_std = (
                self.estimator.parameters.feature_measurement_noise_std
                / math.sqrt(confidence)
            )
            result = self.estimator.update_feature(
                feature.header.stamp.to_sec(),
                [feature.x_normalized, feature.y_normalized],
                measurement_std,
            )
            if not result.accepted:
                rospy.logwarn_throttle(
                    1.0,
                    "DC-EKF feature rejected: %s, mahalanobis=%.3f",
                    result.reason,
                    result.innovation_mahalanobis,
                )
                return
            self.publish_state(True)

    def publish_state(self, measurement_updated):
        if rospy.is_shutdown():
            return
        state_vector = self.estimator.state
        covariance = self.estimator.covariance
        position = state_vector[P_R]
        range_m = float(np.linalg.norm(position))
        if range_m <= 1e-9:
            return
        vision_age = self.estimator.vision_age()
        valid = self.estimator.estimate_valid(
            self.minimum_measurements,
            self.max_position_std_m,
            self.max_vision_age_s,
        )

        message = RelativeState()
        message.header.stamp = rospy.Time.from_sec(self.estimator.stamp)
        message.header.frame_id = "world_enu"
        message.position = vector3_message(position)
        message.velocity = vector3_message(state_vector[V_R])
        rotation_world_body = quaternion_xyzw_to_rotation(state_vector[Q])
        feature_ray = camera_ray(*state_vector[FEATURE])
        measured_los_world = rotation_world_body.dot(
            self.estimator.rotation_body_camera.dot(feature_ray)
        )
        message.los = vector3_message(measured_los_world)
        message.interceptor_attitude = quaternion_message(state_vector[Q])
        message.interceptor_body_rate = vector3_message(
            self.latest_body_rate - state_vector[B_GYR]
        )
        message.range = range_m
        message.vision_age = vision_age
        indices = list(range(4, 10))
        message.covariance = covariance[np.ix_(indices, indices)].reshape(-1).tolist()
        message.measurement_count = self.estimator.measurement_count
        message.source = RelativeState.SOURCE_DC_EKF
        message.measurement_updated = measurement_updated
        message.valid = valid
        try:
            self.publisher.publish(message)
        except rospy.ROSException:
            # rospy can close publishers between the shutdown check and publish.
            return


def main():
    rospy.init_node("pslos_dcekf")
    DCEKFNode()
    rospy.spin()


if __name__ == "__main__":
    main()
