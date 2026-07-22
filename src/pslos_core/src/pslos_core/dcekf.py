from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .geometry import camera_ray, normalize, quaternion_xyzw_to_rotation


STATE_SIZE = 18
Q = slice(0, 4)
P_R = slice(4, 7)
V_R = slice(7, 10)
FEATURE = slice(10, 12)
B_GYR = slice(12, 15)
B_ACC = slice(15, 18)


def quaternion_multiply_xyzw(left, right):
    left = np.asarray(left, dtype=float).reshape(4)
    right = np.asarray(right, dtype=float).reshape(4)
    left_vector, left_w = left[:3], left[3]
    right_vector, right_w = right[:3], right[3]
    vector = (
        left_w * right_vector
        + right_w * left_vector
        + np.cross(left_vector, right_vector)
    )
    scalar = left_w * right_w - float(np.dot(left_vector, right_vector))
    return np.concatenate((vector, [scalar]))


def quaternion_conjugate_xyzw(quaternion):
    quaternion = np.asarray(quaternion, dtype=float).reshape(4)
    return np.concatenate((-quaternion[:3], [quaternion[3]]))


def attitude_residual(measured_xyzw, predicted_xyzw):
    """Return the shortest measured-minus-predicted rotation vector."""
    measured = np.asarray(measured_xyzw, dtype=float).reshape(4)
    predicted = np.asarray(predicted_xyzw, dtype=float).reshape(4)
    measured /= np.linalg.norm(measured)
    predicted /= np.linalg.norm(predicted)
    if np.dot(measured, predicted) < 0.0:
        measured = -measured
    error = quaternion_multiply_xyzw(
        measured, quaternion_conjugate_xyzw(predicted)
    )
    error /= np.linalg.norm(error)
    if error[3] < 0.0:
        error = -error
    vector_norm = float(np.linalg.norm(error[:3]))
    if vector_norm < 1e-10:
        return 2.0 * error[:3]
    angle = 2.0 * math.atan2(vector_norm, float(error[3]))
    return error[:3] * (angle / vector_norm)


def delta_quaternion_xyzw(angular_velocity, dt):
    angular_velocity = np.asarray(angular_velocity, dtype=float).reshape(3)
    angle = float(np.linalg.norm(angular_velocity)) * dt
    if angle < 1e-12:
        return np.array(
            [
                0.5 * angular_velocity[0] * dt,
                0.5 * angular_velocity[1] * dt,
                0.5 * angular_velocity[2] * dt,
                1.0,
            ]
        )
    axis = angular_velocity / np.linalg.norm(angular_velocity)
    return np.concatenate((axis * math.sin(0.5 * angle), [math.cos(0.5 * angle)]))

# gyro_noise_std                  
# accel_noise_std
# feature_measurement_noise_std
# initial_range_std_m
# initial_velocity_std_mps
# max_history_s
# max_step_s
# history_sync_tolerance_s
# innovation_gate_chi2
# IMU噪声有多大
# 视觉测量有多可信
# 初始距离不确定性多大
# 历史缓存保留多久
# 视觉延迟匹配允许多大时间误差
# 创新门限多严格

@dataclass(frozen=True)
class EKFParameters:
    gravity_world: tuple = (0.0, 0.0, -9.80665)
    gyro_noise_std: float = 0.015
    accel_noise_std: float = 0.20
    target_accel_process_std: float = 0.0
    gyro_bias_rw_std: float = 0.0005
    accel_bias_rw_std: float = 0.01
    feature_process_noise_std: float = 0.002
    feature_measurement_noise_std: float = 0.006
    initial_attitude_std: float = 0.03
    initial_range_std_m: float = 30.0
    initial_lateral_std_m: float = 1.0
    initial_velocity_std_mps: float = 5.0
    initial_gyro_bias_std: float = 0.03
    initial_accel_bias_std: float = 0.3
    max_history_s: float = 0.5
    max_step_s: float = 0.03
    history_sync_tolerance_s: float = 0.008
    innovation_gate_chi2: float = 13.82
    min_depth_m: float = 0.2
    jacobian_epsilon: float = 1e-6

    def validate(self):
        positive = (
            self.gyro_noise_std,
            self.accel_noise_std,
            self.feature_measurement_noise_std,
            self.initial_range_std_m,
            self.initial_velocity_std_mps,
            self.max_history_s,
            self.max_step_s,
            self.history_sync_tolerance_s,
            self.innovation_gate_chi2,
            self.min_depth_m,
            self.jacobian_epsilon,
        )
        if min(positive) <= 0.0:
            raise ValueError("EKF standard deviations, time limits and gates must be positive")
        if (
            not math.isfinite(self.target_accel_process_std)
            or self.target_accel_process_std < 0.0
        ):
            raise ValueError("target acceleration process std must be finite and nonnegative")


@dataclass
class IMUStep:
    angular_velocity_body: np.ndarray
    specific_force_body: np.ndarray
    dt: float


@dataclass
class Snapshot:
    stamp: float
    state: np.ndarray
    covariance: np.ndarray
    step_from_previous: IMUStep = None
    attitude_xyzw: np.ndarray = None
    attitude_measurement_std: float = None
    velocity_world: np.ndarray = None
    velocity_measurement_std: float = None
    velocity_radial_only: bool = False
    velocity_transverse_measurement_std: float = math.inf


@dataclass(frozen=True)
class UpdateResult:
    accepted: bool
    reason: str
    innovation_mahalanobis: float
    replayed_steps: int


class DelayCompensatedEKF:
    """Paper-structured 18-state EKF with delayed feature replay.

    The public paper does not provide initialization values or complete DC-EKF
    covariance equations. This implementation keeps its state and nonlinear
    propagation structure, uses a sparse first-order linearization, and makes
    the range prior explicit rather than treating monocular pixels as depth.
    """

    def __init__(self, parameters, rotation_body_camera):
        parameters.validate()
        self.parameters = parameters
        self.rotation_body_camera = np.asarray(rotation_body_camera, dtype=float).reshape(3, 3)
        self.history = deque()
        self.measurement_count = 0
        self.last_feature_stamp = None
        self.last_innovation_mahalanobis = math.nan

    @property
    def initialized(self):
        return bool(self.history)

    @property
    def stamp(self):
        return self.history[-1].stamp if self.history else None

    @property
    def state(self):
        if not self.history:
            raise RuntimeError("EKF is not initialized")
        return self.history[-1].state.copy()

    @property
    def covariance(self):
        if not self.history:
            raise RuntimeError("EKF is not initialized")
        return self.history[-1].covariance.copy()

    def reset(self):
        self.history.clear()
        self.measurement_count = 0
        self.last_feature_stamp = None
        self.last_innovation_mahalanobis = math.nan

    def initialize(
        self,
        stamp,
        attitude_xyzw,
        feature_normalized,
        initial_range_m,
        initial_gyro_bias=None,
        initial_accel_bias=None,
    ):
        if initial_range_m <= self.parameters.min_depth_m:
            raise ValueError("initial_range_m is too small")
        quaternion = np.asarray(attitude_xyzw, dtype=float).reshape(4)
        quaternion /= np.linalg.norm(quaternion)
        feature = np.asarray(feature_normalized, dtype=float).reshape(2)
        rotation_world_body = quaternion_xyzw_to_rotation(quaternion)
        target_direction_world = rotation_world_body.dot(
            self.rotation_body_camera.dot(camera_ray(feature[0], feature[1]))
        )
        # initial_range_m初始距离
        state = np.zeros(STATE_SIZE)
        state[Q] = quaternion
        state[P_R] = -initial_range_m * target_direction_world
        state[FEATURE] = feature
        if initial_gyro_bias is not None:
            state[B_GYR] = np.asarray(initial_gyro_bias, dtype=float).reshape(3)
        if initial_accel_bias is not None:
            state[B_ACC] = np.asarray(initial_accel_bias, dtype=float).reshape(3)

        covariance = np.zeros((STATE_SIZE, STATE_SIZE))
        covariance[Q, Q] = np.eye(4) * self.parameters.initial_attitude_std ** 2
        los_outer = np.outer(target_direction_world, target_direction_world)
        covariance[P_R, P_R] = (
            self.parameters.initial_range_std_m ** 2 * los_outer
            + self.parameters.initial_lateral_std_m ** 2 * (np.eye(3) - los_outer)
        )
        covariance[V_R, V_R] = (
            np.eye(3) * self.parameters.initial_velocity_std_mps ** 2
        )
        covariance[FEATURE, FEATURE] = (
            np.eye(2) * self.parameters.feature_measurement_noise_std ** 2
        )
        covariance[B_GYR, B_GYR] = (
            np.eye(3) * self.parameters.initial_gyro_bias_std ** 2
        )
        covariance[B_ACC, B_ACC] = (
            np.eye(3) * self.parameters.initial_accel_bias_std ** 2
        )
        self.reset()
        self.history.append(Snapshot(float(stamp), state, covariance))
        self.measurement_count = 1
        self.last_feature_stamp = float(stamp)
        self.last_innovation_mahalanobis = 0.0

    # 预测PR和VR
    def _propagate_state(self, state, angular_velocity_body, specific_force_body, dt):
        result = np.asarray(state, dtype=float).copy()
        rotation_world_body = quaternion_xyzw_to_rotation(result[Q])
        omega_body = np.asarray(angular_velocity_body, dtype=float).reshape(3) - result[B_GYR]
        omega_world = rotation_world_body.dot(omega_body)
        delta_world = delta_quaternion_xyzw(omega_world, dt)
        quaternion = quaternion_multiply_xyzw(delta_world, result[Q])
        result[Q] = quaternion / np.linalg.norm(quaternion)

        specific_force = np.asarray(specific_force_body, dtype=float).reshape(3) - result[B_ACC]
        acceleration_world = rotation_world_body.dot(specific_force) + np.asarray(
            self.parameters.gravity_world, dtype=float
        )
        previous_velocity = result[V_R].copy()
        result[P_R] += previous_velocity * dt + 0.5 * acceleration_world * dt * dt
        result[V_R] += acceleration_world * dt

        rotation_world_camera = rotation_world_body.dot(self.rotation_body_camera)
        # Eq. (stateTrans) evaluates every right-hand-side term at k.
        target_position_camera = -rotation_world_camera.T.dot(state[P_R])
        depth = float(target_position_camera[2])
        if depth <= self.parameters.min_depth_m:
            raise ValueError("estimated target depth is non-positive")
        x, y = result[FEATURE]
        translation_jacobian = np.array(
            [[-1.0 / depth, 0.0, x / depth], [0.0, -1.0 / depth, y / depth]]
        )
        rotation_jacobian = np.array(
            [
                [x * y, -(1.0 + x * x), y],
                [1.0 + y * y, -x * y, -x],
            ]
        )
        relative_velocity_camera = rotation_world_camera.T.dot(previous_velocity)
        angular_velocity_camera = self.rotation_body_camera.T.dot(omega_body)
        result[FEATURE] += dt * (
            translation_jacobian.dot(relative_velocity_camera)
            + rotation_jacobian.dot(angular_velocity_camera)
        )
        return result

    def _state_jacobian(self, state, angular_velocity_body, specific_force_body, dt, nominal):
        # Only the quaternion columns need numerical differentiation. The
        # remaining sparse blocks follow directly from the motion and IBVS
        # equations; evaluating all 18 columns is too costly during replay.
        jacobian = np.eye(STATE_SIZE)
        epsilon = self.parameters.jacobian_epsilon
        for index in range(Q.start, Q.stop):
            perturbed = np.asarray(state, dtype=float).copy()
            perturbed[index] += epsilon
            perturbed[Q] /= np.linalg.norm(perturbed[Q])
            propagated = self._propagate_state(
                perturbed, angular_velocity_body, specific_force_body, dt
            )
            jacobian[:, index] = (propagated - nominal) / epsilon

        rotation_world_body = quaternion_xyzw_to_rotation(state[Q])
        rotation_world_camera = rotation_world_body.dot(self.rotation_body_camera)
        camera_forward_world = rotation_world_camera[:, 2]
        omega_body = (
            np.asarray(angular_velocity_body, dtype=float).reshape(3) - state[B_GYR]
        )

        jacobian[P_R, V_R] = np.eye(3) * dt
        jacobian[P_R, B_ACC] = -0.5 * rotation_world_body * dt * dt
        jacobian[V_R, B_ACC] = -rotation_world_body * dt

        target_position_camera = -rotation_world_camera.T.dot(state[P_R])
        depth = float(target_position_camera[2])
        x, y = state[FEATURE]
        translation_jacobian = np.array(
            [[-1.0 / depth, 0.0, x / depth], [0.0, -1.0 / depth, y / depth]]
        )
        rotation_jacobian = np.array(
            [
                [x * y, -(1.0 + x * x), y],
                [1.0 + y * y, -x * y, -x],
            ]
        )
        velocity_camera = rotation_world_camera.T.dot(state[V_R])
        angular_velocity_camera = self.rotation_body_camera.T.dot(omega_body)
        translation_rate = translation_jacobian.dot(velocity_camera)

        feature_derivative = np.array(
            [
                [
                    velocity_camera[2] / depth
                    + y * angular_velocity_camera[0]
                    - 2.0 * x * angular_velocity_camera[1],
                    x * angular_velocity_camera[0] + angular_velocity_camera[2],
                ],
                [
                    -y * angular_velocity_camera[1] - angular_velocity_camera[2],
                    velocity_camera[2] / depth
                    + 2.0 * y * angular_velocity_camera[0]
                    - x * angular_velocity_camera[1],
                ],
            ]
        )
        jacobian[FEATURE, FEATURE] += dt * feature_derivative

        # Depth is z_c=-e3^T R_wc^T p_r. Include its coupling into the
        # translational optical-flow term as well as the direct velocity term.
        depth_rate_derivative = -translation_rate / depth
        depth_position_derivative = -camera_forward_world
        jacobian[FEATURE, P_R] = dt * np.outer(
            depth_rate_derivative, depth_position_derivative
        )
        jacobian[FEATURE, V_R] = dt * translation_jacobian.dot(
            rotation_world_camera.T
        )
        jacobian[FEATURE, B_GYR] = (
            -dt * rotation_jacobian.dot(self.rotation_body_camera.T)
        )

        quaternion = state[Q]
        quaternion_vector = quaternion[:3]
        quaternion_w = quaternion[3]
        skew_quaternion = np.array(
            [
                [0.0, -quaternion_vector[2], quaternion_vector[1]],
                [quaternion_vector[2], 0.0, -quaternion_vector[0]],
                [-quaternion_vector[1], quaternion_vector[0], 0.0],
            ]
        )
        bias_to_quaternion = np.vstack(
            (
                quaternion_w * np.eye(3) + skew_quaternion,
                -quaternion_vector.reshape(1, 3),
            )
        ) * (-0.5 * dt)
        quaternion_projection = np.eye(4) - np.outer(nominal[Q], nominal[Q])
        jacobian[Q, B_GYR] = quaternion_projection.dot(bias_to_quaternion)
        return jacobian

    def _process_covariance(self, state, dt, nominal):
        rotation_world_body = quaternion_xyzw_to_rotation(state[Q])
        x, y = state[FEATURE]
        rotation_jacobian = np.array(
            [
                [x * y, -(1.0 + x * x), y],
                [1.0 + y * y, -x * y, -x],
            ]
        )

        quaternion = state[Q]
        quaternion_vector = quaternion[:3]
        quaternion_w = quaternion[3]
        skew_quaternion = np.array(
            [
                [0.0, -quaternion_vector[2], quaternion_vector[1]],
                [quaternion_vector[2], 0.0, -quaternion_vector[0]],
                [-quaternion_vector[1], quaternion_vector[0], 0.0],
            ]
        )
        gyro_to_quaternion = np.vstack(
            (
                quaternion_w * np.eye(3) + skew_quaternion,
                -quaternion_vector.reshape(1, 3),
            )
        ) * (0.5 * dt)
        quaternion_projection = np.eye(4) - np.outer(nominal[Q], nominal[Q])

        # Propagate the shared IMU noise through the nonlinear state map. This
        # preserves the q/feature and position/velocity cross-covariances.
        imu_noise_jacobian = np.zeros((STATE_SIZE, 6))
        imu_noise_jacobian[Q, 0:3] = quaternion_projection.dot(
            gyro_to_quaternion
        )
        imu_noise_jacobian[FEATURE, 0:3] = (
            dt * rotation_jacobian.dot(self.rotation_body_camera.T)
        )
        imu_noise_jacobian[P_R, 3:6] = 0.5 * rotation_world_body * dt * dt
        imu_noise_jacobian[V_R, 3:6] = rotation_world_body * dt
        imu_noise_covariance = np.diag(
            [self.parameters.gyro_noise_std ** 2] * 3
            + [self.parameters.accel_noise_std ** 2] * 3
        )
        covariance = imu_noise_jacobian.dot(imu_noise_covariance).dot(
            imu_noise_jacobian.T
        )
        # Optional moving-target engineering model. The paper/default path
        # keeps this at zero. Unknown target acceleration enters relative
        # position and velocity independently of interceptor IMU noise.
        if self.parameters.target_accel_process_std > 0.0:
            target_accel_jacobian = np.zeros((STATE_SIZE, 3))
            target_accel_jacobian[P_R, :] = -0.5 * np.eye(3) * dt * dt
            target_accel_jacobian[V_R, :] = -np.eye(3) * dt
            covariance += target_accel_jacobian.dot(
                np.eye(3) * self.parameters.target_accel_process_std ** 2
            ).dot(target_accel_jacobian.T)
        covariance[FEATURE, FEATURE] = (
            covariance[FEATURE, FEATURE]
            + np.eye(2) * self.parameters.feature_process_noise_std ** 2 * dt
        )
        covariance[B_GYR, B_GYR] = (
            np.eye(3) * self.parameters.gyro_bias_rw_std ** 2 * dt
        )
        covariance[B_ACC, B_ACC] = (
            np.eye(3) * self.parameters.accel_bias_rw_std ** 2 * dt
        )
        return covariance

    def _predict(self, state, covariance, step):
        nominal = self._propagate_state(
            state, step.angular_velocity_body, step.specific_force_body, step.dt
        )
        transition = self._state_jacobian(
            state,
            step.angular_velocity_body,
            step.specific_force_body,
            step.dt,
            nominal,
        )
        predicted_covariance = (
            transition.dot(covariance).dot(transition.T)
            + self._process_covariance(state, step.dt, nominal)
        )
        predicted_covariance = 0.5 * (
            predicted_covariance + predicted_covariance.T
        )
        return nominal, predicted_covariance

    def propagate(self, stamp, angular_velocity_body, specific_force_body):
        if not self.history:
            raise RuntimeError("EKF is not initialized")
        stamp = float(stamp)
        dt = stamp - self.history[-1].stamp
        if dt <= 0.0:
            raise ValueError("IMU timestamps must be strictly increasing")
        if dt > self.parameters.max_history_s:
            raise ValueError("IMU gap exceeds retained history")
        segment_count = int(math.ceil(dt / self.parameters.max_step_s))
        segment_dt = dt / segment_count
        angular_velocity = np.asarray(angular_velocity_body, dtype=float).reshape(3)
        specific_force = np.asarray(specific_force_body, dtype=float).reshape(3)
        for segment_index in range(segment_count):
            step = IMUStep(angular_velocity, specific_force, segment_dt)
            state, covariance = self._predict(
                self.history[-1].state, self.history[-1].covariance, step
            )
            segment_stamp = (
                stamp
                if segment_index == segment_count - 1
                else self.history[-1].stamp + segment_dt
            )
            self.history.append(Snapshot(segment_stamp, state, covariance, step))
        cutoff = stamp - self.parameters.max_history_s
        while len(self.history) > 2 and self.history[1].stamp < cutoff:
            self.history.popleft()

    def _history_index(self, stamp, measurement_name):
        distances = [abs(snapshot.stamp - stamp) for snapshot in self.history]
        history_index = int(np.argmin(distances))
        if distances[history_index] > self.parameters.history_sync_tolerance_s:
            return None, UpdateResult(
                False,
                "no_history_at_{}_stamp".format(measurement_name),
                math.nan,
                0,
            )
        return history_index, None

    def _correct_and_replay(
        self,
        history_index,
        observation,
        innovation,
        measurement_covariance,
        enforce_gate=True,
    ):
        snapshot = self.history[history_index]
        correction = self._kalman_correction(
            snapshot.state,
            snapshot.covariance,
            observation,
            innovation,
            measurement_covariance,
            enforce_gate=enforce_gate,
        )
        if correction[0] is None:
            mahalanobis = correction[2]
            return UpdateResult(False, "innovation_gate", mahalanobis, 0)
        corrected_state, corrected_covariance, mahalanobis = correction
        self.last_innovation_mahalanobis = mahalanobis
        replayed_states = []
        previous_state = corrected_state
        previous_covariance = corrected_covariance
        replayed_steps = 0
        try:
            for index in range(history_index + 1, len(self.history)):
                current = self.history[index]
                replayed_state, replayed_covariance = self._predict(
                    previous_state,
                    previous_covariance,
                    current.step_from_previous,
                )
                if current.attitude_xyzw is not None:
                    attitude_terms = self._attitude_observation(
                        replayed_state,
                        current.attitude_xyzw,
                        current.attitude_measurement_std,
                    )
                    replayed_state, replayed_covariance, _ = self._kalman_correction(
                        replayed_state,
                        replayed_covariance,
                        *attitude_terms,
                        enforce_gate=False,
                    )
                if current.velocity_world is not None:
                    velocity_terms = self._velocity_observation(
                        replayed_state,
                        current.velocity_world,
                        current.velocity_measurement_std,
                        radial_only=current.velocity_radial_only,
                        transverse_measurement_std=(
                            current.velocity_transverse_measurement_std
                        ),
                    )
                    replayed_state, replayed_covariance, _ = self._kalman_correction(
                        replayed_state,
                        replayed_covariance,
                        *velocity_terms,
                        enforce_gate=False,
                    )
                replayed_states.append((replayed_state, replayed_covariance))
                previous_state = replayed_state
                previous_covariance = replayed_covariance
                replayed_steps += 1
        except ValueError as error:
            return UpdateResult(
                False,
                "replay_{}".format(str(error).replace(" ", "_")),
                mahalanobis,
                0,
            )

        # Commit only after the full delayed replay succeeds.
        snapshot.state = corrected_state
        snapshot.covariance = corrected_covariance
        for offset, (replayed_state, replayed_covariance) in enumerate(
            replayed_states, start=history_index + 1
        ):
            self.history[offset].state = replayed_state
            self.history[offset].covariance = replayed_covariance
        return UpdateResult(True, "accepted", mahalanobis, replayed_steps)

    def _kalman_correction(
        self,
        state,
        covariance,
        observation,
        innovation,
        measurement_covariance,
        enforce_gate,
    ):
        innovation_covariance = (
            observation.dot(covariance).dot(observation.T)
            + measurement_covariance
        )
        mahalanobis = float(
            innovation.T.dot(np.linalg.solve(innovation_covariance, innovation))
        )
        if enforce_gate and mahalanobis > self.parameters.innovation_gate_chi2:
            return None, None, mahalanobis

        kalman_gain = covariance.dot(observation.T).dot(
            np.linalg.inv(innovation_covariance)
        )
        corrected_state = state + kalman_gain.dot(innovation)
        corrected_state[Q] /= np.linalg.norm(corrected_state[Q])
        identity = np.eye(STATE_SIZE)
        residual_projection = identity - kalman_gain.dot(observation)
        corrected_covariance = (
            residual_projection.dot(covariance).dot(residual_projection.T)
            + kalman_gain.dot(measurement_covariance).dot(kalman_gain.T)
        )
        corrected_covariance = 0.5 * (corrected_covariance + corrected_covariance.T)
        return corrected_state, corrected_covariance, mahalanobis

    def update_feature(self, stamp, feature_normalized, measurement_std=None):
        if not self.history:
            raise RuntimeError("EKF is not initialized")
        stamp = float(stamp)
        feature = np.asarray(feature_normalized, dtype=float).reshape(2)
        history_index, rejection = self._history_index(stamp, "feature")
        if rejection is not None:
            return rejection

        snapshot = self.history[history_index]
        observation = np.zeros((2, STATE_SIZE))
        observation[:, FEATURE] = np.eye(2)
        measurement_sigma = (
            self.parameters.feature_measurement_noise_std
            if measurement_std is None
            else float(measurement_std)
        )
        if measurement_sigma <= 0.0:
            raise ValueError("feature measurement standard deviation must be positive")
        measurement_covariance = np.eye(2) * measurement_sigma ** 2
        innovation = feature - snapshot.state[FEATURE]
        result = self._correct_and_replay(
            history_index, observation, innovation, measurement_covariance
        )
        if not result.accepted:
            return result
        self.measurement_count += 1
        self.last_feature_stamp = stamp
        return result

    def update_attitude(
        self,
        stamp,
        attitude_xyzw,
        measurement_std,
        enforce_innovation_gate=True,
        max_residual_rad=math.pi,
    ):
        """Fuse an optional platform attitude observation without counting it as vision."""
        if not self.history:
            raise RuntimeError("EKF is not initialized")
        stamp = float(stamp)
        measured = np.asarray(attitude_xyzw, dtype=float).reshape(4)
        measured_norm = float(np.linalg.norm(measured))
        if measured_norm < 1e-9:
            raise ValueError("attitude measurement quaternion is invalid")
        measured /= measured_norm
        measurement_sigma = float(measurement_std)
        if measurement_sigma <= 0.0:
            raise ValueError("attitude measurement standard deviation must be positive")
        history_index, rejection = self._history_index(stamp, "attitude")
        if rejection is not None:
            return rejection

        snapshot = self.history[history_index]
        residual = attitude_residual(measured, snapshot.state[Q])
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > float(max_residual_rad):
            return UpdateResult(
                False,
                "attitude_residual_limit",
                (residual_norm / measurement_sigma) ** 2,
                0,
            )
        attitude_terms = self._attitude_observation(
            snapshot.state, measured, measurement_sigma
        )
        result = self._correct_and_replay(
            history_index,
            *attitude_terms,
            enforce_gate=bool(enforce_innovation_gate),
        )
        if result.accepted:
            snapshot.attitude_xyzw = measured.copy()
            snapshot.attitude_measurement_std = measurement_sigma
        return result

    def _attitude_observation(self, state, measured, measurement_sigma):
        innovation = attitude_residual(measured, state[Q])
        observation = np.zeros((3, STATE_SIZE))
        epsilon = self.parameters.jacobian_epsilon
        for index in range(Q.start, Q.stop):
            perturbed = state[Q].copy()
            perturbed[index] += epsilon
            perturbed /= np.linalg.norm(perturbed)
            perturbed_residual = attitude_residual(measured, perturbed)
            observation[:, index] = -(perturbed_residual - innovation) / epsilon
        measurement_covariance = np.eye(3) * measurement_sigma ** 2
        return observation, innovation, measurement_covariance

    def update_velocity(
        self,
        stamp,
        velocity_world,
        measurement_std,
        enforce_innovation_gate=True,
        max_residual_mps=math.inf,
        radial_only=False,
        transverse_measurement_std=math.inf,
    ):
        """Fuse full or line-of-sight-projected relative velocity.

        ``radial_only`` is useful for a moving target whose transverse velocity
        is unknown. The line-of-sight direction is recomputed from the state
        when delayed corrections are replayed.
        """
        if not self.history:
            raise RuntimeError("EKF is not initialized")
        stamp = float(stamp)
        measured = np.asarray(velocity_world, dtype=float).reshape(3)
        if not np.all(np.isfinite(measured)):
            raise ValueError("velocity measurement is invalid")
        measurement_sigma = float(measurement_std)
        if measurement_sigma <= 0.0:
            raise ValueError("velocity measurement standard deviation must be positive")
        transverse_sigma = float(transverse_measurement_std)
        if transverse_sigma <= 0.0 or math.isnan(transverse_sigma):
            raise ValueError(
                "transverse velocity measurement standard deviation must be positive"
            )
        history_index, rejection = self._history_index(stamp, "velocity")
        if rejection is not None:
            return rejection

        snapshot = self.history[history_index]
        if radial_only:
            projection_direction = normalize(
                snapshot.state[P_R], "relative position for radial velocity aiding"
            )
            residual_norm = abs(
                float(np.dot(projection_direction, measured - snapshot.state[V_R]))
            )
        else:
            residual_norm = float(np.linalg.norm(measured - snapshot.state[V_R]))
        if residual_norm > float(max_residual_mps):
            return UpdateResult(
                False,
                "velocity_residual_limit",
                (residual_norm / measurement_sigma) ** 2,
                0,
            )
        velocity_terms = self._velocity_observation(
            snapshot.state,
            measured,
            measurement_sigma,
            radial_only=bool(radial_only),
            transverse_measurement_std=transverse_sigma,
        )
        result = self._correct_and_replay(
            history_index,
            *velocity_terms,
            enforce_gate=bool(enforce_innovation_gate),
        )
        if result.accepted:
            snapshot.velocity_world = measured.copy()
            snapshot.velocity_measurement_std = measurement_sigma
            snapshot.velocity_radial_only = bool(radial_only)
            snapshot.velocity_transverse_measurement_std = transverse_sigma
        return result

    @staticmethod
    def _velocity_observation(
        state,
        measured,
        measurement_sigma,
        radial_only=False,
        transverse_measurement_std=math.inf,
    ):
        if radial_only:
            direction = normalize(
                state[P_R], "relative position for radial velocity aiding"
            )
            if math.isfinite(transverse_measurement_std):
                observation = np.zeros((3, STATE_SIZE))
                observation[:, V_R] = np.eye(3)
                innovation = measured - state[V_R]
                radial_projection = np.outer(direction, direction)
                transverse_projection = np.eye(3) - radial_projection
                measurement_covariance = (
                    measurement_sigma ** 2 * radial_projection
                    + transverse_measurement_std ** 2 * transverse_projection
                )
                return observation, innovation, measurement_covariance
            observation = np.zeros((1, STATE_SIZE))
            observation[0, V_R] = direction
            innovation = np.array([float(np.dot(direction, measured - state[V_R]))])
            measurement_covariance = np.array([[measurement_sigma ** 2]])
            return observation, innovation, measurement_covariance
        observation = np.zeros((3, STATE_SIZE))
        observation[:, V_R] = np.eye(3)
        innovation = measured - state[V_R]
        measurement_covariance = np.eye(3) * measurement_sigma ** 2
        return observation, innovation, measurement_covariance

    def position_standard_deviation(self):
        eigenvalues = np.linalg.eigvalsh(self.history[-1].covariance[P_R, P_R])
        return math.sqrt(max(float(np.max(eigenvalues)), 0.0))

    def vision_age(self):
        if self.last_feature_stamp is None or self.stamp is None:
            return math.inf
        return max(0.0, self.stamp - self.last_feature_stamp)

    # 1. EKF已经初始化
    # 2. 视觉测量次数足够
    # 3. 相对位置协方差不要太大
    # 4. 最近一次视觉更新时间不能太久
    def estimate_valid(self, minimum_measurements, max_position_std_m, max_vision_age_s):
        return (
            self.initialized
            and self.measurement_count >= int(minimum_measurements)
            and self.position_standard_deviation() <= float(max_position_std_m)
            and self.vision_age() <= float(max_vision_age_s)
        )
