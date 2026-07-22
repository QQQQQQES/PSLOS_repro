import math

import numpy as np

from .geometry import camera_ray


def _seconds(value):
    to_sec = getattr(value, "to_sec", None)
    return float(to_sec() if callable(to_sec) else value)


def current_pixel_los_world(
    feature,
    current_time_s,
    maximum_feature_age_s,
    rotation_world_body,
    rotation_body_camera,
):
    """Return the current pixel ray in the world frame and its age."""

    try:
        detected = bool(feature.detected)
        frame_id = str(feature.header.frame_id)
        feature_stamp_s = _seconds(feature.header.stamp)
        x_normalized = float(feature.x_normalized)
        y_normalized = float(feature.y_normalized)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("feature_schema") from error

    if not detected:
        raise ValueError("feature_not_detected")
    if frame_id != "camera_optical":
        raise ValueError("feature_frame")
    if not math.isfinite(feature_stamp_s) or feature_stamp_s <= 0.0:
        raise ValueError("feature_stamp")
    if not all(math.isfinite(value) for value in (x_normalized, y_normalized)):
        raise ValueError("feature_nonfinite")

    try:
        current_time_s = _seconds(current_time_s)
        maximum_feature_age_s = float(maximum_feature_age_s)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("feature_time_config") from error
    if not math.isfinite(current_time_s):
        raise ValueError("current_time")
    if not math.isfinite(maximum_feature_age_s) or maximum_feature_age_s <= 0.0:
        raise ValueError("maximum_feature_age")

    feature_age_s = current_time_s - feature_stamp_s
    if feature_age_s < 0.0:
        raise ValueError("feature_future")
    if feature_age_s > maximum_feature_age_s:
        raise ValueError("feature_stale")

    try:
        rotation_world_body = np.asarray(rotation_world_body, dtype=float).reshape(3, 3)
        rotation_body_camera = np.asarray(rotation_body_camera, dtype=float).reshape(3, 3)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("feature_rotation") from error
    if not (
        np.all(np.isfinite(rotation_world_body))
        and np.all(np.isfinite(rotation_body_camera))
    ):
        raise ValueError("feature_rotation")

    los_camera = camera_ray(x_normalized, y_normalized)
    los_world = rotation_world_body.dot(rotation_body_camera).dot(los_camera)
    return los_world, feature_age_s


class WorldLOSDerivative:
    """Filtered derivative of distinct, timestamped world-frame LOS samples."""

    def __init__(self, filter_tau_s=0.12, reset_gap_s=0.16, maximum_rate_rad_s=1.5):
        self.filter_tau_s = float(filter_tau_s)
        self.reset_gap_s = float(reset_gap_s)
        self.maximum_rate_rad_s = float(maximum_rate_rad_s)
        if not math.isfinite(self.filter_tau_s) or self.filter_tau_s < 0.0:
            raise ValueError("LOS-rate filter tau must be finite and nonnegative")
        if not math.isfinite(self.reset_gap_s) or self.reset_gap_s <= 0.0:
            raise ValueError("LOS-rate reset gap must be finite and positive")
        if not math.isfinite(self.maximum_rate_rad_s) or self.maximum_rate_rad_s <= 0.0:
            raise ValueError("maximum LOS rate must be finite and positive")
        self.reset()

    def reset(self):
        self.last_stamp_s = None
        self.last_los_world = None
        self.filtered_rate_world = np.zeros(3)
        self.initialized = False

    def update(self, los_world, stamp_s):
        los_world = np.asarray(los_world, dtype=float).reshape(3)
        norm = float(np.linalg.norm(los_world))
        stamp_s = _seconds(stamp_s)
        if not np.all(np.isfinite(los_world)) or norm <= 1e-12:
            raise ValueError("LOS-rate sample must be finite and nonzero")
        if not math.isfinite(stamp_s):
            raise ValueError("LOS-rate stamp must be finite")
        los_world = los_world / norm
        status = "duplicate"
        if self.last_stamp_s is None:
            status = "initialized"
            self.last_stamp_s = stamp_s
            self.last_los_world = los_world
        else:
            dt = stamp_s - self.last_stamp_s
            if dt < 0.0 or dt > self.reset_gap_s:
                status = "reset_non_monotonic" if dt < 0.0 else "reset_gap"
                self.reset()
                self.last_stamp_s = stamp_s
                self.last_los_world = los_world
            elif dt > 0.0:
                status = "updated"
                raw_rate = (los_world - self.last_los_world) / dt
                raw_rate -= los_world * float(np.dot(los_world, raw_rate))
                raw_norm = float(np.linalg.norm(raw_rate))
                if raw_norm > self.maximum_rate_rad_s:
                    raw_rate *= self.maximum_rate_rad_s / raw_norm
                alpha = (
                    1.0
                    if self.filter_tau_s == 0.0
                    else dt / (self.filter_tau_s + dt)
                )
                self.filtered_rate_world += alpha * (
                    raw_rate - self.filtered_rate_world
                )
                self.filtered_rate_world -= los_world * float(
                    np.dot(los_world, self.filtered_rate_world)
                )
                filtered_norm = float(np.linalg.norm(self.filtered_rate_world))
                if filtered_norm > self.maximum_rate_rad_s:
                    self.filtered_rate_world *= self.maximum_rate_rad_s / filtered_norm
                self.last_stamp_s = stamp_s
                self.last_los_world = los_world
                self.initialized = True
        return self.filtered_rate_world.copy(), self.initialized, status
