"""Pure Stage 8 validation geometry and statistics.

The functions here never fit calibration parameters. They compare fixed-model
predictions with new ChArUco observations and retain the raw discrepancy even
when a shared board/base drift is estimated for diagnosis.
"""

from __future__ import annotations

import numpy as np

from . import head_model, se3, wrist_model


def observed_camera_pose(T_cam_board: np.ndarray) -> np.ndarray:
    """PnP board pose in camera frame -> camera pose in world/board frame."""
    return se3.invert(np.asarray(T_cam_board, float))


def head_camera_pose(head: dict, pan: float, tilt: float,
                     senses: tuple[float, float]) -> np.ndarray:
    origin = np.asarray(head["gauge"]["pan_axis_origin_m"], float)
    return head_model.T_world_cam(
        np.asarray(head["T_W_B"], float), float(pan), float(tilt),
        np.asarray(head["T_tilt_cam"], float), origin, senses)


def wrist_camera_pose(sim, arm: str, angles: dict[str, float],
                      head: dict, touch: dict) -> np.ndarray:
    arm_result = touch["arms"][arm]
    return wrist_model.camera_in_world(
        sim, arm, angles, np.asarray(arm_result["T_wrist_cam"], float),
        np.asarray(head["T_W_B"], float),
        np.asarray(arm_result["T_B_A"], float))


def angles_from_raw(raw: dict[str, int], zero_raw: dict[str, int],
                    signs: dict[str, int], measured) -> dict[str, float]:
    """Resolve one encoder snapshot through Stage 4's legal single-turn ranges."""
    from .ranges import angles_from_ranges

    return angles_from_ranges(raw, zero_raw, signs, measured)


def head_angles_from_raw(raw: dict[str, int], zeros: dict,
                         senses: dict[str, int]) -> tuple[float, float]:
    """Return head pan/tilt radians from the paired final zero records."""
    from . import servos, zeros as zeros_mod

    zero_set = zeros_mod.ZeroSet.from_dict(zeros.get("zeros", zeros))
    values = []
    for name in ("head_motor_1", "head_motor_2"):
        if name not in raw or name not in zero_set.joints:
            raise ValueError(f"missing head zero or reading: {name}")
        delta = servos.unwrap_delta(int(raw[name]) - zero_set.joints[name].raw)
        values.append(float(senses[name] * delta * servos.RAD_PER_COUNT))
    return values[0], values[1]


def pose_residual(T_observed: np.ndarray, T_predicted: np.ndarray) -> dict:
    """Return a readable SE(3) residual, observed frame to predicted frame."""
    delta = se3.invert(np.asarray(T_observed, float)) @ np.asarray(T_predicted, float)
    xi = se3.log_se3(delta)
    translation = delta[:3, 3]
    return {
        "se3": xi.tolist(),
        "rotation_vector_deg": np.rad2deg(xi[:3]).tolist(),
        "translation_vector_mm": (translation * 1000.0).tolist(),
        "rotation_deg": float(np.rad2deg(np.linalg.norm(xi[:3]))),
        "translation_mm": float(np.linalg.norm(translation) * 1000.0),
    }


def project_board_points(T_W_cam: np.ndarray, object_points: np.ndarray,
                         K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Project board-frame points through a fixed predicted camera pose."""
    import cv2

    T_cam_board = se3.invert(np.asarray(T_W_cam, float))
    rvec = se3.log_so3(T_cam_board[:3, :3]).reshape(3, 1)
    tvec = T_cam_board[:3, 3].reshape(3, 1)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, np.float64), rvec, tvec,
        np.asarray(K, np.float64), np.asarray(dist, np.float64))
    return projected.reshape(-1, 2)


def predicted_pixel_rms(T_W_cam: np.ndarray, detector, detection: dict,
                        K: np.ndarray, dist: np.ndarray) -> float:
    obj, image = detector.match_points(detection["corners"], detection["ids"])
    if obj is None or len(obj) == 0:
        return float("nan")
    projected = project_board_points(T_W_cam, obj, K, dist)
    diff = projected - np.asarray(image, float).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _percentile(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if values.size else None


def summarise_samples(samples: list[dict]) -> dict:
    trans = np.asarray([s["error"]["translation_mm"] for s in samples], float)
    rot = np.asarray([s["error"]["rotation_deg"] for s in samples], float)
    pixels = np.asarray([s.get("predicted_pixel_rms_px", np.nan) for s in samples], float)
    reproj = np.asarray([s.get("pnp_reprojection_px", np.nan) for s in samples], float)
    tvec = np.asarray([s["error"]["translation_vector_mm"] for s in samples], float)
    rvec = np.asarray([s["error"]["rotation_vector_deg"] for s in samples], float)
    valid_pixels = pixels[np.isfinite(pixels)]
    valid_reproj = reproj[np.isfinite(reproj)]
    return {
        "count": int(len(samples)),
        "translation_rms_mm": float(np.sqrt(np.mean(trans ** 2))) if trans.size else None,
        "translation_p95_mm": _percentile(trans, 95),
        "translation_max_mm": float(np.max(trans)) if trans.size else None,
        "rotation_rms_deg": float(np.sqrt(np.mean(rot ** 2))) if rot.size else None,
        "rotation_p95_deg": _percentile(rot, 95),
        "rotation_max_deg": float(np.max(rot)) if rot.size else None,
        "translation_bias_mm": np.mean(tvec, axis=0).tolist() if tvec.size else [None] * 3,
        "rotation_bias_deg": np.mean(rvec, axis=0).tolist() if rvec.size else [None] * 3,
        "predicted_pixel_rms_px": float(np.sqrt(np.mean(valid_pixels ** 2))) if valid_pixels.size else None,
        "pnp_reprojection_rms_px": float(np.sqrt(np.mean(valid_reproj ** 2))) if valid_reproj.size else None,
    }


def mean_transform(transforms: list[np.ndarray], iterations: int = 20) -> np.ndarray:
    """Iterative Lie mean, sufficient for the small Stage 8 drift residuals."""
    if not transforms:
        return np.eye(4)
    mean = np.asarray(transforms[0], float).copy()
    for _ in range(iterations):
        steps = np.asarray([
            se3.log_se3(se3.invert(mean) @ np.asarray(T, float))
            for T in transforms])
        step = np.mean(steps, axis=0)
        mean = mean @ se3.exp_se3(step)
        if np.linalg.norm(step) < 1e-12:
            break
    return mean


def shared_world_drift(samples_by_camera: dict[str, list[dict]]) -> dict:
    """Estimate one world-frame correction shared by all camera observations."""
    transforms = []
    per_camera = {}
    for role, samples in samples_by_camera.items():
        role_transforms = []
        for sample in samples:
            observed = np.asarray(sample["T_W_cam_observed"], float)
            predicted = np.asarray(sample["T_W_cam_predicted"], float)
            # D @ predicted ~= observed when the board/base relation moved.
            role_transforms.append(observed @ se3.invert(predicted))
        per_camera[role] = mean_transform(role_transforms)
        transforms.extend(role_transforms)
    shared = mean_transform(transforms)
    xi = se3.log_se3(shared)
    return {
        "T_observedWorld_predictedWorld": shared.tolist(),
        "translation_mm": (xi[3:] * 1000.0).tolist(),
        "translation_norm_mm": float(np.linalg.norm(xi[3:]) * 1000.0),
        "rotation_vector_deg": np.rad2deg(xi[:3]).tolist(),
        "rotation_deg": float(np.rad2deg(np.linalg.norm(xi[:3]))),
        "per_camera": {role: T.tolist() for role, T in per_camera.items()},
    }


def drift_corrected_samples(samples: list[dict], drift: np.ndarray) -> list[dict]:
    corrected = []
    for sample in samples:
        item = dict(sample)
        predicted = np.asarray(sample["T_W_cam_predicted"], float)
        observed = np.asarray(sample["T_W_cam_observed"], float)
        item["error"] = pose_residual(observed, np.asarray(drift, float) @ predicted)
        corrected.append(item)
    return corrected
