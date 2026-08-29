"""Fusion calibration: solve arm zeros + mounting + camera mount from wrist views.

This combines Stage 5 (arm contact) and Stage 6 (wrist camera) into a single solve:
instead of touching a board with a fixed point on the gripper, the wrist camera
observes the board from multiple arm poses. The camera's 6-DoF pose per view gives
richer geometric constraints than a single 3D contact point, breaking degeneracies
that plague contact-based calibration (especially wrist_flex ↔ touch_point coupling).

What this solves
----------------
Per arm, 15 parameters (wrist_roll zero is held fixed at Stage 4's rough value):
  - T_B_A (5 DoF): arm mounting, with yaw held by convention (same as Stage 5)
  - 4 joint zeros: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex
  - T_wrist_cam (6 DoF): camera mount on Fixed_Jaw body

What is NOT solved (deferred or held)
--------------------------------------
  - wrist_roll zero: held at Stage 4's rough value to avoid gauge freedom with
    camera roll (see core/wrist_model.py gauge discussion). This can be solved
    separately in Stage 6 once the other parameters are known.
  - touch point: not used; we're vision-only

Residuals
---------
Each capture contributes 6 residuals (3 rotation + 3 translation):
    r = log_SE3(T_cam_obs^-1 @ T_cam_pred)

where T_cam_obs comes from PnP (image → T_cam_board, board → T_W_board → T_W_cam),
and T_cam_pred is forward kinematics through the arm + camera mount.

Initial guess
-------------
  - Arm mounting: identity (no correction)
  - Joint zeros: all 0° (Stage 4's rough zeros are "correct")
  - Camera mount: XML nominal from wrist_model.nominal_mount()

Conditioning
------------
With 10-15 diverse arm poses (varying all 4 joints, especially wrist_flex), the
condition number should be ~50-200. If it exceeds 1000, warn about insufficient
pose diversity or gimbal lock configurations.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import arm_model, arm_solve, se3, servos, solver as solver_mod, wrist_model


ROTATION_LEVER_M = 0.1


# Full parameter layout (16-vector). se3 vectors are (rotvec, translation), i.e.
# [rx, ry, rz, tx, ty, tz], matching core/se3.py and core/arm_model.py.
#   [0:6]   mount se3  (T_B_A)      -- rotation about z (index 2) is the gauge
#   [6:10]  four joint zeros        -- shoulder_pan/lift, elbow_flex, wrist_flex
#   [10:16] camera mount se3 (T_wrist_cam)
N_PARAMS = 16
MOUNT_BLOCK = slice(0, 6)
ZERO_BLOCK = slice(6, 10)
CAM_BLOCK = slice(10, 16)

# The mount's yaw (rotation about z, index 2) is degenerate with the shoulder_pan
# zero -- the same gauge freedom arm_model documents. Hold it, let pan absorb it.
MOUNT_YAW_INDEX = 2
FREE_INDICES = tuple(i for i in range(N_PARAMS) if i != MOUNT_YAW_INDEX)


def pack(T_B_A: np.ndarray, zeros: dict, T_wrist_cam: np.ndarray,
         arm: str) -> np.ndarray:
    """Mounting (4x4), zero corrections, camera mount (4x4) -> full 16-vector."""
    p = np.zeros(N_PARAMS)
    p[MOUNT_BLOCK] = se3.log_se3(np.asarray(T_B_A, float))
    jnames = arm_model.joint_names(arm)
    for i, jn in enumerate(jnames):
        p[6 + i] = float(zeros.get(jn, 0.0))
    p[CAM_BLOCK] = se3.log_se3(np.asarray(T_wrist_cam, float))
    return p


def unpack(p: np.ndarray, arm: str) -> tuple[np.ndarray, dict, np.ndarray]:
    """Full 16-vector -> (T_B_A 4x4, zeros dict, T_wrist_cam 4x4)."""
    p = np.asarray(p, float).reshape(-1)
    if p.size != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} parameters, got {p.size}")
    T_B_A = se3.exp_se3(p[MOUNT_BLOCK])
    jnames = arm_model.joint_names(arm)
    zeros = {jn: float(p[6 + i]) for i, jn in enumerate(jnames)}
    T_wrist_cam = se3.exp_se3(p[CAM_BLOCK])
    return T_B_A, zeros, T_wrist_cam


def free_of(p: np.ndarray) -> np.ndarray:
    """The 15 components the optimiser may move (mount yaw held)."""
    return np.asarray(p, float)[list(FREE_INDICES)]


def with_free(p_full: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Reinsert the 15 free components into a full 16-vector."""
    p = np.asarray(p_full, float).copy()
    p[list(FREE_INDICES)] = np.asarray(free, float)
    return p


def initial_guess(arm: str, camera_mount: np.ndarray | None = None) -> np.ndarray:
    """Starting point with an explicit camera-frame convention."""
    T_wrist_cam = (wrist_model.nominal_mount(arm)
                   if camera_mount is None else np.asarray(camera_mount, float))
    return pack(np.eye(4), {}, T_wrist_cam, arm)


# The predicted camera pose reuses wrist_model.camera_in_world verbatim: same FK
# chain (T_W_B @ T_B_A @ T_base_jaw @ T_wrist_cam), same Fixed_Jaw body. Keeping a
# single implementation means the fusion solve and stage 6 can never drift apart.


def residuals_one_view(p: np.ndarray, sim, arm: str, capture: dict,
                       rough_zeros: dict, senses: dict, T_W_B: np.ndarray,
                       intrinsics: dict) -> np.ndarray:
    """Residual for one camera view: 6D pose error in se3 log space.
    
    capture = {raw, T_cam_board, ...}
    rough_zeros = {joint_name: raw_value} from Stage 4
    senses = {joint_name: +1 or -1}
    T_W_B: chassis → world transform from Stage 3
    """
    T_B_A, zero_corrections, T_wrist_cam = unpack(p, arm)
    
    # Prefer the continuous angles latched by the capture thread. Recomputing
    # from a single raw count would choose the shortest arc at the 4095/0 seam.
    angles_stage4 = capture.get('angles')
    if angles_stage4 is None:
        raise ValueError(
            "fusion capture lacks range-resolved angles; recapture with Stage 5 Fusion")
    jnames = arm_model.joint_names(arm)
    angles_true = {**angles_stage4}
    for jn in jnames:
        angles_true[jn] += zero_corrections[jn]
    
    # wrist_roll is held fixed by the optimiser, but its measured angle still
    # comes from the capture's continuous tracker. Only legacy captures without
    # angles fall back to a single-turn raw conversion above.
    
    # Predicted camera pose using wrist_model's FK chain
    T_W_cam_pred = wrist_model.camera_in_world(sim, arm, angles_true, T_wrist_cam,
                                                 T_W_B, T_B_A)
    
    # Observed camera pose from PnP: world/board frame
    T_cam_board = np.asarray(capture['T_cam_board'], float).reshape(4, 4)
    T_W_cam_obs = se3.invert(T_cam_board)  # board is world, so invert
    
    # Residual in se3 log space (6D)
    return se3.log_se3(se3.invert(T_W_cam_obs) @ T_W_cam_pred)


def fit(sim, arm: str, captures: list[dict], intrinsics: dict, T_W_B: np.ndarray,
        rough_zeros: dict, senses: dict,
        camera_mount: np.ndarray | None = None) -> dict:
    """Solve arm mounting + 4 joint zeros + camera mount from wrist camera views.
    
    Parameters
    ----------
    sim : SimModel
        Kinematics model
    arm : str
        "left_arm" or "right_arm"
    captures : list of dict
        Each has {raw, T_cam_board, reproj_px, n_corners, ...}
        raw = {joint_name: raw_counts}
        T_cam_board = 4x4 from PnP
    intrinsics : dict
        Camera intrinsics (not used in current formulation, but available)
    T_W_B : ndarray 4x4
        Chassis → world transform from Stage 3
    rough_zeros : dict
        Stage 4 rough zeros: {joint_name: raw_value}
    senses : dict
        Joint directions: {joint_name: +1 or -1}
    
    Returns
    -------
    dict with:
        T_B_A : ndarray 4x4
            Arm mounting correction
        zeros_deg : dict
            Joint zero corrections in degrees
        T_wrist_cam : ndarray 4x4
            Camera mount on Fixed_Jaw
        condition_number : float
        rms_mm : float
            RMS position error across all views
        rms_deg : float
            RMS rotation error across all views
        captures_used : int
    """
    if len(captures) < 8:
        raise ValueError(f"Need at least 8 views for stable fit, got {len(captures)}")
    
    fit_idx, hold_idx = solver_mod.split_holdout(
        len(captures), fraction=0.25, seed=0, minimum=3)

    def residual_all(p_free):
        p_full = with_free(guess, p_free)
        out = []
        for index in fit_idx:
            r = residuals_one_view(p_full, sim, arm, captures[index],
                                   rough_zeros, senses, T_W_B, intrinsics)
            # A radian is dimensionless and otherwise dominates metre residuals.
            # Scale rotation by a representative 100 mm camera lever arm.
            out.append(np.concatenate([r[:3] * ROTATION_LEVER_M, r[3:]]))
        return np.concatenate(out)
    
    guess = initial_guess(arm, camera_mount=camera_mount)
    guess_free = free_of(guess)

    result = least_squares(
        residual_all,
        guess_free,
        method='trf',
        loss='soft_l1',
        f_scale=0.005,  # 5mm scale for pose errors
        x_scale='jac',
        max_nfev=2000,
        xtol=1e-14,
        ftol=1e-14
    )
    
    p_final = with_free(guess, result.x)
    T_B_A, zeros, T_wrist_cam = unpack(p_final, arm)
    
    def errors(indices):
        pos, rot = [], []
        for index in indices:
            r = residuals_one_view(p_final, sim, arm, captures[index],
                                   rough_zeros, senses, T_W_B, intrinsics)
            pos.append(np.linalg.norm(r[3:]))
            rot.append(np.linalg.norm(r[:3]))
        return np.asarray(pos), np.asarray(rot)

    fit_pos, fit_rot = errors(fit_idx)
    hold_pos, hold_rot = errors(hold_idx)
    
    # Condition number from Jacobian
    try:
        U, s, Vt = np.linalg.svd(result.jac, full_matrices=False)
        condition = s[0] / s[-1] if s[-1] > 1e-12 else 1e12
    except:
        condition = -1.0
    
    zeros_deg = {jn: float(np.rad2deg(zeros[jn])) for jn in zeros}

    return {
        'T_B_A': T_B_A,
        'zeros_deg': zeros_deg,
        'T_wrist_cam': T_wrist_cam,
        'condition_number': float(condition),
        'rms_mm': float(np.sqrt(np.mean(fit_pos**2)) * 1000),
        'rms_deg': float(np.rad2deg(np.sqrt(np.mean(fit_rot**2)))),
        'fit_rms_mm': float(np.sqrt(np.mean(fit_pos**2)) * 1000),
        'fit_rms_deg': float(np.rad2deg(np.sqrt(np.mean(fit_rot**2)))),
        'holdout_rms_mm': float(np.sqrt(np.mean(hold_pos**2)) * 1000),
        'holdout_rms_deg': float(np.rad2deg(np.sqrt(np.mean(hold_rot**2)))),
        'n_views_fit': int(len(fit_idx)),
        'n_views_holdout': int(len(hold_idx)),
        'captures_used': len(captures),
        'success': result.success,
        'message': result.message
    }
