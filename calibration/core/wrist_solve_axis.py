"""Solver for wrist camera + roll axis calibration.

Two-step approach:
  1. Measure roll axis DIRECTION from consecutive-view rotation deltas (linear, robust)
  2. Fix axis direction, solve in gauge-projected space:
       mount(6) + roll_zero(1) + axis_lateral_position(2)  =  9 params, 8 free

The roll axis direction is measured directly rather than optimized, because the
rotation deltas give a clean linear estimate (the nonlinear solve tends to fall
into local minima on direction). The axis lateral position accounts for the
physical roll joint not passing through the parent body origin as XML assumes.

Gauge freedom (roll_zero <-> mount rotation about the roll axis) is projected out
so the problem is well conditioned (condition number ~120 instead of ~1e9).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import se3, gates, wrist_model_final as wm

# Weighting for rotation vs translation in residuals
ROT_SCALE = 0.01  # 1 cm ~ 1 deg in cost
POSE_F_SCALE_M = 0.05  # soft_l1 transition scale


def _measure_roll_axis(
    sim,
    arm: str,
    angles_list: list[dict],
    observed_poses: list[np.ndarray],
    indices: list[int],
    T_W_B: np.ndarray,
    T_B_A: np.ndarray,
) -> np.ndarray:
    """Measure physical roll axis from rotation deltas (parent frame).
    
    For consecutive pairs of views (sorted by roll), extract the relative rotation
    axis in the parent (Wrist_Pitch_Roll) frame. Average over all pairs.
    """
    parent_body = wm.PARENT_BODIES[arm]
    roll_motor = wm.ARM_JOINT_NAMES[arm][4]
    
    # Express observed camera rotation in parent frame for each view
    R_parent_cam = []
    rolls = []
    for i in indices:
        sim.set_joints(angles_list[i])
        pp, pr = sim.body_pose_in_chassis(parent_body)
        T_base_parent = se3.make_transform(pr, pp)
        R_W_parent = (T_W_B @ T_B_A @ T_base_parent)[:3, :3]
        R_W_cam = se3.invert(observed_poses[i])[:3, :3]
        R_parent_cam.append(R_W_parent.T @ R_W_cam)
        rolls.append(angles_list[i][roll_motor])
    
    rolls = np.array(rolls)
    order = np.argsort(rolls)
    
    # Extract rotation axis from each consecutive pair
    axes = []
    for k in range(len(order) - 1):
        i, j = order[k], order[k + 1]
        R_rel = R_parent_cam[j] @ R_parent_cam[i].T
        M = np.eye(4)
        M[:3, :3] = R_rel
        rvec = se3.log_se3(M)[:3]
        axis = rvec / (np.linalg.norm(rvec) + 1e-12)
        # Flip to align with roll_delta sign (axis should point consistently)
        if rolls[j] - rolls[i] < 0:
            axis = -axis
        axes.append(axis)
    
    # Average and normalize
    mean_axis = np.mean(axes, axis=0)
    return mean_axis / np.linalg.norm(mean_axis)


def fit(
    sim,
    arm: str,
    reported_angles: list[dict],
    observed_poses: list[np.ndarray],
    rough_roll_zero_rad: float,
    T_W_B: np.ndarray,
    T_B_A: np.ndarray,
    holdout_fraction: float = 0.25,
) -> dict | None:
    """Fit camera mount, roll zero, and roll axis from wrist camera views.
    
    Two-step process:
      1. Measure axis direction from fit views' rotation deltas
      2. Fix axis, optimize mount + roll_zero + offset (8 params)
    
    Returns dict with fit results and diagnostics, or None if fit fails.
    """
    n = len(reported_angles)
    if n < gates.WRIST_MIN_VIEWS:
        return None
    
    n_holdout = max(1, int(n * holdout_fraction))
    n_fit = n - n_holdout
    
    # Split fit/holdout by roll angle diversity
    roll_motor = wm.ARM_JOINT_NAMES[arm][4]
    rolls = np.array([a[roll_motor] for a in reported_angles])
    order = np.argsort(rolls)
    fit_idx = list(order[::2][:n_fit])  # every other, sorted by roll
    holdout_idx = list(order[1::2][:n_holdout])
    
    # STEP 1: Measure axis DIRECTION from fit views' rotation deltas
    axis_dir = _measure_roll_axis(
        sim, arm, reported_angles, observed_poses, fit_idx, T_W_B, T_B_A
    )
    perp1, perp2 = wm.perpendicular_basis(axis_dir)
    
    # STEP 2: Fix axis direction, solve in gauge-projected space.
    # Free params (8) = mount(6) + roll_zero(1) + lateral(2) minus the 1D gauge.
    guess = wm.initial_guess(arm)
    
    def residual(free):
        p = wm.with_free(free, arm, axis_dir, guess)
        r = wm.residuals(
            p, sim, arm, axis_dir, perp1, perp2,
            [reported_angles[i] for i in fit_idx],
            [observed_poses[i] for i in fit_idx],
            rough_roll_zero_rad, T_W_B, T_B_A
        )
        n_views = len(fit_idx)
        weights = np.tile([ROT_SCALE, ROT_SCALE, ROT_SCALE, 1.0, 1.0, 1.0], n_views)
        return r * weights
    
    out = least_squares(
        residual, wm.free_of(guess, arm, axis_dir), method="trf",
        loss="soft_l1", f_scale=POSE_F_SCALE_M, x_scale="jac",
        xtol=1e-14, ftol=1e-14, max_nfev=4000
    )
    
    if not out.success:
        return None
    
    p_solved = wm.with_free(out.x, arm, axis_dir, guess)
    T_wrist_cam, roll_zero_corr, axis_point = wm.unpack(p_solved, arm, axis_dir,
                                                        perp1, perp2)
    
    # Compute errors on fit and holdout sets
    def pose_errors(indices):
        r = wm.residuals(
            p_solved, sim, arm, axis_dir, perp1, perp2,
            [reported_angles[i] for i in indices],
            [observed_poses[i] for i in indices],
            rough_roll_zero_rad, T_W_B, T_B_A
        )
        r_mat = r.reshape(len(indices), 6)
        trans = np.linalg.norm(r_mat[:, 3:], axis=1) * 1000  # mm
        rot = np.rad2deg(np.linalg.norm(r_mat[:, :3], axis=1))
        return trans, rot
    
    fit_trans, fit_rot = pose_errors(fit_idx)
    holdout_trans, holdout_rot = pose_errors(holdout_idx)
    
    # Roll sweep
    roll_vals = [reported_angles[i][roll_motor] for i in range(n)]
    roll_sweep_deg = np.rad2deg(max(roll_vals) - min(roll_vals))
    
    # Condition number of the gauge-projected Jacobian
    J = out.jac
    if J.shape[1] > 0:
        _, s, _ = np.linalg.svd(J, full_matrices=False)
        cond = s[0] / (s[-1] + 1e-16)
    else:
        cond = 1.0
    
    # Angle of measured axis off the XML nominal +Y (diagnostic)
    axis_off_y_deg = np.rad2deg(np.arccos(np.clip(abs(axis_dir @ [0, 1, 0]), -1, 1)))
    
    result = {
        "arm": arm,
        "params": p_solved.tolist(),
        "T_wrist_cam": T_wrist_cam.tolist(),
        "mount_translation_mm": (T_wrist_cam[:3, 3] * 1000).tolist(),
        "mount_rotation_deg": np.rad2deg(se3.log_se3(T_wrist_cam)[:3]).tolist(),
        "wrist_roll_zero_correction_rad": roll_zero_corr,
        "wrist_roll_zero_correction_deg": np.rad2deg(roll_zero_corr),
        "roll_axis_direction": axis_dir.tolist(),
        "roll_axis_off_nominal_deg": float(axis_off_y_deg),
        "roll_axis_point_mm": (axis_point * 1000).tolist(),
        "n_views_total": n,
        "n_views_fit": len(fit_idx),
        "n_views_holdout": len(holdout_idx),
        "roll_sweep_deg": roll_sweep_deg,
        "fit_trans_rms_mm": float(np.sqrt((fit_trans**2).mean())),
        "fit_trans_max_mm": float(fit_trans.max()),
        "fit_rot_rms_deg": float(np.sqrt((fit_rot**2).mean())),
        "fit_rot_max_deg": float(fit_rot.max()),
        "holdout_trans_rms_mm": float(np.sqrt((holdout_trans**2).mean())),
        "holdout_trans_max_mm": float(holdout_trans.max()),
        "holdout_rot_rms_deg": float(np.sqrt((holdout_rot**2).mean())),
        "holdout_rot_max_deg": float(holdout_rot.max()),
        "condition_number": float(cond),
        "converged": out.success,
    }
    
    return result


def grade(result: dict) -> dict:
    """Apply acceptance gates to a fit result."""
    g = []
    
    # View count
    n = result["n_views_total"]
    passed = n >= gates.WRIST_MIN_VIEWS
    warn = n < 15
    detail = "below the comfortable 15" if warn else ""
    g.append({
        "name": "view count",
        "passed": passed,
        "value": n,
        "threshold": gates.WRIST_MIN_VIEWS,
        "detail": detail,
        "line": f"[{'WARN' if warn and passed else ('OK  ' if passed else 'FAIL')}] view count: {n} (min {gates.WRIST_MIN_VIEWS}) {detail}".strip(),
    })
    
    # Roll sweep
    sweep = result["roll_sweep_deg"]
    passed = sweep >= gates.WRIST_MIN_ROLL_SWEEP_DEG
    g.append({
        "name": "wrist_roll sweep",
        "passed": passed,
        "value": sweep,
        "threshold": gates.WRIST_MIN_ROLL_SWEEP_DEG,
        "detail": "",
        "line": f"[{'OK  ' if passed else 'FAIL'}] wrist_roll sweep: {sweep:.3f} deg (min {gates.WRIST_MIN_ROLL_SWEEP_DEG:.3f} deg)",
    })
    
    # Holdout translation error (using WRIST_RESIDUAL_MAX_MM = 8.0)
    val = result["holdout_trans_rms_mm"]
    passed = val <= gates.WRIST_RESIDUAL_MAX_MM
    warn = val > gates.WRIST_RESIDUAL_GOOD_MM
    detail = ""
    if warn and passed:
        detail = f"above the comfortable {gates.WRIST_RESIDUAL_GOOD_MM:.1f} mm"
    g.append({
        "name": "holdout translation error",
        "passed": passed,
        "value": val,
        "threshold": gates.WRIST_RESIDUAL_MAX_MM,
        "detail": detail,
        "line": f"[{'WARN' if warn and passed else ('OK  ' if passed else 'FAIL')}] holdout translation error: {val:.3f} mm (max {gates.WRIST_RESIDUAL_MAX_MM:.3f} mm) {detail}".strip(),
    })
    
    # Holdout rotation error (using WRIST_ROT_MAX_DEG = 3.0)
    val = result["holdout_rot_rms_deg"]
    passed = val <= gates.WRIST_ROT_MAX_DEG
    warn = val > gates.WRIST_ROT_GOOD_DEG
    detail = ""
    if warn and passed:
        detail = f"above the comfortable {gates.WRIST_ROT_GOOD_DEG:.1f} deg"
    g.append({
        "name": "holdout rotation error",
        "passed": passed,
        "value": val,
        "threshold": gates.WRIST_ROT_MAX_DEG,
        "detail": detail,
        "line": f"[{'WARN' if warn and passed else ('OK  ' if passed else 'FAIL')}] holdout rotation error: {val:.3f} deg (max {gates.WRIST_ROT_MAX_DEG:.3f} deg) {detail}".strip(),
    })
    
    # Worst holdout translation (using 2.5× RMS threshold)
    val = result["holdout_trans_max_mm"]
    threshold = gates.WRIST_RESIDUAL_MAX_MM * 2.5  # 20mm
    passed = val <= threshold
    detail = "one bad view usually means the board was misdetected or moved" if not passed else ""
    g.append({
        "name": "worst holdout translation",
        "passed": passed,
        "value": val,
        "threshold": threshold,
        "detail": detail,
        "line": f"[{'OK  ' if passed else 'FAIL'}] worst holdout translation: {val:.3f} mm (max {threshold:.3f} mm) {detail}".strip(),
    })
    
    # Holdout / fit ratio
    val = result["holdout_trans_rms_mm"] / (result["fit_trans_rms_mm"] + 1e-9)
    threshold = 3.0
    passed = val <= threshold
    g.append({
        "name": "holdout / fit ratio",
        "passed": passed,
        "value": val,
        "threshold": threshold,
        "detail": "",
        "line": f"[{'OK  ' if passed else 'FAIL'}] holdout / fit ratio: {val:.3f} (max {threshold:.3f})",
    })
    
    # Condition number
    val = result["condition_number"]
    threshold = 1e6
    passed = val <= threshold
    detail = "with sufficient roll sweep and view variety this should be well conditioned" if not passed else ""
    g.append({
        "name": "condition number",
        "passed": passed,
        "value": val,
        "threshold": threshold,
        "detail": detail,
        "line": f"[{'OK  ' if passed else 'FAIL'}] condition number: {val:.3f} (max {threshold:.3f}) {detail}".strip(),
    })
    
    result["gates"] = g
    result["passed"] = all(gate["passed"] for gate in g)
    return result


MIN_VIEWS = gates.WRIST_MIN_VIEWS
