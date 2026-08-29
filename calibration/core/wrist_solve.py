"""Stage 6 solving: wrist camera extrinsics and wrist_roll zero from board views.

Per arm, solves 7 parameters:
  - T_wrist_cam (6 DoF): camera mount relative to Fixed_Jaw body
  - wrist_roll_zero (1): correction to stage 4's rough wrist_roll zero

The wrist camera sits ~24mm off the wrist_roll axis, so rolling the wrist moves
the camera in a visible arc. This lever arm makes wrist_roll_zero observable
from visual data, unlike in stage 5 where contact data along the axis could not
separate roll from the touch point sliding.

Measurement
-----------
The operator poses the arm so the wrist camera sees the ChArUco board, from many
different arm configurations. Critically, wrist_roll must sweep widely (≥90°) to
separate its zero from the camera mount rotation — the same coupling that makes
the head's tilt zero unsolvable.

Each view records:
  - 5 arm joint encoder readings (including wrist_roll)
  - wrist camera image
  - ChArUco detection → PnP → T_cam_board (= T_cam_world, since W = board)

Residual
--------
e_cam = log_se3(T_obs^-1 @ T_pred)

where T_pred comes from the kinematic chain:
  T_W_cam_pred = T_W_B @ T_B_A @ FK(q_arm) @ T_wrist_cam

Rotation and translation are weighted separately since they carry different units.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import gates, se3, servos, wrist_model

# Robust loss transition. Camera pose from PnP is typically good to a few mm and
# a degree or two; beyond 10mm something is wrong (bad detection, wrong board, etc).
POSE_F_SCALE_M = 0.010
POSE_F_SCALE_RAD = np.deg2rad(2.0)

# Minimum views to attempt a solve. 7 parameters need at least 3 views for the
# count alone (18 residuals), but below 10 the posture variety is rarely enough.
MIN_VIEWS = 10


def angles_from_raw(
    raw: dict[str, int],
    zero_raw: dict[str, int],
    signs: dict[str, int],
) -> dict[str, float]:
    """Servo counts → joint angles in radians, using stage 4 zeros and stage 2 signs.
    
    Wrap-aware for single-turn absolute encoders. The zeros here are stage 4's
    rough ones; what the fit solves is the correction to wrist_roll's zero.
    """
    return {
        name: servos.raw_to_rad(value, zero_raw[name], signs.get(name, 1))
        for name, value in raw.items()
        if name in zero_raw
    }


def pose_errors(
    p: np.ndarray,
    sim,
    arm: str,
    reported_angles: list[dict],
    observed_poses: list[np.ndarray],
    rough_roll_zero_rad: float,
    T_W_B: np.ndarray,
    T_B_A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-view translation error (metres) and rotation error (radians).
    
    Reported separately so they can be weighted differently in the fit and
    presented in readable units to the operator.
    """
    trans_errs = []
    rot_errs = []
    
    for angles_dict, obs_pose in zip(reported_angles, observed_poses):
        # Per-view pose error (6-vector in se3), in readable units.
        r = wrist_model.residuals(
            p, sim, arm, [angles_dict], [obs_pose],
            rough_roll_zero_rad, T_W_B, T_B_A
        )
        # r is [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        rot_errs.append(np.linalg.norm(r[:3]))
        trans_errs.append(np.linalg.norm(r[3:]))
    
    return np.array(trans_errs), np.array(rot_errs)


def _optical_z_at_zero(
    p_vec: np.ndarray, sim, arm: str,
    T_W_B: np.ndarray, T_B_A: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Optical axis Z (chassis frame) at the XML-zero configuration.

    "XML zero" means every joint's TRUE angle is 0 -- the kinematic zero pose of
    the model, where the operator observed the optical axis to be horizontal. This
    is NOT "motor reading = roll_zero"; it is the physical configuration wrist_roll=0.

    In this configuration the optical axis direction is
        R_W_B @ R_B_A @ R_base_FixedJaw(0) @ T_wrist_cam[:,2]
    which depends on the mount T_wrist_cam. Because the gauge transformation rotates
    the mount (R_roll(-delta) @ T_wrist_cam), this direction sweeps a cone as we slide
    the gauge -- unlike the pose at "motor = roll_zero", which is gauge-invariant. So
    THIS is the quantity that lets the horizontal constraint pin the gauge.

    Returns (z_component, optical_axis_chassis).
    """
    T_wrist_cam, _roll_zero = wrist_model.unpack(p_vec, arm)
    joint_names = wrist_model.ARM_JOINT_NAMES[arm]
    angles = {jn: 0.0 for jn in joint_names}  # XML zero: all true angles = 0
    T_W_cam = wrist_model.camera_in_world(sim, arm, angles, T_wrist_cam, T_W_B, T_B_A)
    optical_world = T_W_cam[:3, 2]
    T_B_W = se3.invert(np.asarray(T_W_B, float))
    optical_chassis = T_B_W[:3, :3] @ optical_world
    return float(optical_chassis[2]), optical_chassis


def _roll_about_axis(delta: float) -> np.ndarray:
    """4x4 rotation by `delta` about the wrist_roll axis through the Fixed_Jaw origin.

    The wrist_roll axis in the Fixed_Jaw frame is wrist_model.ROLL_AXIS_IN_WRIST.
    """
    screw = np.concatenate([wrist_model.ROLL_AXIS_IN_WRIST * delta, np.zeros(3)])
    return se3.exp_se3(screw)


def _apply_gauge(p_vec: np.ndarray, arm: str, delta: float) -> np.ndarray:
    """Exact gauge transformation by `delta` (radians).

    The gauge orbit that leaves every observed-view prediction invariant is:
        roll_zero      -> roll_zero + delta
        T_wrist_cam    -> R_roll(-delta) @ T_wrist_cam
    where R_roll is rotation about the wrist_roll axis. This is exact for any delta,
    unlike a linear step along the infinitesimal gauge vector. Shifting roll_zero by
    delta adds R_roll(delta) into every FK; the mount's R_roll(-delta) cancels it, so
    predictions are unchanged. But the optical axis AT ZERO rotates by -delta about
    the roll axis, which is what lets us satisfy the horizontal constraint.
    """
    T_wrist_cam, roll_zero = wrist_model.unpack(p_vec, arm)
    T_new = _roll_about_axis(-delta) @ T_wrist_cam
    # Re-express as a local perturbation of nominal so pack/unpack round-trips.
    T_nom = wrist_model.nominal_mount(arm)
    xi_new = se3.log_se3(se3.invert(T_nom) @ T_new)
    return wrist_model.pack(xi_new, roll_zero + delta)


def _resolve_gauge_horizontal(
    p_start: np.ndarray, sim, arm: str,
    T_W_B: np.ndarray, T_B_A: np.ndarray,
) -> np.ndarray:
    """Slide along the gauge orbit until the optical axis at zero is horizontal.

    Applying the exact gauge transformation by delta leaves every observed-view
    prediction invariant, so it costs nothing in fit error. It rotates the optical
    axis at zero about the wrist_roll axis, so as delta sweeps a full turn the axis
    traces a cone and crosses the chassis XY plane (Z=0) at TWO points 180° apart.
    We pick the crossing whose optical axis lands in the operator's stated quadrant:
      - left_arm:  quadrant III (X<0, Y<0)
      - right_arm: quadrant I  (X>0, Y>0)
    """
    def z_at(delta: float) -> tuple[float, np.ndarray]:
        return _optical_z_at_zero(_apply_gauge(p_start, arm, delta),
                                  sim, arm, T_W_B, T_B_A)

    # Scan a full turn of the gauge parameter to find sign changes in Z.
    deltas = np.linspace(-np.pi, np.pi, 361)
    zs = np.array([z_at(d)[0] for d in deltas])

    crossings = []
    for i in range(len(deltas) - 1):
        if zs[i] == 0.0 or (zs[i] < 0) != (zs[i + 1] < 0):
            lo, hi = deltas[i], deltas[i + 1]
            zlo = zs[i]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                zmid, _ = z_at(mid)
                if (zmid < 0) == (zlo < 0):
                    lo, zlo = mid, zmid
                else:
                    hi = mid
            d_cross = 0.5 * (lo + hi)
            _, optical = z_at(d_cross)
            crossings.append((d_cross, optical))

    if not crossings:
        return p_start

    def in_quadrant(optical: np.ndarray) -> bool:
        if arm == "left_arm":
            return optical[0] < 0 and optical[1] < 0  # quadrant III
        else:
            return optical[0] > 0 and optical[1] > 0  # quadrant I

    for d_cross, optical in crossings:
        if in_quadrant(optical):
            return _apply_gauge(p_start, arm, d_cross)

    # Fallback: no crossing in the ideal quadrant. Match the nominal direction.
    _, optical_nom = _optical_z_at_zero(
        wrist_model.pack(np.zeros(6), 0.0), sim, arm, T_W_B, T_B_A
    )
    best = max(crossings, key=lambda c: np.dot(c[1], optical_nom))
    return _apply_gauge(p_start, arm, best[0])


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
    """Solve one arm's camera mount and wrist_roll zero from board views.
    
    `reported_angles` are joint angles already converted from counts via stage 5's
    zeros (shoulder/elbow/wrist_pitch) and stage 4's rough zero (wrist_roll).
    The solved wrist_roll_zero is a correction to that rough zero.
    
    `observed_poses` are T_cam_board from PnP (= T_cam_world since W = board).
    """
    from . import solver as solver_mod
    
    n = len(reported_angles)
    if n < MIN_VIEWS:
        return None
    
    fit_idx, hold_idx = solver_mod.split_holdout(
        n, fraction=holdout_fraction, seed=0, minimum=3
    )
    
    guess = wrist_model.initial_guess(arm)
    
    # Weight rotation and translation separately. At the working distance (~0.3-0.6m),
    # 1 degree of camera rotation moves the image about as much as several mm of
    # translation. ROT_SCALE converts radians to an equivalent translation scale.
    ROT_SCALE = 0.05  # 1 rad ≈ 50mm of effective motion
    
    # STEP 1: Solve the 6 observable parameters with pose residuals only.
    # The gauge freedom (mount roll about wrist_roll axis <-> wrist_roll_zero) is
    # projected out so this is well conditioned. This determines everything the
    # camera images can see -- but leaves the gauge (where "zero" sits) undetermined.
    def residual(free):
        p_vec = wrist_model.with_free(free, arm, guess)
        r = wrist_model.residuals(
            p_vec, sim, arm,
            [reported_angles[i] for i in fit_idx],
            [observed_poses[i] for i in fit_idx],
            rough_roll_zero_rad, T_W_B, T_B_A
        )
        n_views = len(fit_idx)
        weights = np.tile([ROT_SCALE, ROT_SCALE, ROT_SCALE, 1.0, 1.0, 1.0], n_views)
        return r * weights
    
    out = least_squares(
        residual, wrist_model.free_of(guess, arm), method="trf",
        loss="soft_l1", f_scale=POSE_F_SCALE_M, x_scale="jac",
        xtol=1e-14, ftol=1e-14, max_nfev=1200
    )
    
    p_gauge_fixed = wrist_model.with_free(out.x, arm, guess)
    
    # STEP 2: Slide along the gauge direction to satisfy the physical constraint
    # "at wrist_roll = 0 (XML zero), the optical axis is horizontal in chassis XY".
    # Sliding along the gauge leaves every observed-view prediction unchanged (poses
    # are gauge-invariant), but rotates the optical-axis-at-zero about the wrist_roll
    # axis, so it crosses horizontal at a well-defined offset. This is what breaks
    # the gauge using the operator's measurement, not an arbitrary XML convention.
    p_solved = _resolve_gauge_horizontal(
        p_gauge_fixed, sim, arm, T_W_B, T_B_A
    )
    T_wrist_cam, roll_zero_correction = wrist_model.unpack(p_solved, arm)
    
    # Compute errors in readable units
    fit_trans, fit_rot = pose_errors(
        p_solved, sim, arm,
        [reported_angles[i] for i in fit_idx],
        [observed_poses[i] for i in fit_idx],
        rough_roll_zero_rad, T_W_B, T_B_A
    )
    hold_trans, hold_rot = pose_errors(
        p_solved, sim, arm,
        [reported_angles[i] for i in hold_idx],
        [observed_poses[i] for i in hold_idx],
        rough_roll_zero_rad, T_W_B, T_B_A
    )
    
    # Compute wrist_roll sweep
    roll_motor = wrist_model.ARM_JOINT_NAMES[arm][4]
    rolls = [a[roll_motor] for a in reported_angles if roll_motor in a]
    roll_sweep = float(np.rad2deg(max(rolls) - min(rolls))) if rolls else 0.0
    
    J = out.jac
    sv = np.linalg.svd(J, compute_uv=False) if J is not None and J.size else None
    condition = float(sv[0] / sv[-1]) if sv is not None and sv[-1] > 0 else float("inf")
    
    mount = se3.log_se3(T_wrist_cam)
    
    # Verify horizontal constraint at XML zero for diagnostic
    z_xml_zero, optical_xml_zero = _optical_z_at_zero(p_solved, sim, arm, T_W_B, T_B_A)
    
    return {
        "arm": arm,
        "params": p_solved.tolist(),
        "T_wrist_cam": T_wrist_cam.tolist(),
        "mount_translation_mm": (T_wrist_cam[:3, 3] * 1000).tolist(),
        "mount_rotation_deg": np.rad2deg(mount[:3]).tolist(),
        "wrist_roll_zero_correction_rad": float(roll_zero_correction),
        "wrist_roll_zero_correction_deg": float(np.rad2deg(roll_zero_correction)),
        "optical_axis_at_xml_zero": optical_xml_zero.tolist(),
        "optical_z_at_xml_zero": float(z_xml_zero),
        "n_views_total": int(n),
        "n_views_fit": int(len(fit_idx)),
        "n_views_holdout": int(len(hold_idx)),
        "roll_sweep_deg": roll_sweep,
        "fit_trans_rms_mm": float(np.sqrt(np.mean(fit_trans ** 2)) * 1000),
        "fit_trans_max_mm": float(fit_trans.max() * 1000),
        "fit_rot_rms_deg": float(np.rad2deg(np.sqrt(np.mean(fit_rot ** 2)))),
        "fit_rot_max_deg": float(np.rad2deg(fit_rot.max())),
        "holdout_trans_rms_mm": float(np.sqrt(np.mean(hold_trans ** 2)) * 1000),
        "holdout_trans_max_mm": float(hold_trans.max() * 1000),
        "holdout_rot_rms_deg": float(np.rad2deg(np.sqrt(np.mean(hold_rot ** 2)))),
        "holdout_rot_max_deg": float(np.rad2deg(hold_rot.max())),
        "condition_number": condition,
        "converged": bool(out.success),
    }


def grade(result: dict | None) -> list[gates.GateResult]:
    """Acceptance checks for one arm's wrist camera calibration."""
    checks: list[gates.GateResult] = []
    
    if result is None:
        checks.append(gates.GateResult(
            "fit", False,
            detail=f"fewer than {MIN_VIEWS} views; cannot solve"
        ))
        return checks
    
    checks.append(gates.lower_bound(
        "view count", result["n_views_total"], gates.WRIST_MIN_VIEWS,
        warn_at=15  # Comfortable margin for a 25% holdout
    ))
    
    checks.append(gates.lower_bound(
        "wrist_roll sweep", result["roll_sweep_deg"],
        gates.WRIST_MIN_ROLL_SWEEP_DEG, " deg",
        detail="roll must sweep widely to separate its zero from camera mount rotation"
    ))
    
    checks.append(gates.upper_bound(
        "holdout translation error", result["holdout_trans_rms_mm"],
        gates.WRIST_RESIDUAL_MAX_MM, " mm",
        warn_at=gates.WRIST_RESIDUAL_GOOD_MM
    ))
    
    checks.append(gates.upper_bound(
        "holdout rotation error", result["holdout_rot_rms_deg"],
        gates.WRIST_ROT_MAX_DEG, " deg",
        warn_at=gates.WRIST_ROT_GOOD_DEG
    ))
    
    checks.append(gates.upper_bound(
        "worst holdout translation", result["holdout_trans_max_mm"],
        gates.WRIST_RESIDUAL_MAX_MM * 2.5, " mm",
        detail="one bad view usually means the board was misdetected or moved"
    ))
    
    ratio = (result["holdout_trans_rms_mm"] / result["fit_trans_rms_mm"]
             if result["fit_trans_rms_mm"] > 0 else float("inf"))
    checks.append(gates.upper_bound(
        "holdout / fit ratio", ratio, 3.0,
        detail="a large gap means the arm poses were too alike, or PnP was noisy"
    ))
    
    checks.append(gates.upper_bound(
        "condition number", result["condition_number"],
        gates.MAX_CONDITION_NUMBER,
        detail="with sufficient roll sweep and view variety this should be well conditioned"
    ))
    
    if not result["converged"]:
        checks.append(gates.GateResult(
            "convergence", False, detail="the optimizer stopped early"
        ))
    
    return checks
