"""Stage 2-3 solving: fit T_W^B and the head camera mount from board views.

Kept apart from the interface so it can be driven by synthetic data in tests.

The world frame is the board frame, so PnP's T_cam_board IS the measurement of
T_cam_world. That is the entire reason for defining W that way, and it is why the
old stage 2 produced nothing worth storing.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import gates, head_model, se3, solver

# Where the tilt joint sits in the base frame, for reporting the camera mount as
# an offset from it rather than as an absolute height.
TILT_JOINT = head_model.PAN_ORIGIN + head_model.TILT_OFFSET


def pose_spread(pans: np.ndarray, tilts: np.ndarray) -> dict:
    """Total sweep of each joint, in degrees.

    Total rather than per-side: the board's own width eats asymmetrically into
    the budget, so a real capture is rarely centred.
    """
    pans = np.rad2deg(np.asarray(pans, float))
    tilts = np.rad2deg(np.asarray(tilts, float))
    if pans.size == 0:
        return {"pan_sweep_deg": 0.0, "tilt_sweep_deg": 0.0,
                "pan_range_deg": [0.0, 0.0], "tilt_range_deg": [0.0, 0.0]}
    return {
        "pan_sweep_deg": float(pans.max() - pans.min()),
        "tilt_sweep_deg": float(tilts.max() - tilts.min()),
        "pan_range_deg": [float(pans.min()), float(pans.max())],
        "tilt_range_deg": [float(tilts.min()), float(tilts.max())],
    }


def position_errors_mm(p: np.ndarray, pans, tilts, observed,
                       senses=None) -> np.ndarray:
    """Per-view camera position error, which is the number an operator can judge.

    The optimiser works on a 6-vector per view mixing radians and metres; that is
    right for the fit but meaningless to read. This reports millimetres.
    """
    T_W_B, T_tilt_cam = head_model.unpack(p)
    out = []
    for pan, tilt, obs in zip(pans, tilts, observed):
        pred = head_model.T_cam_world(T_W_B, float(pan), float(tilt), T_tilt_cam,
                                      senses=senses)
        d = np.linalg.inv(np.asarray(obs, float)) @ pred
        out.append(np.linalg.norm(d[:3, 3]) * 1000.0)
    return np.asarray(out)


def rotation_errors_deg(p: np.ndarray, pans, tilts, observed,
                        senses=None) -> np.ndarray:
    T_W_B, T_tilt_cam = head_model.unpack(p)
    out = []
    for pan, tilt, obs in zip(pans, tilts, observed):
        pred = head_model.T_cam_world(T_W_B, float(pan), float(tilt), T_tilt_cam,
                                      senses=senses)
        d = np.linalg.inv(np.asarray(obs, float)) @ pred
        out.append(np.rad2deg(np.linalg.norm(se3.log_so3(d[:3, :3]))))
    return np.asarray(out)


def fit(pans, tilts, observed, holdout_fraction: float = 0.25,
        senses: tuple[float, float] | None = None) -> dict | None:
    """Solve the 12 parameters, scoring on views held out of the fit.

    Rotation and translation residuals are weighted so that a radian and a metre
    do not compete on raw magnitude: at the working distance, one degree of camera
    rotation moves the image about as much as several millimetres of translation.
    Without this the fit quietly favours whichever block has the larger numbers.

    `senses` are the joint senses from stage 2. They are an input, not something
    this fit can determine: both pan senses reach the same residual, so passing
    the wrong one yields a confident, well-conditioned, mirrored answer.

    Multi-start with adaptive thoroughness: tries several initial guesses to
    avoid the degenerate basin where the camera ends up ~0.8 m from the tilt
    joint. Starts with 12 quick guesses (typical case), escalates to 48 medium
    guesses if no candidate passes the quality gates, and finally tries all views
    exhaustively if needed. A near-nominal lever alone is not a successful solve.
    """
    pans = np.asarray(pans, float)
    tilts = np.asarray(tilts, float)
    n = len(observed)
    if n < 6:
        return None

    fit_idx, hold_idx = solver.split_holdout(
        n, fraction=holdout_fraction, seed=0, minimum=3)

    # 1 rad of rotation is comparable to ROT_SCALE metres of translation.
    ROT_SCALE = 0.05
    weights = np.tile(np.array([ROT_SCALE] * 3 + [1.0] * 3), len(fit_idx))

    def residual(p):
        r = head_model.residuals(p, pans[fit_idx], tilts[fit_idx],
                                 [observed[i] for i in fit_idx], senses=senses)
        return r * weights

    # Adaptive multi-start: a physically plausible basin is not enough to stop.
    # The candidate must also clear every solution-dependent acceptance gate.
    best_out = None
    best_cost = float("inf")
    fallback_out = None
    fallback_cost = float("inf")
    nominal_lever = np.linalg.norm(head_model.CAM_NOMINAL) * 1000  # mm
    fit_views = [observed[k] for k in fit_idx]
    hold_views = [observed[k] for k in hold_idx]

    for thoroughness in ["quick", "medium", "exhaustive"]:
        guesses = head_model.initial_guesses(pans, tilts, observed, senses=senses,
                                             thoroughness=thoroughness)
        if not guesses:
            continue

        found_acceptable = False
        for guess in guesses:
            out = least_squares(residual, guess, method="trf", loss="soft_l1",
                                f_scale=0.002, x_scale="jac",
                                xtol=1e-14, ftol=1e-14, max_nfev=800)
            if out.cost < fallback_cost:
                fallback_cost = out.cost
                fallback_out = out

            _, T_tilt_cam = head_model.unpack(out.x)
            lever = np.linalg.norm(T_tilt_cam[:3, 3] - TILT_JOINT) * 1000
            lever_ok = abs(lever - nominal_lever) < nominal_lever * 0.5
            if lever_ok and out.cost < best_cost:
                best_cost = out.cost
                best_out = out

            fit_err = position_errors_mm(out.x, pans[fit_idx], tilts[fit_idx],
                                         fit_views, senses)
            hold_err = position_errors_mm(out.x, pans[hold_idx], tilts[hold_idx],
                                          hold_views, senses)
            fit_rms = float(np.sqrt(np.mean(fit_err ** 2)))
            hold_rms = float(np.sqrt(np.mean(hold_err ** 2)))
            ratio = hold_rms / fit_rms if fit_rms > 0 else float("inf")
            J = out.jac
            sv = np.linalg.svd(J, compute_uv=False) if J is not None and J.size else None
            condition = (float(sv[0] / sv[-1])
                         if sv is not None and sv[-1] > 0 else float("inf"))

            residuals_ok = (
                hold_rms <= gates.HEAD_RESIDUAL_MAX_MM
                and float(hold_err.max()) <= gates.HEAD_RESIDUAL_MAX_MM * 2
                and ratio <= 3.0
            )
            if lever_ok and residuals_ok and condition <= gates.MAX_CONDITION_NUMBER \
                    and out.success:
                best_out = out
                found_acceptable = True
                break

        if found_acceptable:
            break
        if thoroughness != "exhaustive":
            print(f"  [{thoroughness}] No candidate passed the quality gates; "
                  "expanding initial guesses...")

    out = best_out if best_out is not None else fallback_out

    fit_err = position_errors_mm(out.x, pans[fit_idx], tilts[fit_idx],
                                 [observed[i] for i in fit_idx], senses)
    hold_err = position_errors_mm(out.x, pans[hold_idx], tilts[hold_idx],
                                  [observed[i] for i in hold_idx], senses)
    fit_rot = rotation_errors_deg(out.x, pans[fit_idx], tilts[fit_idx],
                                  [observed[i] for i in fit_idx], senses)
    hold_rot = rotation_errors_deg(out.x, pans[hold_idx], tilts[hold_idx],
                                   [observed[i] for i in hold_idx], senses)

    T_W_B, T_tilt_cam = head_model.unpack(out.x)

    # Conditioning of the full set, so a direction that did not get pinned down
    # shows up here rather than hiding behind a small residual.
    J = out.jac
    sv = np.linalg.svd(J, compute_uv=False) if J is not None and J.size else None
    condition = float(sv[0] / sv[-1]) if sv is not None and sv[-1] > 0 else float("inf")

    return {
        "params": out.x.tolist(),
        "T_W_B": T_W_B.tolist(),
        "T_tilt_cam": T_tilt_cam.tolist(),
        # T_tilt_cam is expressed in the base frame, so its translation is where
        # the camera sits at the zero posture -- roughly 0.77 m up. What can be
        # checked against the XML is the offset from the tilt joint, which is the
        # lever arm the whole head calibration turns on.
        # Recorded because the result is only meaningful in these signs, and a
        # mirrored solve is otherwise indistinguishable from a correct one.
        "senses": [float(s) for s in (senses if senses is not None
                                      else (head_model.PAN_SENSE,
                                            head_model.TILT_SENSE))],
        "camera_in_base_mm": (T_tilt_cam[:3, 3] * 1000).tolist(),
        "camera_from_tilt_joint_mm": ((T_tilt_cam[:3, 3] - TILT_JOINT) * 1000).tolist(),
        "lever_arm_mm": float(np.linalg.norm(T_tilt_cam[:3, 3] - TILT_JOINT) * 1000),
        "lever_arm_nominal_mm": float(np.linalg.norm(head_model.CAM_NOMINAL) * 1000),
        "n_views_total": int(n),
        "n_views_fit": int(len(fit_idx)),
        "n_views_holdout": int(len(hold_idx)),
        "fit_rms_mm": float(np.sqrt(np.mean(fit_err ** 2))),
        "fit_max_mm": float(fit_err.max()),
        "holdout_rms_mm": float(np.sqrt(np.mean(hold_err ** 2))),
        "holdout_max_mm": float(hold_err.max()),
        "fit_rms_deg": float(np.sqrt(np.mean(fit_rot ** 2))),
        "holdout_rms_deg": float(np.sqrt(np.mean(hold_rot ** 2))),
        "condition_number": condition,
        "converged": bool(out.success),
        **pose_spread(pans, tilts),
    }


def grade(result: dict | None) -> list[gates.GateResult]:
    """Acceptance checks for a head fit."""
    checks: list[gates.GateResult] = []
    if result is None:
        checks.append(gates.GateResult("fit", False,
                                       detail="too few views to solve"))
        return checks

    checks.append(gates.lower_bound(
        "view count", result["n_views_total"], gates.HEAD_MIN_VIEWS,
        warn_at=gates.HEAD_MIN_VIEWS + 10))

    checks.append(gates.lower_bound(
        "pan sweep", result["pan_sweep_deg"], gates.HEAD_PAN_SWEEP_MIN_DEG,
        " deg", warn_at=gates.HEAD_PAN_SWEEP_GOOD_DEG,
        detail="a narrow sweep leaves the vertical axis poorly determined"))

    checks.append(gates.upper_bound(
        "holdout error", result["holdout_rms_mm"], gates.HEAD_RESIDUAL_MAX_MM,
        " mm", warn_at=gates.HEAD_RESIDUAL_GOOD_MM))

    checks.append(gates.upper_bound(
        "worst holdout view", result["holdout_max_mm"],
        gates.HEAD_RESIDUAL_MAX_MM * 2, " mm",
        detail="one bad view usually means a smeared frame or a misdetection"))

    ratio = (result["holdout_rms_mm"] / result["fit_rms_mm"]
             if result["fit_rms_mm"] > 0 else float("inf"))
    checks.append(gates.upper_bound(
        "holdout / fit ratio", ratio, 3.0,
        detail="a large gap means the head postures were too alike"))

    checks.append(gates.upper_bound(
        "condition number", result["condition_number"],
        gates.MAX_CONDITION_NUMBER,
        detail="an ill-conditioned fit has a direction that data never pinned down"))

    if not result["converged"]:
        checks.append(gates.GateResult("convergence", False,
                                       detail="the optimiser stopped early"))
    return checks
