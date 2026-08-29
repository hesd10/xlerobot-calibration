"""Stage 5 solving: arm zeros and mounting from touched board corners.

The measurement
---------------
The operator drives a fixed point on the gripper jaw onto a known ChArUco corner
and the raw counts are recorded. The corner's position is known in the world frame
(the board frame, by stage 3's definition), and stage 3 solved T_W^B, so each touch
gives one known point in the base frame that a known joint configuration must reach.

Three residual components per touch, thirteen parameters per arm: six for the arm
mounting, four joint zeros, three for the touch point on the jaw. Twelve are
observable; the mount's yaw is fixed by convention, see `core/arm_model.py`.

What this stage does not solve
------------------------------
Wrist roll's zero. Rolling the wrist and sliding the touch point along the roll
axis are the same motion, so contact alone cannot separate them. The Fusion path
resolves wrist roll in Stage 5b; the gripper uses its Stage 4 closed zero and the
standard absolute-encoder scale.

Reading the result
------------------
Holdout error is the number to trust. Fit error on touches that shaped the fit will
look good even when the arm mounting has absorbed an error that belongs to a zero,
and with thirteen parameters there is room for that to happen.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import arm_model, gates, se3, servos

# Robust loss transition, in metres. Manual touching lands within a couple of
# millimetres when it goes well; beyond about 8 mm it was a bad touch, not noise.
TOUCH_F_SCALE = 0.008

# Minimum touches to attempt a solve at all. Thirteen parameters need at least five
# touches for the count alone; below ten the geometry is never varied enough.
MIN_TOUCHES = 10


def angles_from_raw(raw: dict[str, int], zero_raw: dict[str, int],
                    signs: dict[str, int]) -> dict[str, float]:
    """Servo counts -> joint angles in radians, using stage 4 zeros and stage 2 signs.

    Wrap-aware, so a joint sitting across the 4095/0 seam reads correctly. The zeros
    here are stage 4's rough ones; what the fit solves is the correction to them.
    """
    return {name: servos.raw_to_rad(value, zero_raw[name], signs.get(name, 1))
            for name, value in raw.items() if name in zero_raw}


def errors_mm(p: np.ndarray, sim, arm: str, reported: list[dict],
              targets: list[np.ndarray]) -> np.ndarray:
    """Per-touch distance error in millimetres, the number an operator can judge."""
    r = arm_model.residuals(p, sim, arm, reported, targets)
    return np.linalg.norm(r.reshape(-1, 3), axis=1) * 1000.0


def fit(sim, arm: str, reported: list[dict], targets: list[np.ndarray],
        zeros_guess: dict | None = None,
        holdout_fraction: float = 0.25) -> dict | None:
    """Solve one arm's mounting, four zeros and touch point from contacts.

    `reported` are joint angles already converted from counts via stage 4's rough
    zeros, so the solved zeros are corrections to those. `targets` are the touched
    corners in the base frame.
    """
    from . import solver as solver_mod

    n = len(reported)
    if n < MIN_TOUCHES:
        return None

    fit_idx, hold_idx = solver_mod.split_holdout(
        n, fraction=holdout_fraction, seed=0, minimum=4)

    guess = arm_model.initial_guess(arm, zeros_guess)

    def residual(free):
        p = arm_model.with_free(guess, free)
        return arm_model.residuals(p, sim, arm,
                                   [reported[i] for i in fit_idx],
                                   [targets[i] for i in fit_idx])

    out = least_squares(residual, arm_model.free_of(guess), method="trf",
                        loss="soft_l1", f_scale=TOUCH_F_SCALE, x_scale="jac",
                        xtol=1e-14, ftol=1e-14, max_nfev=1200)

    p = arm_model.with_free(guess, out.x)
    T_B_A, zeros, touch = arm_model.unpack(p, arm)

    fit_err = errors_mm(p, sim, arm, [reported[i] for i in fit_idx],
                        [targets[i] for i in fit_idx])
    hold_err = errors_mm(p, sim, arm, [reported[i] for i in hold_idx],
                         [targets[i] for i in hold_idx])

    J = out.jac
    sv = np.linalg.svd(J, compute_uv=False) if J is not None and J.size else None
    condition = float(sv[0] / sv[-1]) if sv is not None and sv[-1] > 0 else float("inf")

    mount = se3.log_se3(T_B_A)
    return {
        "arm": arm,
        "params": p.tolist(),
        "T_B_A": T_B_A.tolist(),
        "mount_translation_mm": (T_B_A[:3, 3] * 1000).tolist(),
        "mount_rotation_deg": np.rad2deg(mount[:3]).tolist(),
        "zeros_rad": {k: float(v) for k, v in zeros.items()},
        "zeros_deg": {k: float(np.rad2deg(v)) for k, v in zeros.items()},
        "touch_point_mm": (touch * 1000).tolist(),
        # Held, not solved. Stated in the result so nobody reads the shoulder pan
        # zero as a pure joint property when it also carries the mount's yaw.
        "gauge": {"arm_mount_yaw": 0.0,
                  "absorbed_by": f"{arm}_shoulder_pan zero"},
        "n_touches_total": int(n),
        "n_touches_fit": int(len(fit_idx)),
        "n_touches_holdout": int(len(hold_idx)),
        "fit_rms_mm": float(np.sqrt(np.mean(fit_err ** 2))),
        "fit_max_mm": float(fit_err.max()),
        "holdout_rms_mm": float(np.sqrt(np.mean(hold_err ** 2))),
        "holdout_max_mm": float(hold_err.max()),
        "condition_number": condition,
        "converged": bool(out.success),
        **posture_spread(reported, arm),
    }


def posture_spread(reported: list[dict], arm: str) -> dict:
    """How much each solved joint varied across the touches.

    A zero is only determined by touches that moved its joint. Touching twenty
    corners from one arm posture looks like plenty of data and determines nothing.
    """
    out = {}
    worst = 360.0
    for name in arm_model.joint_names(arm):
        values = [np.rad2deg(a[name]) for a in reported if name in a]
        span = float(max(values) - min(values)) if values else 0.0
        out[f"spread_{name}_deg"] = span
        worst = min(worst, span)
    out["min_joint_spread_deg"] = float(worst if reported else 0.0)
    return out


# Acceptance thresholds. A hand-placed contact is good to a couple of millimetres,
# so a holdout error of 5 mm is a fair pass and 2 mm is a good one.
TOUCH_MAX_MM = 5.0
TOUCH_GOOD_MM = 2.0
MIN_JOINT_SPREAD_DEG = 40.0
GOOD_JOINT_SPREAD_DEG = 70.0


def grade(result: dict | None) -> list[gates.GateResult]:
    """Acceptance checks for one arm's contact fit."""
    checks: list[gates.GateResult] = []
    if result is None:
        checks.append(gates.GateResult(
            "fit", False,
            detail=f"fewer than {MIN_TOUCHES} touches; cannot solve"))
        return checks

    checks.append(gates.lower_bound(
        "touch count", result["n_touches_total"], MIN_TOUCHES,
        warn_at=MIN_TOUCHES + 8))

    checks.append(gates.upper_bound(
        "holdout error", result["holdout_rms_mm"], TOUCH_MAX_MM, " mm",
        warn_at=TOUCH_GOOD_MM))

    checks.append(gates.upper_bound(
        "worst holdout touch", result["holdout_max_mm"], TOUCH_MAX_MM * 2.5, " mm",
        detail="one bad touch usually means the jaw point slipped off the corner"))

    ratio = (result["holdout_rms_mm"] / result["fit_rms_mm"]
             if result["fit_rms_mm"] > 0 else float("inf"))
    checks.append(gates.upper_bound(
        "holdout / fit ratio", ratio, 3.0,
        detail="a large gap means the postures were too alike, so the mounting "
               "absorbed error that belongs to a zero"))

    checks.append(gates.lower_bound(
        "smallest joint spread", result["min_joint_spread_deg"],
        MIN_JOINT_SPREAD_DEG, " deg", warn_at=GOOD_JOINT_SPREAD_DEG,
        detail="a joint that barely moved across the touches has an undetermined "
               "zero, however small the residual"))

    checks.append(gates.upper_bound(
        "condition number", result["condition_number"],
        gates.MAX_CONDITION_NUMBER,
        detail="with the mount yaw held this should be well conditioned; if it "
               "is not, some other direction went undetermined too"))

    if not result["converged"]:
        checks.append(gates.GateResult(
            "convergence", False, detail="the optimiser stopped early"))
    return checks
