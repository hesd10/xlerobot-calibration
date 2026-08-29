"""Acceptance thresholds for each stage.

Errors compound down the chain: a bad K distorts every solved pose, which the arm
zeros then absorb as a fake offset. Each stage therefore has to clear a numeric
gate before the runner will let the operator continue, and every gate is checked
on data that took no part in the fit wherever that is possible.

Thresholds are deliberately explicit rather than buried in each stage script, so
they can be reviewed in one place and tightened as the hardware is understood.
"""

from __future__ import annotations

from dataclasses import dataclass

# Intrinsics. These cameras are cheap 640x480 modules, so sub-0.3 px is a
# realistic target and anything above 1 px means the model does not fit.
INTRINSICS_RMS_GOOD_PX = 0.35
INTRINSICS_RMS_MAX_PX = 1.0
INTRINSICS_MIN_VIEWS = 15
INTRINSICS_MIN_HOLDOUT = 5
# Holdout error should not exceed fit error by much; a big gap means overfitting
# to a narrow set of viewpoints.
INTRINSICS_HOLDOUT_RATIO_MAX = 2.0
# Corner spread, as a fraction of the frame, required for the distortion terms to
# be constrained rather than extrapolated.
INTRINSICS_MIN_COVERAGE = 0.55

# Board pose quality, used whenever PnP is run. Six corners is the algebraic
# minimum, but not a useful one: measured on the stage 2-3 capture, views with
# fewer than twelve corners averaged 8.7 mm of pose error against 3.9 mm for
# views with twenty or more, and every one of the worst views was corner-starved.
# A few corners in one corner of the frame pin down orientation poorly, and the
# reprojection error stays small because there is little left to disagree with.
PNP_MIN_CORNERS = 12
PNP_MAX_REPROJ_PX = 1.5

# What a view should have for the head stages, where a pose error feeds straight
# into the solved geometry. Below this a view is accepted but the operator is told
# to fill more of the frame with the board.
PNP_GOOD_CORNERS = 20

# Stage 2/3. The reference view lets later stages notice the base has moved.
BASE_SHIFT_WARN_MM = 2.0
BASE_SHIFT_FAIL_MM = 5.0

# Head axis. Derived from a sensitivity study: with 0.5 mm PnP noise, a 60 deg
# total pan sweep pins the vertical axis to roughly 1 mm, 30 deg only to about
# 10 mm. These are TOTAL sweeps, max minus min, not per-side amplitudes: the
# board's own width eats into the budget, so what a real setup allows is usually
# asymmetric. On this robot, 86 deg of measured fov and a 200 mm board at 60 cm
# leave about +/-28 deg, so 30 deg total is comfortably reachable.
HEAD_PAN_SWEEP_MIN_DEG = 30.0
HEAD_PAN_SWEEP_GOOD_DEG = 50.0
HEAD_MIN_VIEWS = 20
HEAD_RESIDUAL_GOOD_MM = 2.0
HEAD_RESIDUAL_MAX_MM = 6.0

# Stage 4. A joint must move enough that its direction is unambiguous against
# encoder noise of a count or two.
DIRECTION_MIN_TRAVEL_COUNTS = 40

# Stage 5. Manual touching is the accuracy floor of the whole procedure; a human
# aligning a jaw tip to a printed corner is doing well to hit half a millimetre.
TOUCH_MIN_POINTS = 25
TOUCH_MIN_HOLDOUT = 6
TOUCH_RESIDUAL_GOOD_MM = 2.0
TOUCH_RESIDUAL_MAX_MM = 5.0
TOUCH_MIN_POSTURE_SPREAD_DEG = 40.0

# Stage 6. Wrist roll must sweep widely or its zero cannot be told apart from the
# camera mount rotation. 10 views is the hard minimum (matches the solver's
# MIN_VIEWS); a warning nudges toward 15+ for a comfortable holdout margin.
WRIST_MIN_VIEWS = 10
WRIST_MIN_ROLL_SWEEP_DEG = 90.0
WRIST_RESIDUAL_GOOD_MM = 3.0
WRIST_RESIDUAL_MAX_MM = 8.0
WRIST_ROT_GOOD_DEG = 1.0
WRIST_ROT_MAX_DEG = 3.0

# Stage 7. Jaw angle, not width, per the revolute jaw joint in the model.
GRIPPER_MIN_SAMPLES = 6
GRIPPER_RESIDUAL_MAX_DEG = 2.0

# Stage 8, end to end.
FINAL_TCP_GOOD_MM = 3.0
FINAL_TCP_MAX_MM = 8.0

# Solver conditioning. A parameter direction this weak is effectively
# unobservable and should be fixed by convention instead of fitted.
MIN_SINGULAR_VALUE = 1e-6
MAX_CONDITION_NUMBER = 1e6


@dataclass
class GateResult:
    """Outcome of one acceptance check."""

    name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    unit: str = ""
    detail: str = ""
    warning: bool = False
    # "max" when the requirement is value <= threshold, "min" for value >=.
    direction: str = "max"

    def line(self) -> str:
        if self.passed and self.warning:
            mark = "WARN"
        else:
            mark = "OK  " if self.passed else "FAIL"
        text = f"  [{mark}] {self.name}"
        if self.value is not None:
            text += f": {self.value:.3f}{self.unit}"
            if self.threshold is not None:
                need = "max" if self.direction == "max" else "min"
                text += f" ({need} {self.threshold:.3f}{self.unit})"
        # Only explain when there is something to explain. A passing gate that
        # trails the reason it exists ("a large gap means the views were too
        # alike") reads as a complaint about a result that was in fact fine.
        if self.detail and (not self.passed or self.warning):
            text += f"  {self.detail}"
        return text


def upper_bound(name: str, value: float | None, limit: float, unit: str = "",
                warn_at: float | None = None, detail: str = "") -> GateResult:
    """Gate that requires value <= limit, optionally warning earlier."""
    if value is None:
        return GateResult(name, False, None, limit, unit,
                          detail or "not measured", direction="max")
    passed = value <= limit
    warning = passed and warn_at is not None and value > warn_at
    if warning and not detail:
        detail = f"above the comfortable {warn_at:g}{unit}"
    return GateResult(name, passed, value, limit, unit, detail, warning, "max")


def lower_bound(name: str, value: float | None, limit: float, unit: str = "",
                warn_at: float | None = None, detail: str = "") -> GateResult:
    """Gate that requires value >= limit, optionally warning earlier."""
    if value is None:
        return GateResult(name, False, None, limit, unit,
                          detail or "not measured", direction="min")
    passed = value >= limit
    warning = passed and warn_at is not None and value < warn_at
    if warning and not detail:
        detail = f"below the comfortable {warn_at:g}{unit}"
    return GateResult(name, passed, value, limit, unit, detail, warning, "min")


def summarise(results: list[GateResult]) -> tuple[bool, str]:
    """Render all gates and report whether every one passed."""
    lines = [r.line() for r in results]
    failed = [r for r in results if not r.passed]
    warned = [r for r in results if r.passed and r.warning]
    if failed:
        lines.append(f"  {len(failed)} gate(s) failed: "
                     + ", ".join(r.name for r in failed))
    elif warned:
        lines.append(f"  all gates passed, {len(warned)} with warnings")
    else:
        lines.append("  all gates passed")
    return not failed, "\n".join(lines)


def coverage_fraction(corners, width: int, height: int, bins: int = 6) -> float:
    """Fraction of image cells containing at least one detected corner.

    Distortion coefficients are only constrained where corners were actually
    observed. A tight central cluster leaves the edges to extrapolation, which is
    exactly where distortion matters most.
    """
    import numpy as np

    pts = np.asarray(corners, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return 0.0
    gx = np.clip((pts[:, 0] / width * bins).astype(int), 0, bins - 1)
    gy = np.clip((pts[:, 1] / height * bins).astype(int), 0, bins - 1)
    return len(set(zip(gx.tolist(), gy.tolist()))) / float(bins * bins)
