"""The one place that says which way the robot faces and which side is which.

Why this module exists
----------------------
The body-frame convention used to live as bare literals scattered across the
stages: `p[1] < 0` here, `optical[0] < 0.0` there, `(-1, -1) if arm ==
"left_arm"` somewhere else. `model_map.FORWARD_AXIS` and `LEFT_AXIS` looked
like the source of truth but had no readers at all, so changing them changed
nothing. That arrangement let a real bug through.

The bug is worth recording, because it shapes this module. A robot used
back-to-front had its base frame solved a half turn from the one the model
assumes. Every gate passed anyway: the checks that could have caught it are
phrased in the base frame, and the base frame is itself defined by where the
operator set the head pan zero. Facing the board makes the head look along
-X *by construction*, flipped or not, so those checks are self-consistent with
a frame that is 180 degrees wrong. They cannot fail, and cannot detect it.

The lesson is the distinction this module draws.

Two kinds of direction test
---------------------------
`FRAME_RELATIVE`
    Phrased purely in the base frame, e.g. "the head optical axis at pan zero
    points along -X". These are true by construction once the operator has set
    a zero, so they confirm the solve is self-consistent and nothing more.
    They are worth keeping as sanity checks, but they can never detect a
    mis-set frame. Do not add new gates of this kind and expect protection.

`MODEL_ANCHORED`
    Compares a solved quantity against a fact about the physical robot that
    holds no matter where the zeros were set, e.g. "the left arm is bolted at
    y < 0". These are the only tests that can catch a wrongly-oriented frame,
    because the model's arm mounts do not move when the operator turns the
    head.

When adding a check, decide which kind it is. If it is frame-relative, do not
rely on it for correctness.

Scope
-----
This module describes the MODEL's layout, which is a property of the XML and
never changes. It deliberately says nothing about how the robot is standing in
the room. A robot used back-to-front still has its left arm bolted at y < 0 in
its own base frame; what changes is the mapping from that base frame to the
world, which is Stage 5's business, not this module's.
"""

from __future__ import annotations

import json

import numpy as np

# The model faces -X and stands up along +Z, so its left hand is toward -Y:
#
#     left = up x forward = z_hat x (-x_hat) = -y_hat
#
# Two independent pieces of geometry back the -X forward direction:
#   - the head camera mounts at xyz 0.025 0 0.03 on head_tilt_link, which
#     rotates into world -X, i.e. the lens protrudes forward
#   - the whole payload leans that way: both arm roots at x = -0.09, head -0.10
#
# model_map.check_consistency() verifies both against the loaded model, so these
# are checked facts rather than assumptions.
FORWARD = np.array([-1.0, 0.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
LEFT = np.cross(UP, FORWARD)

# Bodies closer than this to the sagittal plane count as centred, not sided.
CENTRE_TOL = 0.01

ARMS = ("left_arm", "right_arm")


# ---- how the robot is being used ------------------------------------------
#
# The chassis can be driven back-to-front, so the storage half faces the board
# and the arm flanges are physically rotated 180 degrees to keep the arms facing
# it. Nothing about the MODEL changes: `left_arm` is still the arm bolted toward
# LEFT in the XML. What changes is which physical arm that name lands on.
#
# Everything the operator sees should be in physical terms, because that is what
# they can point at: the arm on their left as they face the robot's working
# side. Everything the solver stores stays in model terms. This is the one place
# that converts between them, so no stage has to reason about it.
NORMAL = "normal"
FLIPPED = "flipped"
MOUNTINGS = (NORMAL, FLIPPED)

# The physical sides, named from the point of view of someone facing the board
# alongside the robot, looking the same way the robot works.
SIDES = ("left", "right")


def check_mounting(mounting: str) -> str:
    """Reject anything that is not a mounting, with the options named."""
    if mounting not in MOUNTINGS:
        raise ValueError(
            f"unknown mounting {mounting!r}, expected one of {MOUNTINGS}")
    return mounting


def named_arm(side: str, mounting: str = NORMAL) -> str:
    """The model's arm name for the arm physically on `side`.

    Used back-to-front, the flanges are turned so the arm the operator sees on
    their left is the one the model calls `right_arm`.
    """
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}, expected one of {SIDES}")
    check_mounting(mounting)
    if mounting == FLIPPED:
        side = "right" if side == "left" else "left"
    return f"{side}_arm"


def physical_side(arm: str, mounting: str = NORMAL) -> str:
    """Which side the operator sees `arm` on. The inverse of named_arm()."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    check_mounting(mounting)
    side = arm.removesuffix("_arm")
    if mounting == FLIPPED:
        side = "right" if side == "left" else "left"
    return side


def named_camera(side: str, mounting: str = NORMAL) -> str:
    """The wrist camera role for the camera physically on `side`.

    The roles are named after the arm they are bolted to, so they turn with the
    arms. Getting this wrong is expensive: the wrist roles feed the intrinsics,
    arm and verify stages, and nothing downstream can detect the swap.
    """
    return f"{named_arm(side, mounting).removesuffix('_arm')}_wrist"


def physical_camera_side(camera: str, mounting: str = NORMAL) -> str:
    """Which side the operator sees a wrist camera on.

    The inverse of named_camera(). The wrist roles are named after the arm they
    are bolted to, so they turn with the flanges: back-to-front, the camera the
    model calls `left_wrist` rides the arm on the operator's right.
    """
    check_mounting(mounting)
    for side in SIDES:
        if named_camera(side, mounting) == camera:
            return side
    raise ValueError(f"unknown wrist camera {camera!r}")


def spoken_camera(camera: str, mounting: str = NORMAL) -> str:
    """How to name a camera to the operator, by the side they can point at.

    The stored role stays the key for every saved file; only the words on
    screen follow the robot. A page that prints the stored role instead has the
    operator checking the picture from the opposite arm, and nothing downstream
    can notice -- the wrist roles feed intrinsics, arms and verify alike.
    """
    if camera == "head":
        return "head camera"
    return f"{physical_camera_side(camera, mounting)} wrist camera"


def camera_order(mounting: str = NORMAL) -> tuple[str, ...]:
    """The camera roles: head first, then the wrists the operator sees left to
    right."""
    return ("head",) + tuple(named_camera(side, mounting) for side in SIDES)


def working_order(mounting: str = NORMAL) -> tuple[str, ...]:
    """The arms in the order the operator should work through them: left, then
    right, by what they can SEE.

    The convention is left first, and it is the operator's left that matters.
    Used back-to-front the flanges are turned, so iterating ARMS in model order
    would start with the arm on their right -- the same instruction reversed.
    Reordering here keeps the habit intact without touching the stored order,
    which stays model-canonical so saved files compare cleanly across runs.
    """
    check_mounting(mounting)
    return tuple(named_arm(side, mounting) for side in SIDES)


def in_working_order(motors, mounting: str = NORMAL) -> list[str]:
    """Re-order joints so the operator meets the physically-left arm first.

    Within an arm the given order is preserved, and joints that belong to no arm
    (the head) keep their place at the end.
    """
    check_mounting(mounting)
    motors = list(motors)
    out: list[str] = []
    for arm in working_order(mounting):
        out.extend(m for m in motors if m.startswith(f"{arm}_"))
    out.extend(m for m in motors if not any(
        m.startswith(f"{arm}_") for arm in ARMS))
    return out


def spoken_joint(motor: str, mounting: str = NORMAL) -> str:
    """How to name a joint to the operator, by the arm they can point at.

    Every instruction the operator acts on has to name the arm in front of them.
    Used back-to-front the flanges are turned, so `left_arm_shoulder_pan` drives
    the arm on their RIGHT; printing the stored name would have them reaching for
    an arm that never moves.

    Only the side they can see is given. They are standing in front of one arm
    and need to know which; adding the stored name asks them to decide which of
    two opposite answers is theirs, which is the confusion this exists to
    remove. Callers that also need the stored name print it separately, on the
    lines the dashboard parses.

    Joints that are not on an arm (the head) are returned unchanged.
    """
    check_mounting(mounting)
    for arm in ARMS:
        if not motor.startswith(f"{arm}_"):
            continue
        side = physical_side(arm, mounting)
        if side == arm.removesuffix("_arm"):
            return motor
        return f"{side} arm {motor[len(arm) + 1:]}"
    return motor


def declared_mounting(results_dir=None) -> str:
    """How the operator says the robot is standing, read from the workspace.

    Declared, never inferred. A back-to-front robot and a normal robot whose
    head pan zero was set half a turn out produce identical measurements, and
    the two want opposite responses: absorb the turn, or refuse and make the
    operator redo the head stage. Only the operator can tell them apart, so
    only the operator gets to say.

    Defaults to normal, which is what an older workspace with no declaration
    was calibrated as.
    """
    from pathlib import Path

    if results_dir is None:
        from core import storage
        results_dir = storage.RESULTS_DIR
    for base in (Path(results_dir).resolve().parent, Path.cwd()):
        state = base / "workflow.json"
        if state.is_file():
            try:
                declared = json.loads(state.read_text()).get("mounting")
            except (ValueError, OSError):
                continue
            if declared:
                return check_mounting(declared)
    return NORMAL


def side_sign(arm: str) -> float:
    """+1 if this arm is bolted toward LEFT, -1 if toward its opposite.

    Signs a quantity measured along the LEFT axis. Prefer this to writing
    `-1 if arm == "left_arm" else 1`, which hides which axis is meant.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    return 1.0 if arm == "left_arm" else -1.0


def lateral(point: np.ndarray) -> float:
    """How far `point` lies toward the robot's left, in metres.

    Positive is left. Reading this instead of `point[1]` keeps the sign
    convention in one place, so the ambiguity of a bare index cannot creep back.
    """
    return float(np.dot(np.asarray(point, float)[:3], LEFT))


def forward_of(point: np.ndarray) -> float:
    """How far `point` lies in front of the robot, in metres."""
    return float(np.dot(np.asarray(point, float)[:3], FORWARD))


def side_of(point: np.ndarray) -> str:
    """Name the side `point` sits on: "left", "right" or "centre"."""
    offset = lateral(point)
    if abs(offset) <= CENTRE_TOL:
        return "centre"
    return "left" if offset > 0.0 else "right"


def is_on_expected_side(arm: str, point: np.ndarray,
                        tolerance: float = CENTRE_TOL) -> bool:
    """Is `point` on the side where `arm` is bolted?

    MODEL_ANCHORED, in the sense above: the mounts are bolted to the chassis, so
    this holds whatever the operator did with the zeros. This is the test that
    catches a base frame solved a half turn out, which no frame-relative check
    can see.
    """
    return lateral(point) * side_sign(arm) > tolerance


def wrong_side_report(roots: dict[str, np.ndarray]) -> str:
    """Explain a wrong-sided arm root, or return "" if both are fine.

    `roots` maps arm name to its solved root position in the base frame.
    """
    wrong = [arm for arm in ARMS
             if arm in roots and not is_on_expected_side(arm, roots[arm])]
    if not wrong:
        return ""

    lines = ["arm roots are on the wrong sides of the base frame."]
    for arm in ARMS:
        if arm not in roots:
            continue
        want = "left" if side_sign(arm) > 0 else "right"
        lines.append(f"  {arm:<10} sits {lateral(roots[arm]) * 1000:+7.1f} mm "
                     f"toward the robot's left, expected the {want}")
    lines += [
        "",
        "The usual cause is a robot used back-to-front while the head pan zero",
        "was set facing the board. Facing the board then means facing",
        "chassis-BACK, so the base frame comes out 180 degrees from the one",
        "model/xlerobot_calib.xml assumes. Stage 6 hides this by moving each arm",
        "mount ~300 mm sideways and each shoulder-pan zero by ~180 degrees,",
        "which cancel, so residuals still look fine.",
        "",
        "The captures are usable. Set the mounting switch at the top of the",
        "dashboard to back-to-front and re-run stage 5b, which absorbs the half",
        "turn as it solves. If the robot is in fact standing normally, the head",
        "pan zero was set half a turn out; redo the head stage instead.",
    ]
    return "\n".join(lines)


def points_forward(direction: np.ndarray, tolerance: float = 0.0) -> bool:
    """Does `direction` have a forward component?

    FRAME_RELATIVE: for anything defined at a zero the operator chose, this is
    true by construction. Use it to confirm a solve is self-consistent, never as
    evidence that the frame itself is right.
    """
    return float(np.dot(np.asarray(direction, float)[:3], FORWARD)) > tolerance


def horizontal_quadrant_sign(arm: str) -> np.ndarray:
    """Signs of (forward, left) for each wrist camera's optical axis at zero.

    Beware: the two arms are NOT mirror images here. Measured on real runs the
    left camera points forward-and-left while the right points backward-and-
    right, both roughly 25 degrees off the lateral axis:

        left_arm   forward +0.42  left +0.91
        right_arm  forward -0.43  left -0.90

    The asymmetry comes from the model's wrist chains, not from the operator, so
    it is a fact to encode rather than a mistake to correct. Writing it as a
    signed pair keeps that visible; the old form `(-1, -1) if left else (1, 1)`
    read like a mirror and was easy to misread.

    FRAME_RELATIVE (see module docstring).
    """
    return np.array([1.0, 1.0]) if arm == "left_arm" else np.array([-1.0, -1.0])


def resolve_forward(direction: np.ndarray) -> np.ndarray:
    """Pick whichever of `direction` and its negation points forward.

    Used to break the tie when a construction yields an axis but not its sign,
    such as the perpendicular to the line joining the two arm mounts.
    """
    direction = np.asarray(direction, float)
    return direction if float(np.dot(direction[:len(FORWARD)],
                                     FORWARD[:len(direction)])) > 0.0 \
        else -direction
