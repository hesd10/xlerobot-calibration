"""How the robot is standing, and what that renames.

The chassis can be driven back-to-front so the storage half faces the board. To
keep the arms usable the flanges are physically turned 180 degrees, which means
the arm the operator sees on their left is the one the model calls `right_arm`.

The operator should never have to do that swap in their head: every prompt names
the arm they can point at, and every stored result keeps the model's name. This
module is the seam between those two vocabularies for the web tool. The mapping
itself lives in calibration/frames.py so the solver and the tool cannot drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

_CALIBRATION = Path(__file__).resolve().parent.parent / "calibration"
if str(_CALIBRATION) not in sys.path:
    sys.path.insert(0, str(_CALIBRATION))

import frames  # noqa: E402

NORMAL = frames.NORMAL
FLIPPED = frames.FLIPPED
MOUNTINGS = frames.MOUNTINGS
SIDES = frames.SIDES

check = frames.check_mounting
named_arm = frames.named_arm
named_camera = frames.named_camera
physical_side = frames.physical_side


def is_flipped(mounting: str) -> bool:
    return check(mounting) == FLIPPED


def label(mounting: str) -> str:
    """One line naming the mounting, for headings."""
    return {NORMAL: "Normal (arms face the board)",
            FLIPPED: "Back-to-front (chassis turned, flanges rotated)"}[
        check(mounting)]


def side_label(side: str, mounting: str) -> str:
    """Name a physical side to the operator, disclosing the model name.

    Under a flipped mounting the two names disagree, and hiding that would make
    the saved results look wrong to anyone who opens them.
    """
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}, expected one of {SIDES}")
    if not is_flipped(mounting):
        return f"{side} arm"
    return f"{side} arm (the model calls it {named_arm(side, mounting)})"


def arm_order(mounting: str) -> tuple[str, ...]:
    """The model arm names, ordered physically left first.

    Every user-facing list of arms goes through here so the operator always
    reads their own left before their own right. Under a normal mounting this
    is the model order unchanged, so nothing moves.
    """
    return frames.working_order(check(mounting))


def camera_order(mounting: str) -> tuple[str, ...]:
    """The camera roles, head first then the wrists physically left to right."""
    return ("head",) + tuple(named_camera(side, mounting) for side in SIDES)


def physical_camera_side(camera: str, mounting: str) -> str:
    """Which side the operator sees a wrist camera on.

    The wrist roles are named after the arm they are bolted to, so they turn
    with the flanges: back-to-front, the model's `left_wrist` camera rides the
    arm on the operator's right.
    """
    for side in SIDES:
        if named_camera(side, mounting) == camera:
            return side
    raise ValueError(f"unknown wrist camera {camera!r}")
