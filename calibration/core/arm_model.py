"""Arm kinematics for contact calibration: the model, and its one gauge freedom.

Per arm the contact stage solves the arm mounting T_B^A (6), four joint zeros
(shoulder pan, shoulder lift, elbow flex, wrist flex) and the touch point (3).

What is observable
------------------
Thirteen parameters, twelve observable. Exactly one direction is unobservable
from contact data, with a singular value 4.5e-8 of the largest:

    zero shoulder_pan  <->  arm mount yaw

Rotating the first shoulder joint and yawing the whole arm move the touch point
identically, so no number of touches separates them. The mount's yaw is therefore
fixed at zero by convention and the shoulder pan zero absorbs it, exactly as the
head stage fixes its own gauge quantities.

Getting this wrong is expensive and quiet. In an se3 vector the first three
components are the rotation and the last three the translation, so the mount's
yaw is index 2. Holding index 5 instead -- a z translation, which carries no
weight in the null space at all -- leaves the freedom untouched and yields a 1mm
residual with 31 degrees of zero error. The tell is that the error does not move
when the initial guess changes: a convergence problem varies with the guess, a
gauge problem does not.

Wrist roll is excluded for a related reason: it is the last joint before the
touch point, so rotating it and sliding the touch point along its axis are the
same motion. Stage 6 solves it from the wrist camera instead.

How good the starting guess must be
-----------------------------------
Once the gauge is fixed, it barely matters. With 1mm of touch placement error and
0.1 degrees of joint noise, an initial zero guess wrong by +/-90 degrees still
recovers the zeros to 0.23 degrees and the touch point to 0.41mm. Stage 4 exists
to bound the joint ranges and to keep the encoder off its wrap seam, not to
supply an accurate zero.
"""

from __future__ import annotations

import numpy as np

from . import se3

# Joints whose zeros this stage solves, in order. Wrist roll is deliberately absent.
SOLVED_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")

# Parameter layout: T_B_A as an se3 6-vector, four zeros, the touch point.
N_PARAMS = 13
MOUNT_BLOCK = slice(0, 6)
ZERO_BLOCK = slice(6, 10)
TOUCH_BLOCK = slice(10, 13)

# The gauge: index 2 of the se3 vector is rotation about z, the arm mount's yaw.
MOUNT_YAW_INDEX = 2
FREE_INDICES = tuple(i for i in range(N_PARAMS) if i != MOUNT_YAW_INDEX)

# The body whose frame the touch point is expressed in: the last rigid link before
# the jaws open, so the point does not move when the gripper does.
TOUCH_BODY = {"left_arm": "Fixed_Jaw", "right_arm": "Fixed_Jaw_2"}

# A nominal touch point on the jaw, used only as a starting guess.
TOUCH_NOMINAL = np.array([0.0, -0.055, 0.0])


def joint_names(arm: str) -> list[str]:
    """The four motors whose zeros are solved, for one arm."""
    return [f"{arm}_{j}" for j in SOLVED_JOINTS]


def pack(T_B_A: np.ndarray, zeros: dict, touch: np.ndarray,
         arm: str) -> np.ndarray:
    """Mounting, zeros and touch point -> the 13-vector the optimiser uses."""
    p = np.zeros(N_PARAMS)
    p[MOUNT_BLOCK] = se3.log_se3(np.asarray(T_B_A, float))
    for i, name in enumerate(joint_names(arm)):
        p[6 + i] = float(zeros.get(name, 0.0))
    p[TOUCH_BLOCK] = np.asarray(touch, float)
    return p


def unpack(p: np.ndarray, arm: str) -> tuple[np.ndarray, dict, np.ndarray]:
    """The 13-vector -> T_B_A, the four zeros by motor name, the touch point."""
    p = np.asarray(p, float).reshape(-1)
    if p.size != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} parameters, got {p.size}")
    zeros = {name: float(p[6 + i]) for i, name in enumerate(joint_names(arm))}
    return se3.exp_se3(p[MOUNT_BLOCK]), zeros, p[TOUCH_BLOCK].copy()


def touch_in_base(sim, arm: str, angles: dict, touch: np.ndarray,
                  T_B_A: np.ndarray | None = None) -> np.ndarray:
    """Where the touch point lands in the base frame, for one posture.

    `angles` are true joint angles, i.e. servo readings with their zeros already
    applied. T_B_A is a correction on the whole arm, applied after the model's own
    kinematics.
    """
    sim.set_joints(angles)
    p, R = sim.body_pose_in_chassis(TOUCH_BODY[arm])
    point = p + R @ np.asarray(touch, float)
    if T_B_A is None:
        return point
    return (np.asarray(T_B_A, float) @ np.append(point, 1.0))[:3]


def residuals(p: np.ndarray, sim, arm: str, reported: list[dict],
              targets: list[np.ndarray]) -> np.ndarray:
    """Touch point error per contact, in metres.

    `reported` holds servo angles as read, `targets` the known board corner each
    touch was made against, in the base frame.
    """
    T_B_A, zeros, touch = unpack(p, arm)
    all_joints = list(reported[0].keys()) if reported else []
    out = []
    for angles, target in zip(reported, targets):
        true = {j: angles[j] + zeros.get(j, 0.0) for j in all_joints}
        got = touch_in_base(sim, arm, true, touch, T_B_A)
        out.append(got - np.asarray(target, float))
    return np.concatenate(out) if out else np.zeros(0)


def initial_guess(arm: str, zeros: dict | None = None) -> np.ndarray:
    """A starting point: no mounting error, the given zeros, a nominal touch point.

    Deliberately unfussy. Measured on synthetic contacts with realistic noise, a
    zero guess wrong by 90 degrees converges to the same answer as an exact one,
    so effort spent here buys nothing.
    """
    return pack(np.eye(4), zeros or {}, TOUCH_NOMINAL, arm)


def free_of(p: np.ndarray) -> np.ndarray:
    """The twelve components the optimiser is allowed to move."""
    return np.asarray(p, float)[list(FREE_INDICES)]


def with_free(p_full: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Reinsert the twelve free components into a full 13-vector."""
    p = np.asarray(p_full, float).copy()
    p[list(FREE_INDICES)] = np.asarray(free, float)
    return p
