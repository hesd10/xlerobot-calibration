"""Head kinematics: the forward model, and exact re-definition of the body frame.

The chain is base -> pan -> tilt -> camera. Two revolute joints, both taken from
the model's XML, and one fixed camera mount that calibration solves for.

Which parameters are worth solving
----------------------------------
The obvious set is T_W^B (6), the two joint zeros (2), the pan axis position (3)
and the camera mount (6): 17 in all. Five of those directions are exactly
unobservable from a camera watching one fixed board, with singular values eleven
orders of magnitude below the largest. The null space says why:

    pan zero      <-> base yaw          turning the head looks like turning the base
    tilt zero     <-> camera pitch      nodding looks like tilting the camera
    axis position <-> base translation  moving the axis looks like moving the base

This is gauge freedom, not weak data: no amount of extra capture removes it. So
the zeros are fixed at the current mechanical position and the axis position is
taken from the XML, leaving 12 parameters that are well conditioned.

Fixing them at the wrong value costs nothing. T_W^B absorbs the error, and
predictions at poses that took no part in the fit are unaffected to machine
precision -- the body frame shifts, but every frame attached to it shifts with it.

Changing the convention later
-----------------------------
Because rotations about one axis compose, A(q) = A(q - d) A(d), so shifting a
joint zero by d is exactly a change of the neighbouring fixed transform. That
makes `shift_pan_zero` and `shift_tilt_zero` closed-form and exact, so a later
re-definition of the head zero needs no recapture and no re-solve.
"""

from __future__ import annotations

import numpy as np

from . import se3

# Geometry from calibration/model/xlerobot_calib.xml. The pan axis is vertical in
# the base frame; tilt is about the sideways axis of the pan link.
PAN_ORIGIN = np.array([-0.103, 0.0, 0.673])
PAN_AXIS = np.array([0.0, 0.0, 1.0])
TILT_OFFSET = np.array([0.001, 0.002, 0.09815])

# The XML writes the tilt axis as "0 1 0", but inside head_tilt_link, which
# carries quat="0 0 0 1" -- a half turn about z. Rotated back into the pan link's
# coordinates, where everything here is expressed, that axis is -y. Taking the
# XML number at face value flips the sense of every tilt rotation; measured on
# this unit that alone drove holdout error from 5.7mm to 92mm.
TILT_AXIS = np.array([0.0, -1.0, 0.0])

# Nominal camera mount on head_tilt_link, used only as a starting guess.
CAM_NOMINAL = np.array([0.025, 0.0, 0.03])

# Joint limits from the XML, in radians. Tilt is negated along with its axis, so
# the XML's (-0.76, 1.45) becomes (-1.45, 0.76) in this convention.
PAN_RANGE = (-3.2, 3.2)
TILT_RANGE = (-1.45, 0.76)

# Which way a servo turns the mechanism, which the XML cannot know: it describes
# geometry, not how the motor was wired or assembled. Applied to the measured
# angle before the kinematics, so PAN_AXIS/TILT_AXIS stay as MuJoCo reports them.
#
# Measured in stage 2, never assumed. A residual cannot find these: on this unit
# both pan senses fit the capture to 3.56mm, while the wrong one placed the board
# 1515mm above the floor against a measured 750mm and had the camera looking up
# at a board that was plainly below it. Only an external fact separates them.
#
# These module-level values are the fallback used by tests and by synthetic work,
# where no robot has been measured. Anything touching real data must go through
# `senses_from_result()` so a missing stage 2 is an error rather than a guess.
PAN_SENSE = -1.0
TILT_SENSE = +1.0


# Real motor names, in the order (pan, tilt). Both live on bus1 as ids 7 and 8.
PAN_MOTOR = "head_motor_1"
TILT_MOTOR = "head_motor_2"
HEAD_MOTORS = (PAN_MOTOR, TILT_MOTOR)


def load_senses() -> tuple[float, float]:
    """(pan, tilt) senses as recorded in stage 2, refusing to guess.

    Callers working with real captures use this rather than the constants above,
    so that a robot whose senses were never measured stops instead of quietly
    calibrating a mirror image of itself.
    """
    from . import senses as _senses
    got = _senses.require([PAN_MOTOR, TILT_MOTOR])
    return float(got.sign(PAN_MOTOR)), float(got.sign(TILT_MOTOR))


def mounting_pan_offset() -> float:
    """The model pan angle of the posture the operator records as the head zero.

    The operator sets the head zero facing the calibration board, because that
    is the only posture that sees anything. Which model angle that posture is
    depends entirely on how the robot is standing:

      normal          facing the board is facing chassis-FRONT  -> q = 0
      back-to-front   facing the board is facing chassis-BACK   -> q = pi

    That correspondence is fixed by the mounting and nothing else, so it is
    derived here rather than carried through the pipeline. No stage records it,
    no stage passes it on, and no result file has to agree about it.

    The offset belongs at the point where an encoder count becomes a model
    angle, and only there. Applying it earlier -- by moving the stored zero and
    letting T_W_B absorb the matching rotation -- is an exact gauge change and
    so looks harmless, but it puts the half turn in two places at once: the
    seated zero and the conversion below. Both then supply it, the pan lands a
    full turn from where it belongs, and every head prediction comes back
    rotated by 180 deg.

    The signature of that failure: a head rotation error near 180 deg that does
    not vary with pan or tilt, while both wrist cameras pass. The wrists reach
    the board through T_B_A rather than through the head's pan angle, so they
    are untouched by it. An error that does track the joint angles is a zero or
    a sense, not this.
    """
    import sys
    from pathlib import Path

    # frames lives one level up, beside the stages that own the declaration.
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import frames

    return np.pi if frames.declared_mounting() == frames.FLIPPED else 0.0


def rotate_about(axis: np.ndarray, angle: float,
                 point: np.ndarray) -> np.ndarray:
    """Transform for a rotation of `angle` about `axis` through `point`.

    A joint rotates about a line, not about the frame origin, so the translation
    part is whatever keeps `point` fixed.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    R = se3.exp_so3(axis * float(angle))
    point = np.asarray(point, dtype=float)
    return se3.make_transform(R, point - R @ point)


def pan_transform(pan: float, axis_origin: np.ndarray | None = None,
                  sense: float | None = None) -> np.ndarray:
    """T_base_pan for a pan angle, in radians as read from the servo."""
    origin = PAN_ORIGIN if axis_origin is None else np.asarray(axis_origin, float)
    s = PAN_SENSE if sense is None else float(sense)
    return rotate_about(PAN_AXIS, s * pan, origin)


def tilt_transform(tilt: float, axis_origin: np.ndarray | None = None,
                   sense: float | None = None) -> np.ndarray:
    """T_pan_tilt for a tilt angle, in radians.

    The tilt axis passes through the tilt joint, which sits at a fixed offset
    from the pan axis and therefore moves with pan. Expressed in the pan link's
    coordinates -- which is what this returns -- that point is stationary.
    """
    origin = PAN_ORIGIN if axis_origin is None else np.asarray(axis_origin, float)
    s = TILT_SENSE if sense is None else float(sense)
    return rotate_about(TILT_AXIS, s * tilt, origin + TILT_OFFSET)


def T_base_cam(pan: float, tilt: float, T_tilt_cam: np.ndarray,
               axis_origin: np.ndarray | None = None,
               senses: tuple[float, float] | None = None) -> np.ndarray:
    """Camera pose in the base frame for a given head posture."""
    sp, st = (None, None) if senses is None else senses
    return (pan_transform(pan, axis_origin, sp)
            @ tilt_transform(tilt, axis_origin, st)
            @ np.asarray(T_tilt_cam, dtype=float))


def T_world_cam(T_W_B: np.ndarray, pan: float, tilt: float,
                T_tilt_cam: np.ndarray,
                axis_origin: np.ndarray | None = None,
                senses: tuple[float, float] | None = None) -> np.ndarray:
    """Camera pose in the world frame."""
    return np.asarray(T_W_B, float) @ T_base_cam(
        pan, tilt, T_tilt_cam, axis_origin, senses)


def T_cam_world(T_W_B: np.ndarray, pan: float, tilt: float,
                T_tilt_cam: np.ndarray,
                axis_origin: np.ndarray | None = None,
                senses: tuple[float, float] | None = None) -> np.ndarray:
    """World pose in the camera frame, which is what PnP measures."""
    return np.linalg.inv(T_world_cam(T_W_B, pan, tilt, T_tilt_cam,
                                     axis_origin, senses))


# ---- changing the zero convention, exactly --------------------------------

def shift_pan_zero(T_W_B: np.ndarray, delta: float,
                   axis_origin: np.ndarray | None = None,
                   sense: float | None = None) -> np.ndarray:
    """T_W^B after the pan zero moves by `delta` radians.

    The same encoder count now reports an angle `delta` smaller. Since rotations
    about one axis compose, A(pan) = A(pan - delta) A(delta), and the A(delta) is
    absorbed into T_W^B. Exact, so no recapture and no re-solve.
    """
    return np.asarray(T_W_B, float) @ pan_transform(
        delta, axis_origin, sense=sense)


def shift_tilt_zero(T_tilt_cam: np.ndarray, delta: float,
                    axis_origin: np.ndarray | None = None,
                    sense: float | None = None) -> np.ndarray:
    """T_tilt_cam after the tilt zero moves by `delta` radians.

    The tilt joint is the last one before the camera, so its surplus rotation is
    absorbed on the far side: into the camera mount rather than into T_W^B.
    """
    origin = PAN_ORIGIN if axis_origin is None else np.asarray(axis_origin, float)
    # Expressed about the tilt joint, which is where the mount hangs from, and
    # through TILT_SENSE so that `delta` means the same thing here as it does in
    # tilt_transform. Without that the shift is no longer exact.
    s = TILT_SENSE if sense is None else float(sense)
    return rotate_about(TILT_AXIS, s * delta,
                        origin + TILT_OFFSET) @ np.asarray(T_tilt_cam, float)


# ---- the 12 parameters that are actually solved ---------------------------

BLOCKS = {"T_W_B": 6, "T_tilt_cam": 6}
N_PARAMS = 12


def pack(T_W_B: np.ndarray, T_tilt_cam: np.ndarray) -> np.ndarray:
    """Two transforms -> the 12-vector the optimiser works on."""
    return np.concatenate([se3.log_se3(np.asarray(T_W_B, float)),
                           se3.log_se3(np.asarray(T_tilt_cam, float))])


def unpack(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 12-vector -> T_W_B, T_tilt_cam."""
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} parameters, got {p.size}")
    return se3.exp_se3(p[0:6]), se3.exp_se3(p[6:12])


def residuals(p: np.ndarray, pans: np.ndarray, tilts: np.ndarray,
              observed: list[np.ndarray],
              axis_origin: np.ndarray | None = None,
              senses: tuple[float, float] | None = None) -> np.ndarray:
    """Pose error per view, as a 6-vector in the Lie algebra.

    `observed` holds T_cam_board from PnP. With W defined as the board frame,
    T_cam_board and T_cam_world are the same thing, which is the whole point of
    defining the world that way.
    """
    T_W_B, T_tilt_cam = unpack(p)
    out = []
    for pan, tilt, obs in zip(pans, tilts, observed):
        pred = T_cam_world(T_W_B, float(pan), float(tilt), T_tilt_cam,
                           axis_origin, senses)
        out.append(se3.log_se3(np.linalg.inv(np.asarray(obs, float)) @ pred))
    return np.concatenate(out) if out else np.zeros(0)


def nominal_mount(pitch_deg: float = 30.0) -> np.ndarray:
    """A plausible T_tilt_cam: camera on the head, aimed `pitch_deg` below level.

    Two things here are easy to get wrong. The translation is the camera's
    position in the *base* frame, not an offset from the tilt joint, so the joint
    position has to be included. And the rotation cannot be identity: identity
    points the optical axis along base +z, i.e. straight at the ceiling, which is
    a poor place to start a fit for a head that looks ahead and down.
    """
    t = np.deg2rad(pitch_deg)
    z = np.array([np.cos(t), 0.0, -np.sin(t)])   # optical axis, forward and down
    x = np.array([0.0, -1.0, 0.0])                # image right
    R = np.column_stack([x, np.cross(z, x), z])
    return se3.make_transform(R, PAN_ORIGIN + TILT_OFFSET + CAM_NOMINAL)


def initial_guess(pans: np.ndarray, tilts: np.ndarray,
                  observed: list[np.ndarray],
                  senses: tuple[float, float] | None = None) -> np.ndarray:
    """A starting point from the view nearest the head's zero posture.

    At pan = tilt = 0 the chain reduces to T_W_B = T_world_cam @ inv(T_tilt_cam),
    so the nominal mount gives a T_W_B good enough to converge from. Picking the
    closest view rather than assuming one exists keeps this usable even if the
    capture never sat exactly at zero.

    This single-anchor approach is now deprecated; use initial_guesses for a
    robust multi-start that avoids basin traps.
    """
    pans = np.asarray(pans, float)
    tilts = np.asarray(tilts, float)
    i = int(np.argmin(pans ** 2 + tilts ** 2))

    T_tilt_cam = nominal_mount()
    T_world_cam_i = np.linalg.inv(np.asarray(observed[i], float))
    T_base_cam_i = T_base_cam(float(pans[i]), float(tilts[i]), T_tilt_cam,
                              senses=senses)
    return pack(T_world_cam_i @ np.linalg.inv(T_base_cam_i), T_tilt_cam)


def initial_guesses(pans: np.ndarray, tilts: np.ndarray,
                    observed: list[np.ndarray],
                    senses: tuple[float, float] | None = None,
                    thoroughness: str = "quick") -> list[np.ndarray]:
    """Multiple starting points to avoid local minima where the camera ends up
    ~0.8 m from the tilt joint instead of the correct ~0.04 m.

    Each view can anchor a T_W_B estimate given the nominal mount. When a single
    anchor is unlucky—noisy PnP or far from zero—the fit converges to a wrong
    basin. Trying several anchors and several pitch angles makes success almost
    certain: the data from check_stage3_basin.py showed 2/5 failed sessions
    recovered with nominal+jitter, proving the data supported a correct solution.

    Candidates are tried in order. The first one is the current default (nearest
    view, 30° pitch) so behavior stays unchanged when it already works.

    `thoroughness` controls the number of guesses:
      - "quick": 12 guesses (3 pitch × 4 views), for typical well-distributed data
      - "medium": 48 guesses (6 pitch × 8 views), when quick fails
      - "exhaustive": all views × 6 pitch, when medium fails
    """
    pans = np.asarray(pans, float)
    tilts = np.asarray(tilts, float)
    n = len(pans)
    if n == 0:
        return []

    # Views sorted by distance from zero posture, nearest first
    distances = pans ** 2 + tilts ** 2
    sorted_indices = np.argsort(distances)

    # Select pitch angles and anchor views based on thoroughness
    if thoroughness == "quick":
        pitches = [30.0, 10.0, 50.0]
        pick_indices = sorted_indices[:min(4, n)]
    elif thoroughness == "medium":
        pitches = [30.0, 10.0, 20.0, 40.0, 50.0, 60.0]
        pick_indices = sorted_indices[:min(8, n)]
    else:  # exhaustive
        pitches = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        pick_indices = sorted_indices  # all views

    candidates = []
    for pitch in pitches:
        T_tilt_cam = nominal_mount(pitch)
        for i in pick_indices:
            T_world_cam_i = np.linalg.inv(np.asarray(observed[i], float))
            T_base_cam_i = T_base_cam(float(pans[i]), float(tilts[i]),
                                     T_tilt_cam, senses=senses)
            candidates.append(pack(T_world_cam_i @ np.linalg.inv(T_base_cam_i),
                                  T_tilt_cam))
    return candidates


def pan_budget(fovx_deg: float, board_width_m: float,
               distance_m: float, margin: float = 0.85) -> float:
    """How far pan may sweep with the board staying fully in frame, in degrees.

    Uses the calibrated field of view rather than a nominal one, since the two
    differ by several degrees on these modules. The margin keeps the board off
    the very edge, where detection is least reliable and distortion largest.
    """
    half_fov = np.deg2rad(fovx_deg) / 2.0
    half_board = np.arctan2(board_width_m / 2.0, distance_m)
    budget = (half_fov - half_board) * margin
    return float(max(0.0, np.rad2deg(budget)))
