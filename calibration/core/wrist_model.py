"""Wrist camera kinematics for stage 6 calibration.

Per arm, solves 7 parameters, one of which is fixed by convention:
  - T_wrist_cam (6 DoF, but 1 held): camera mount relative to Fixed_Jaw body
  - wrist_roll_zero (1): correction to stage 4's rough wrist_roll zero

Gauge freedom
-------------
    wrist_roll zero  <->  camera roll about the roll axis

Rolling the wrist joint by δ and rolling the camera mount by -δ about the same
axis (through the same origin) produce IDENTICAL camera poses at every posture.
Verified numerically: the prediction difference is 0.000 mm / 0.0000 deg. This
is exact gauge freedom, not weak observability -- the lever arm does not help,
because the roll joint rotates the whole Fixed_Jaw body (camera included) about
that axis, and the mount's own rotation about the same axis cancels it term for
term. It is the same coupling that makes the head's tilt zero unsolvable.

How the gauge is broken (physical, not by XML convention)
---------------------------------------------------------
The operator's measurement fixes it: at the XML zero configuration (all TRUE
joint angles = 0), the wrist camera's optical axis lies in the chassis XY plane
(horizontal). Sliding along the gauge orbit rotates the optical-axis-AT-XML-ZERO
about the roll axis, so it crosses horizontal at a well-defined offset. This
defines where wrist_roll_zero sits.

The solve is done in two steps (see wrist_solve.fit):
  1. Fit the 6 observable parameters with the gauge projected out (well
     conditioned, condition number ~20). This nails everything the camera images
     can see, but leaves the gauge (where "zero" sits) undetermined.
  2. Slide exactly along the gauge orbit -- which leaves every observed-view
     prediction unchanged -- until the optical axis at XML zero is horizontal and
     points into the operator's stated quadrant. This costs nothing in fit error
     and breaks the gauge by the physical measurement, not an arbitrary value.

The functions gauge_vector/free_basis/free_of/with_free implement step 1's
projection; wrist_solve._apply_gauge / _resolve_gauge_horizontal implement the
exact orbit slide of step 2.

Forward kinematics
------------------
World (board) → Body (chassis) → Arm_root → ... → Fixed_Jaw → wrist_camera

T_W_cam_pred = T_W_B @ T_B_A @ FK(q_shoulder_pan, q_shoulder_lift, q_elbow,
                                   q_wrist_pitch, q_wrist_roll) @ T_wrist_cam

Known from prior stages:
  - T_W_B: stage 3
  - T_B_A: stage 5 (arm root mounting correction)
  - shoulder_pan/lift/elbow/wrist_pitch zeros: stage 5

To solve here:
  - T_wrist_cam: camera mount (roll component held from XML)
  - wrist_roll_zero: correction to stage 4's rough zero
"""

from __future__ import annotations

import numpy as np

from . import se3

# Parameter layout: T_wrist_cam as an se3 6-vector, wrist_roll_zero as 1 scalar.
# But the mount's rotation about the roll axis is a gauge freedom shared with
# wrist_roll_zero, so it is held at the XML value and removed from the free set.
N_PARAMS = 7
MOUNT_BLOCK = slice(0, 6)
ROLL_ZERO_INDEX = 6

# The wrist_roll joint axis in the Fixed_Jaw body frame (XML: axis="0 -1 0").
# The camera mount's rotation about THIS axis is the gauge direction that is
# held by convention, since it is indistinguishable from wrist_roll_zero.
ROLL_AXIS_IN_WRIST = np.array([0.0, -1.0, 0.0])

# Wrist body names from model_map.WRIST_BODIES
WRIST_BODIES = {"left_arm": "Fixed_Jaw", "right_arm": "Fixed_Jaw_2"}

# Arm motor names (stage 5 solves the first 4, stage 6 solves the 5th)
ARM_JOINT_NAMES = {
    "left_arm": [
        "left_arm_shoulder_pan",
        "left_arm_shoulder_lift",
        "left_arm_elbow_flex",
        "left_arm_wrist_flex",
        "left_arm_wrist_roll",
    ],
    "right_arm": [
        "right_arm_shoulder_pan",
        "right_arm_shoulder_lift",
        "right_arm_elbow_flex",
        "right_arm_wrist_flex",
        "right_arm_wrist_roll",
    ],
}


def nominal_mount(arm: str) -> np.ndarray:
    """Camera mount with optical axis horizontal, roughly in XY plane at ~30° from Y.

    Physical constraint from user (measured observation, not XML): at XML zero, the
    wrist camera optical axis lies in the horizontal XY plane (Z_world ≈ 0), pointing
    into quadrant III for left arm (left-forward, "left dominant, ~30° toward front")
    and quadrant I for right arm (right-backward, "right dominant, ~30° toward back").

    More precisely (chassis frame):
      - Left:  optical axis ≈ [-sin(30°), -cos(30°), 0] ≈ [-0.5, -0.866, 0]
      - Right: optical axis ≈ [+sin(30°), +cos(30°), 0] ≈ [+0.5, +0.866, 0]

    The 30° is approximate; calibration will refine. In Fixed_Jaw local frame, the
    wrist_roll joint axis is local -Y (maps to world ±Y depending on arm). The gauge
    freedom is: rotating the camera about this wrist_roll axis (local -Y) changes
    the optical axis direction in the XY plane, which is indistinguishable from
    adjusting wrist_roll_zero. We project out this gauge direction so the optimizer
    can't vary it; wrist_roll_zero absorbs the correction instead.

    Camera position from XML geom: local [~0, -0.007, 0.024] in Fixed_Jaw frame.
    """
    if arm == "left_arm":
        pos = np.array([5.44e-08, -0.007, 0.024])
    else:  # right_arm
        pos = np.array([5.44e-08, -0.006, 0.024])

    # Construct rotation for optical axis in Fixed_Jaw local frame.
    # At XML zero, Fixed_Jaw is rotated so its local +Z maps to world -X (forward).
    # We want cam_Z (optical axis) to point roughly 30° off from local -Y (the arm's
    # lateral direction) toward local -Z (forward reach). But we work in local coords.
    #
    # Strategy: optical axis (cam +Z) tilts from local -Y toward local -Z by ~30°.
    # Left arm user said "left-forward": local -Y dominant, -Z secondary.
    # In local frame: cam_Z ≈ [-sin(30°), -cos(30°), 0] rotated into local basis?
    # Actually simpler: use the fact that at zero, local -Y → world -Y (left),
    # local -Z → world -X (forward). So "left-forward ~30°" in world is directly
    # [-0.5, -0.866, 0] in world, which we need to express in Fixed_Jaw local.
    #
    # At XML zero: Fixed_Jaw quat = [0.707, 0, 0.707, 0] = 90° about Y.
    # R_world_jaw rotates local +Z → world -X, local +Y → world +Y, local +X → world +Z.
    # Inverse: world -X → local +Z, world -Y → local -Y, world 0 → local 0.
    # So world optical [-0.5, -0.866, 0] → local [0, -0.866, -0.5] (X comp goes to Z).
    #
    # For simplicity and since this is just a starting guess, set:
    #   cam_Z_local ≈ [-0.5, -0.866, 0]  (treat as if in a frame that's not rotated)
    # This gives the right *world* direction after Fixed_Jaw's 90° rotation is applied.
    
    # Actually, let's be more careful. Build the nominal in world frame first, then
    # convert to Fixed_Jaw local. At zero:
    #   Left:  optical_world ≈ [-0.5, -0.866, 0]
    #   Right: optical_world ≈ [+0.5, +0.866, 0]
    # Fixed_Jaw zero orientation: quat [0.707108, 0, 0.707105, 0] = Ry(90°).
    # R_world_jaw columns = [jaw_X_world, jaw_Y_world, jaw_Z_world]
    #   = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  (approx, from 90° Y rot)
    # Inverse: jaw_local_X = world_Z, jaw_local_Y = world_Y, jaw_local_Z = -world_X.
    # So optical_world → optical_local via R_jaw_world @ optical_world:
    #   Left:  [0,0,1; 0,1,0; -1,0,0] @ [-0.5, -0.866, 0] = [0, -0.866, 0.5]
    #   Right: [0,0,1; 0,1,0; -1,0,0] @ [+0.5, +0.866, 0] = [0, +0.866, -0.5]
    
    # By the arms' mirror symmetry, BOTH cameras point the same way in their own
    # Fixed_Jaw local frame; the different world orientations of Fixed_Jaw produce
    # the mirrored world directions the operator described. Verified numerically:
    #   local [0, -c, -s] → world [-0.5, -0.866, 0] (left,  quadrant III) ✓
    #   local [0, -c, -s] → world [+0.5, +0.866, 0] (right, quadrant I)  ✓
    # where c = cos(30°) ≈ 0.866 (Y dominant), s = sin(30°) ≈ 0.5.
    angle_deg = 30.0
    c = np.cos(np.radians(angle_deg))  # ~0.866
    s = np.sin(np.radians(angle_deg))  # ~0.5
    optical_local = np.array([0.0, -c, -s])
    optical_local /= np.linalg.norm(optical_local)
    
    # Build orthonormal frame with cam_Z = optical_local.
    # Pick cam_Y (image down) to be roughly local -Z (downward in jaw frame),
    # ensuring cam_X × cam_Y = cam_Z.
    # Gram-Schmidt: start with rough_Y = [0, 0, -1], orthogonalize to cam_Z.
    rough_y = np.array([0, 0, -1])
    cam_y = rough_y - np.dot(rough_y, optical_local) * optical_local
    cam_y /= np.linalg.norm(cam_y)
    cam_x = np.cross(cam_y, optical_local)
    cam_x /= np.linalg.norm(cam_x)
    
    R = np.column_stack([cam_x, cam_y, optical_local])
    return se3.make_transform(R, pos)


def pack(xi: np.ndarray, roll_zero: float) -> np.ndarray:
    """Local mount perturbation (6) and roll zero (1) -> the 7-vector.

    `xi` is a local se3 perturbation of the nominal mount: the actual mount is
    nominal_mount(arm) @ exp_se3(xi). `roll_zero` is the correction to stage 4's
    rough wrist_roll zero, in radians.
    """
    p = np.zeros(N_PARAMS)
    p[MOUNT_BLOCK] = np.asarray(xi, float).reshape(6)
    p[ROLL_ZERO_INDEX] = float(roll_zero)
    return p


def unpack(p: np.ndarray, arm: str) -> tuple[np.ndarray, float]:
    """The 7-vector -> T_wrist_cam, roll_zero.

    T_wrist_cam = nominal_mount(arm) @ exp_se3(xi).
    """
    p = np.asarray(p, float).reshape(-1)
    if p.size != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} parameters, got {p.size}")
    T_wrist_cam = nominal_mount(arm) @ se3.exp_se3(p[MOUNT_BLOCK])
    return T_wrist_cam, float(p[ROLL_ZERO_INDEX])


def _adjoint_inv(T: np.ndarray) -> np.ndarray:
    """Ad_{T^-1}, mapping twists [w, u] in the world frame to the body frame."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ad_inv = np.zeros((6, 6))
    Ad_inv[:3, :3] = R.T
    Ad_inv[:3, 3:] = -R.T @ se3.skew(t)
    Ad_inv[3:, 3:] = R.T
    return Ad_inv


def gauge_vector(arm: str) -> np.ndarray:
    """The 7-vector direction in (xi, roll_zero) space that leaves all
    predictions invariant.

    Moving along this direction changes wrist_roll_zero and the mount's rotation
    about the wrist_roll axis (Fixed_Jaw local -Y) together, with no effect on
    any camera pose. It is projected out of the free parameters so the fit is
    well conditioned.

    Derivation: predictions are invariant under
        T_wrist_cam -> R_roll(-delta) @ T_wrist_cam,  roll_zero -> +delta
    where R_roll is rotation about ROLL_AXIS_IN_WRIST (local -Y) through the
    Fixed_Jaw origin (screw [a, 0]). With the local parameterisation
    T_wrist_cam = T_nom @ exp(xi), the left multiplication maps to a local
    perturbation via the adjoint:
        d_xi = -Ad_{T_nom^-1} @ [a, 0]
    where a = ROLL_AXIS_IN_WRIST = [0, -1, 0] (Fixed_Jaw local -Y).
    """
    T_nom = nominal_mount(arm)
    screw_roll = np.concatenate([ROLL_AXIS_IN_WRIST, np.zeros(3)])
    d_xi = -_adjoint_inv(T_nom) @ screw_roll
    g = np.concatenate([d_xi, [1.0]])
    return g / np.linalg.norm(g)


def free_basis(arm: str) -> np.ndarray:
    """A 7x6 orthonormal basis for the parameter directions orthogonal to the
    gauge. The optimiser works in these 6 coordinates, so the gauge component
    stays fixed at its initial value (the XML nominal roll)."""
    g = gauge_vector(arm)
    # Complete g to an orthonormal basis via QR, then drop the g direction.
    A = np.eye(N_PARAMS)
    A[:, 0] = g
    Q, _ = np.linalg.qr(A)
    # Q[:, 0] is parallel to g (up to sign); the rest span the complement.
    return Q[:, 1:]


def free_of(p: np.ndarray, arm: str) -> np.ndarray:
    """Full 7-vector -> 6 free coordinates."""
    return free_basis(arm).T @ np.asarray(p, float)


def with_free(free: np.ndarray, arm: str,
              p0: np.ndarray | None = None) -> np.ndarray:
    """6 free coordinates -> full 7-vector, holding the gauge at p0's value."""
    B = free_basis(arm)
    base = np.zeros(N_PARAMS) if p0 is None else np.asarray(p0, float)
    # Keep base's gauge component, replace the complement with `free`.
    g = gauge_vector(arm)
    gauge_component = (base @ g) * g
    return gauge_component + B @ np.asarray(free, float)


def camera_in_world(
    sim,
    arm: str,
    angles: dict[str, float],
    T_wrist_cam: np.ndarray,
    T_W_B: np.ndarray,
    T_B_A: np.ndarray,
) -> np.ndarray:
    """Predicted camera pose in world frame for one arm posture.
    
    `angles` are true joint angles (servo readings with zeros already applied).
    T_B_A is the arm root mounting correction from stage 5.
    """
    # Set joint angles and run FK to get wrist body pose
    sim.set_joints(angles)
    wrist_pos, wrist_rot = sim.body_pose_in_chassis(WRIST_BODIES[arm])
    T_base_wrist = se3.make_transform(wrist_rot, wrist_pos)
    
    # Compose: World -> Body -> Arm_root (corrected) -> Wrist -> Camera
    T_W_cam = (
        np.asarray(T_W_B, float)
        @ np.asarray(T_B_A, float)
        @ T_base_wrist
        @ np.asarray(T_wrist_cam, float)
    )
    return T_W_cam


def residuals(
    p: np.ndarray,
    sim,
    arm: str,
    reported_angles: list[dict[str, float]],
    observed_poses: list[np.ndarray],
    rough_roll_zero: float,
    T_W_B: np.ndarray,
    T_B_A: np.ndarray,
) -> np.ndarray:
    """Pose error per view, as a 6-vector in the Lie algebra.
    
    `reported_angles` are servo readings with stage 5 zeros applied (but not
    the wrist_roll correction we're solving for).
    `observed_poses` are T_cam_board from PnP (= T_cam_world since W = board).
    `rough_roll_zero` is the stage 4 rough zero for wrist_roll, in radians.
    
    Only the pose residuals appear here. The horizontal-optical-axis constraint
    that breaks the gauge is applied separately, by sliding along the gauge orbit
    after this pose fit converges (see wrist_solve._resolve_gauge_horizontal), so
    it never fights the pose fit or degrades conditioning.
    """
    T_wrist_cam, roll_zero_correction = unpack(p, arm)
    
    # The true wrist_roll zero is the rough one plus the correction
    roll_motor = ARM_JOINT_NAMES[arm][4]  # wrist_roll is the 5th joint
    
    out = []
    for angles_dict, obs_pose in zip(reported_angles, observed_poses):
        # Apply the roll zero correction to get true angles
        true_angles = dict(angles_dict)
        true_angles[roll_motor] = angles_dict[roll_motor] + roll_zero_correction
        
        # Predict camera pose
        T_W_cam_pred = camera_in_world(sim, arm, true_angles, T_wrist_cam,
                                        T_W_B, T_B_A)
        
        # Observed is T_cam_world from PnP; invert to get T_world_cam
        T_W_cam_obs = se3.invert(np.asarray(obs_pose, float))
        
        # Residual: log(T_obs^-1 @ T_pred)
        out.append(se3.log_se3(se3.invert(T_W_cam_obs) @ T_W_cam_pred))
    
    return np.concatenate(out) if out else np.zeros(0)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def initial_guess(arm: str) -> np.ndarray:
    """A starting point: XML nominal mount (xi=0), zero roll correction.

    The mount is a local perturbation of nominal_mount(arm), so xi=0 means the
    mount starts exactly at the XML camera geom pose. The mount's rotation about
    the roll axis is held there (gauge), and wrist_roll_zero absorbs the rest.
    """
    return pack(np.zeros(6), 0.0)
