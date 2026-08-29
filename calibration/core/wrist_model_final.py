"""Wrist camera + roll axis calibration (final version with axis position).

9-parameter model:
  - T_wrist_cam (6 DoF): camera mount relative to Fixed_Jaw body
  - wrist_roll_zero (1): correction to stage 4's rough wrist_roll zero
  - axis_lateral_position (2): position of roll axis in plane perpendicular to axis

The roll axis direction is measured directly from rotation deltas (robust, linear).
The axis lateral position accounts for the physical roll joint not passing through
the parent body origin as XML assumes.

Gauge freedom: roll_zero <-> mount rotation about the roll axis. Projected out.
"""

from __future__ import annotations

import numpy as np

from . import se3

# Parameter layout for the 9-vector:
#   [0:6]   T_wrist_cam as se3 6-vector (local perturbation of nominal)
#   [6]     wrist_roll_zero correction (radians)
#   [7:9]   axis_lateral_position: 2 coefficients for perpendicular basis vectors
N_PARAMS = 9
MOUNT_BLOCK = slice(0, 6)
ROLL_ZERO_INDEX = 6
LATERAL_BLOCK = slice(7, 9)

# Wrist body names
WRIST_BODIES = {"left_arm": "Fixed_Jaw", "right_arm": "Fixed_Jaw_2"}
PARENT_BODIES = {"left_arm": "Wrist_Pitch_Roll", "right_arm": "Wrist_Pitch_Roll_2"}

# Arm joint names
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
    """Camera mount with optical axis horizontal, ~30° off lateral."""
    if arm == "left_arm":
        pos = np.array([5.44e-08, -0.007, 0.024])
    else:
        pos = np.array([5.44e-08, -0.006, 0.024])
    
    angle_deg = 30.0
    c = np.cos(np.radians(angle_deg))
    s = np.sin(np.radians(angle_deg))
    optical_local = np.array([0.0, -c, -s])
    optical_local /= np.linalg.norm(optical_local)
    
    rough_y = np.array([0, 0, -1])
    cam_y = rough_y - np.dot(rough_y, optical_local) * optical_local
    cam_y /= np.linalg.norm(cam_y)
    cam_x = np.cross(cam_y, optical_local)
    cam_x /= np.linalg.norm(cam_x)
    
    R = np.column_stack([cam_x, cam_y, optical_local])
    return se3.make_transform(R, pos)


def perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build orthonormal basis perpendicular to axis for lateral position params."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    tmp = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    perp1 = np.cross(axis, tmp)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)
    perp2 /= np.linalg.norm(perp2)
    return perp1, perp2


def pack(xi: np.ndarray, roll_zero: float, lateral: np.ndarray) -> np.ndarray:
    """Pack parameters into the 9-vector."""
    p = np.zeros(N_PARAMS)
    p[MOUNT_BLOCK] = np.asarray(xi, float).reshape(6)
    p[ROLL_ZERO_INDEX] = float(roll_zero)
    p[LATERAL_BLOCK] = np.asarray(lateral, float).reshape(2)
    return p


def unpack(p: np.ndarray, arm: str, axis: np.ndarray,
           perp1: np.ndarray, perp2: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Unpack the 9-vector -> (T_wrist_cam, roll_zero, axis_point)."""
    p = np.asarray(p, float).reshape(-1)
    if p.size != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} parameters, got {p.size}")
    
    T_wrist_cam = nominal_mount(arm) @ se3.exp_se3(p[MOUNT_BLOCK])
    roll_zero = float(p[ROLL_ZERO_INDEX])
    axis_point = p[7] * perp1 + p[8] * perp2
    
    return T_wrist_cam, roll_zero, axis_point


def roll_transform(axis: np.ndarray, angle: float, q: np.ndarray) -> np.ndarray:
    """SE(3) for rotation `angle` about line through point `q` with direction `axis`.
    
    T = Trans(q) @ Rot(axis, angle) @ Trans(-q)
    """
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = se3.skew(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    # Rotation about line through q: translate to q, rotate, translate back
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = q - R @ q
    return T


def camera_in_world(
    sim,
    arm: str,
    angles: dict[str, float],
    T_wrist_cam: np.ndarray,
    axis: np.ndarray,
    axis_point: np.ndarray,
    T_W_B: np.ndarray,
    T_B_A: np.ndarray,
) -> np.ndarray:
    """Predicted camera pose using the parameterized roll axis."""
    parent_body = PARENT_BODIES[arm]
    sim.set_joints(angles)
    parent_pos, parent_rot = sim.body_pose_in_chassis(parent_body)
    T_base_parent = se3.make_transform(parent_rot, parent_pos)
    
    roll_motor = ARM_JOINT_NAMES[arm][4]
    roll_angle = angles[roll_motor]
    T_parent_jaw = roll_transform(axis, roll_angle, axis_point)
    
    T_W_cam = (
        np.asarray(T_W_B, float)
        @ np.asarray(T_B_A, float)
        @ T_base_parent
        @ T_parent_jaw
        @ np.asarray(T_wrist_cam, float)
    )
    return T_W_cam


def residuals(
    p: np.ndarray,
    sim,
    arm: str,
    axis: np.ndarray,
    perp1: np.ndarray,
    perp2: np.ndarray,
    reported_angles: list[dict[str, float]],
    observed_poses: list[np.ndarray],
    rough_roll_zero: float,
    T_W_B: np.ndarray,
    T_B_A: np.ndarray,
) -> np.ndarray:
    """Pose error per view."""
    T_wrist_cam, roll_zero_correction, axis_point = unpack(p, arm, axis, perp1, perp2)
    roll_motor = ARM_JOINT_NAMES[arm][4]
    
    out = []
    for angles_dict, obs_pose in zip(reported_angles, observed_poses):
        true_angles = dict(angles_dict)
        true_angles[roll_motor] = angles_dict[roll_motor] + roll_zero_correction
        
        T_W_cam_pred = camera_in_world(sim, arm, true_angles, T_wrist_cam,
                                        axis, axis_point, T_W_B, T_B_A)
        T_W_cam_obs = se3.invert(np.asarray(obs_pose, float))
        out.append(se3.log_se3(se3.invert(T_W_cam_obs) @ T_W_cam_pred))
    
    return np.concatenate(out) if out else np.zeros(0)


def _adjoint_inv(T: np.ndarray) -> np.ndarray:
    """Ad_{T^-1}, mapping twists [w, u] in world frame to body frame."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ad_inv = np.zeros((6, 6))
    Ad_inv[:3, :3] = R.T
    Ad_inv[:3, 3:] = -R.T @ se3.skew(t)
    Ad_inv[3:, 3:] = R.T
    return Ad_inv


def gauge_vector(arm: str, axis: np.ndarray) -> np.ndarray:
    """The 9-vector direction that leaves all predictions invariant.
    
    Gauge freedom: roll_zero + δ, mount -> R_axis(-δ) @ mount.
    In the local parameterization T_wrist_cam = T_nom @ exp(xi), this maps to:
        d_xi = -Ad_{T_nom^-1} @ [axis, 0],  d_roll_zero = 1,  d_lateral = [0,0]
    """
    T_nom = nominal_mount(arm)
    screw_roll = np.concatenate([axis, np.zeros(3)])
    d_xi = -_adjoint_inv(T_nom) @ screw_roll
    g = np.concatenate([d_xi, [1.0], [0.0, 0.0]])
    return g / np.linalg.norm(g)


def free_basis(arm: str, axis: np.ndarray) -> np.ndarray:
    """9x8 orthonormal basis for parameter directions orthogonal to the gauge."""
    g = gauge_vector(arm, axis)
    A = np.eye(N_PARAMS)
    A[:, 0] = g
    Q, _ = np.linalg.qr(A)
    return Q[:, 1:]  # Drop the gauge direction


def free_of(p: np.ndarray, arm: str, axis: np.ndarray) -> np.ndarray:
    """Full 9-vector -> 8 free coordinates."""
    return free_basis(arm, axis).T @ np.asarray(p, float)


def with_free(free: np.ndarray, arm: str, axis: np.ndarray,
              p0: np.ndarray | None = None) -> np.ndarray:
    """8 free coordinates -> full 9-vector, holding gauge at p0's value."""
    B = free_basis(arm, axis)
    base = np.zeros(N_PARAMS) if p0 is None else np.asarray(p0, float)
    g = gauge_vector(arm, axis)
    gauge_component = (base @ g) * g
    return gauge_component + B @ np.asarray(free, float)


def initial_guess(arm: str) -> np.ndarray:
    """Starting point: nominal mount, zero roll correction, axis at origin."""
    return pack(np.zeros(6), 0.0, np.zeros(2))
