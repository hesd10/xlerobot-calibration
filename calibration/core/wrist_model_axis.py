"""Wrist camera + roll axis calibration for stage 6 (extended model).

Per arm, solves 10 parameters:
  - T_wrist_cam (6 DoF): camera mount relative to Fixed_Jaw body
  - wrist_roll_zero (1): correction to stage 4's rough wrist_roll zero
  - roll_axis_direction (2): unit vector in Wrist_Pitch_Roll frame (θ, φ)
  - roll_axis_offset (1): translation along the roll axis (gauge, fixed by constraint)

This extends wrist_model.py to calibrate the roll joint axis itself, not assume
it from XML. The physical roll axis direction and position are solved from data.

Gauge freedom
-------------
Three coupled gauge freedoms:
  1. wrist_roll_zero <-> camera roll about the roll axis
  2. roll_axis_offset <-> camera translation along the roll axis
  3. roll_axis_direction <-> subtle coupling with mount orientation

The first two are exact 1D gauge orbits. We project them out during optimization,
then fix them by physical constraints:
  - Optical axis horizontal at XML zero (same as wrist_model.py)
  - Roll axis passes through a reference point (e.g., the nominal XML axis origin)

Forward kinematics
------------------
World → Body → Arm_root → ... → Wrist_Pitch_Roll → [roll transform] → Camera

We compute FK up to Wrist_Pitch_Roll using MuJoCo, then manually apply the
parameterized roll transform:
  T_parent_jaw = Rot(axis, roll_angle) @ Trans(axis, offset)
  T_W_cam = T_W_B @ T_B_A @ T_base_parent @ T_parent_jaw @ T_wrist_cam
"""

from __future__ import annotations

import numpy as np

from . import se3

# Parameter layout:
#   [0:6]   T_wrist_cam as se3 6-vector (local perturbation of nominal)
#   [6]     wrist_roll_zero correction (radians)
#   [7:9]   roll_axis_direction as (theta, phi) spherical coords in parent frame
#   [9]     roll_axis_offset along the axis (meters, gauge fixed by constraint)
N_PARAMS = 10
MOUNT_BLOCK = slice(0, 6)
ROLL_ZERO_INDEX = 6
AXIS_THETA_INDEX = 7
AXIS_PHI_INDEX = 8
AXIS_OFFSET_INDEX = 9

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


def spherical_to_cart(theta: float, phi: float) -> np.ndarray:
    """Spherical (θ, φ) -> unit vector in Cartesian."""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def cart_to_spherical(v: np.ndarray) -> tuple[float, float]:
    """Unit vector -> (θ, φ). θ ∈ [0, π], φ ∈ [-π, π]."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    theta = np.arccos(np.clip(v[2], -1, 1))
    phi = np.arctan2(v[1], v[0])
    return theta, phi


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


def nominal_roll_axis(arm: str) -> tuple[np.ndarray, float]:
    """Nominal roll axis: direction [0, +1, 0], offset 0.
    
    Physical measurement shows the roll axis is +Y in parent frame, not -Y as
    originally stated in XML. Returns (axis_direction, axis_offset) in parent
    (Wrist_Pitch_Roll) frame.
    """
    return np.array([0.0, 1.0, 0.0]), 0.0


def pack(xi: np.ndarray, roll_zero: float, axis_theta: float, axis_phi: float,
         axis_offset: float) -> np.ndarray:
    """Pack parameters into the 10-vector."""
    p = np.zeros(N_PARAMS)
    p[MOUNT_BLOCK] = np.asarray(xi, float).reshape(6)
    p[ROLL_ZERO_INDEX] = float(roll_zero)
    p[AXIS_THETA_INDEX] = float(axis_theta)
    p[AXIS_PHI_INDEX] = float(axis_phi)
    p[AXIS_OFFSET_INDEX] = float(axis_offset)
    return p


def unpack(p: np.ndarray, arm: str) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Unpack the 10-vector -> (T_wrist_cam, roll_zero, axis_dir, axis_offset)."""
    p = np.asarray(p, float).reshape(-1)
    if p.size != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} parameters, got {p.size}")
    
    T_wrist_cam = nominal_mount(arm) @ se3.exp_se3(p[MOUNT_BLOCK])
    roll_zero = float(p[ROLL_ZERO_INDEX])
    axis_dir = spherical_to_cart(p[AXIS_THETA_INDEX], p[AXIS_PHI_INDEX])
    axis_offset = float(p[AXIS_OFFSET_INDEX])
    
    return T_wrist_cam, roll_zero, axis_dir, axis_offset


def roll_transform(axis: np.ndarray, angle: float, offset: float) -> np.ndarray:
    """SE(3) transform for rotation `angle` about `axis` through origin, plus
    translation `offset` along the axis.
    
    Screw: (axis, axis × 0 + offset * axis) = (axis, offset * axis).
    """
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    # Rodrigues for rotation
    K = se3.skew(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    # Translation: along the axis
    t = offset * axis
    return se3.make_transform(R, t)


def camera_in_world(
    sim,
    arm: str,
    angles: dict[str, float],
    T_wrist_cam: np.ndarray,
    roll_axis: np.ndarray,
    roll_offset: float,
    T_W_B: np.ndarray,
    T_B_A: np.ndarray,
) -> np.ndarray:
    """Predicted camera pose using the parameterized roll axis.
    
    FK chain: World → Body → Arm_root → ... → Parent → [Roll] → Camera.
    """
    # FK up to the roll joint's parent body (Wrist_Pitch_Roll)
    parent_body = PARENT_BODIES[arm]
    sim.set_joints(angles)
    parent_pos, parent_rot = sim.body_pose_in_chassis(parent_body)
    T_base_parent = se3.make_transform(parent_rot, parent_pos)
    
    # Apply the parameterized roll transform
    roll_motor = ARM_JOINT_NAMES[arm][4]
    roll_angle = angles[roll_motor]
    T_parent_jaw = roll_transform(roll_axis, roll_angle, roll_offset)
    
    # Compose full chain
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
    reported_angles: list[dict[str, float]],
    observed_poses: list[np.ndarray],
    rough_roll_zero: float,
    T_W_B: np.ndarray,
    T_B_A: np.ndarray,
) -> np.ndarray:
    """Pose error per view."""
    T_wrist_cam, roll_zero_correction, axis_dir, axis_offset = unpack(p, arm)
    roll_motor = ARM_JOINT_NAMES[arm][4]
    
    out = []
    for angles_dict, obs_pose in zip(reported_angles, observed_poses):
        true_angles = dict(angles_dict)
        true_angles[roll_motor] = angles_dict[roll_motor] + roll_zero_correction
        
        T_W_cam_pred = camera_in_world(sim, arm, true_angles, T_wrist_cam,
                                        axis_dir, axis_offset, T_W_B, T_B_A)
        T_W_cam_obs = se3.invert(np.asarray(obs_pose, float))
        out.append(se3.log_se3(se3.invert(T_W_cam_obs) @ T_W_cam_pred))
    
    return np.concatenate(out) if out else np.zeros(0)


def initial_guess(arm: str) -> np.ndarray:
    """Starting point: nominal mount, zero roll correction, XML axis."""
    nom_axis, nom_offset = nominal_roll_axis(arm)
    theta, phi = cart_to_spherical(nom_axis)
    return pack(np.zeros(6), 0.0, theta, phi, nom_offset)
