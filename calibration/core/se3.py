"""SE(3) helpers for calibration, parameterised by Lie algebra exponential maps.

Every unknown rigid transform in the calibration is carried as a 6-vector
xi = (rotation_vector, translation) and mapped to a 4x4 matrix by exp(). That
keeps the optimiser working in an unconstrained space, so it can never produce a
non-orthogonal rotation, and avoids gimbal lock and quaternion normalisation.

Conventions
-----------
  - A transform T maps points from its source frame into its target frame:
    p_target = T @ p_source. Named T_target_source in code, matching the
    document's T_W^B notation (target W, source B).
  - Rotation vectors follow the right-hand rule, magnitude in radians.
  - The rotation block comes first in xi, matching Ceres and Sophus.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix, so skew(a) @ b == cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def unskew(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rotation vector -> rotation matrix (Rodrigues)."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1e-8:
        # Second-order expansion; exact enough well past float precision here.
        return np.eye(3) + K + 0.5 * (K @ K)
    return (np.eye(3)
            + (np.sin(theta) / theta) * K
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K))


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector, with |result| always in [0, pi].

    Goes via a quaternion using Shepperd's method, which picks whichever branch
    has the largest denominator. Recovering the axis directly from the
    antisymmetric part of R degrades as the angle approaches pi, where that part
    vanishes; the quaternion route stays well conditioned for every rotation.
    """
    R = np.asarray(R, dtype=float)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qv = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qv = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qv = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qv = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])

    # q and -q are the same rotation; the positive-qw branch gives theta <= pi.
    if qw < 0.0:
        qw, qv = -qw, -qv

    sin_half = float(np.linalg.norm(qv))
    if sin_half < 1e-12:
        return np.zeros(3)
    theta = 2.0 * float(np.arctan2(sin_half, qw))
    return qv * (theta / sin_half)


def _left_jacobian_inv_factor(theta: float) -> tuple[float, float]:
    """Coefficients for the inverse left Jacobian of SO(3)."""
    if theta < 1e-8:
        return 0.5, 1.0 / 12.0
    half = theta / 2.0
    return 0.5, (1.0 - (theta * np.cos(half)) / (2.0 * np.sin(half))) / (theta * theta)


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """(rotvec, translation) -> 4x4 homogeneous transform.

    Uses the true SE(3) exponential, so the translation passes through the left
    Jacobian V rather than being copied verbatim. That makes exp/log exact
    inverses, which matters when the optimiser takes large steps.
    """
    xi = np.asarray(xi, dtype=float).reshape(6)
    w, u = xi[:3], xi[3:]
    theta = float(np.linalg.norm(w))
    R = exp_so3(w)
    K = skew(w)

    if theta < 1e-8:
        V = np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    else:
        V = (np.eye(3)
             + ((1.0 - np.cos(theta)) / (theta * theta)) * K
             + ((theta - np.sin(theta)) / (theta ** 3)) * (K @ K))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ u
    return T


def log_se3(T: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform -> (rotvec, translation)."""
    T = np.asarray(T, dtype=float)
    w = log_so3(T[:3, :3])
    theta = float(np.linalg.norm(w))
    K = skew(w)
    a, b = _left_jacobian_inv_factor(theta)
    V_inv = np.eye(3) - a * K + b * (K @ K)
    return np.concatenate([w, V_inv @ T[:3, 3]])


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """Rigid inverse, cheaper and better conditioned than a general inverse."""
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform a single point (3,) or a stack of points (N, 3)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        return T[:3, :3] @ pts + T[:3, 3]
    return pts @ T[:3, :3].T + T[:3, 3]


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Discrepancy between two poses as (translation metres, rotation radians)."""
    delta = invert(T_a) @ T_b
    return (float(np.linalg.norm(delta[:3, 3])),
            float(np.linalg.norm(log_so3(delta[:3, :3]))))


def rotvec_between(axis_from: np.ndarray, axis_to: np.ndarray) -> np.ndarray:
    """Shortest rotation vector taking one unit axis onto another."""
    a = np.asarray(axis_from, float) / max(np.linalg.norm(axis_from), EPS)
    b = np.asarray(axis_to, float) / max(np.linalg.norm(axis_to), EPS)
    c = float(np.clip(a @ b, -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.zeros(3)
    if c < -1.0 + 1e-12:
        # Antiparallel: any perpendicular axis works, pick a stable one.
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        return axis / max(np.linalg.norm(axis), EPS) * np.pi
    axis = np.cross(a, b)
    return axis / max(np.linalg.norm(axis), EPS) * float(np.arccos(c))
