"""Intrinsics fitting and capture guidance.

Split out from the stage script so the decision logic can be tested without a
camera. Two jobs:

  - decide whether a candidate view is worth keeping, and tell the operator what
    is still missing. Blurred frames and near-duplicate poses add nothing, and a
    stack of centred frontal views produces a low reprojection error with a
    useless distortion model.
  - fit K and distortion, then score the result on views held out of the fit.

OpenCV 5 removed aruco.calibrateCameraCharuco, so the route is
CharucoDetector.detectBoard -> board.matchImagePoints -> cv2.calibrateCamera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# A view is rejected as a near-duplicate if its board centre and apparent size
# are both close to one already kept. Distortion needs variety, not repetition.
DUPLICATE_CENTRE_PX = 40.0
DUPLICATE_SIZE_RATIO = 0.12

# Laplacian variance below this is too soft to trust corner positions.
MIN_SHARPNESS = 40.0

# Grid used both for coverage scoring and for telling the operator where to move.
GRID = 4


@dataclass
class ViewStats:
    """What one candidate view contributes."""

    n_corners: int
    sharpness: float
    centre: tuple[float, float]
    extent: float
    cells: set[tuple[int, int]] = field(default_factory=set)
    tilt_deg: float | None = None


def describe_view(corners: np.ndarray, width: int, height: int,
                  sharpness: float, grid: int = GRID) -> ViewStats:
    """Summarise a detection: where the board is, how big, which cells it covers."""
    pts = np.asarray(corners, dtype=float).reshape(-1, 2)
    centre = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    # Mean distance from centre stands in for apparent size; robust to a few
    # missing corners in a way that a bounding box is not.
    extent = float(np.mean(np.linalg.norm(pts - np.array(centre), axis=1)))
    gx = np.clip((pts[:, 0] / width * grid).astype(int), 0, grid - 1)
    gy = np.clip((pts[:, 1] / height * grid).astype(int), 0, grid - 1)
    cells = set(zip(gx.tolist(), gy.tolist()))
    return ViewStats(len(pts), float(sharpness), centre, extent, cells)


def with_tilt(view: ViewStats, T_cam_board: np.ndarray | None) -> ViewStats:
    """Attach pose-derived obliqueness when a provisional pose is available."""
    if T_cam_board is not None:
        view.tilt_deg = tilt_from_pose(T_cam_board)
    return view


def tilt_from_pose(T_cam_board: np.ndarray) -> float:
    """Angle between the board normal and the camera axis, in degrees.

    Oblique views are what make the focal length separable from the distance;
    purely frontal views leave that direction weakly constrained.
    """
    # Board normal is its local +z, expressed in camera coordinates.
    normal = T_cam_board[:3, 2]
    cos = abs(float(normal[2])) / max(float(np.linalg.norm(normal)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class CaptureGuide:
    """Tracks what has been collected and what is still needed."""

    def __init__(self, width: int, height: int, target: int = 30,
                 grid: int = GRID, min_sharpness: float = MIN_SHARPNESS):
        self.width = width
        self.height = height
        self.target = target
        self.grid = grid
        self.min_sharpness = min_sharpness
        self.kept: list[ViewStats] = []

    @property
    def covered(self) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for view in self.kept:
            out |= view.cells
        return out

    @property
    def coverage(self) -> float:
        return len(self.covered) / float(self.grid * self.grid)

    def missing_cells(self) -> list[tuple[int, int]]:
        all_cells = {(x, y) for x in range(self.grid) for y in range(self.grid)}
        return sorted(all_cells - self.covered)

    def tilt_range(self) -> tuple[float, float]:
        tilts = [v.tilt_deg for v in self.kept if v.tilt_deg is not None]
        return (min(tilts), max(tilts)) if tilts else (0.0, 0.0)

    def is_duplicate(self, view: ViewStats) -> bool:
        for kept in self.kept:
            dc = np.hypot(view.centre[0] - kept.centre[0],
                          view.centre[1] - kept.centre[1])
            if dc > DUPLICATE_CENTRE_PX:
                continue
            ratio = abs(view.extent - kept.extent) / max(kept.extent, 1e-6)
            if ratio < DUPLICATE_SIZE_RATIO:
                return True
        return False

    def judge(self, view: ViewStats, min_corners: int) -> tuple[bool, str]:
        """Accept or reject a candidate, with a reason the operator can act on."""
        if view.n_corners < min_corners:
            return False, f"only {view.n_corners} corners visible"
        if view.sharpness < self.min_sharpness:
            return False, f"too blurred (sharpness {view.sharpness:.0f})"
        if self.is_duplicate(view):
            return False, "too similar to a view already captured"
        return True, "accepted"

    def add(self, view: ViewStats) -> None:
        self.kept.append(view)

    def advice(self) -> str:
        """The single most useful thing for the operator to do next."""
        if len(self.kept) == 0:
            return "Hold the board so the whole of it is visible, then keep still."

        missing = self.missing_cells()
        if missing:
            names = {(0, 0): "top-left", (self.grid - 1, 0): "top-right",
                     (0, self.grid - 1): "bottom-left",
                     (self.grid - 1, self.grid - 1): "bottom-right"}
            for cell in missing:
                if cell in names:
                    return f"Move the board toward the {names[cell]} of the frame."
            gx, gy = missing[0]
            horizontal = "left" if gx < self.grid / 2 else "right"
            vertical = "upper" if gy < self.grid / 2 else "lower"
            return f"Move the board toward the {vertical} {horizontal} area."

        lo, hi = self.tilt_range()
        if hi < 25.0:
            return "Now tilt the board: aim for 30 to 45 degrees, both directions."

        extents = [v.extent for v in self.kept]
        if max(extents) / max(min(extents), 1e-6) < 1.4:
            return "Vary the distance: take some views closer and some further."

        if len(self.kept) < self.target:
            return f"Good coverage. Keep going to {self.target} views."
        return "Coverage looks complete. You can finish."

    def status(self) -> str:
        lo, hi = self.tilt_range()
        return (f"{len(self.kept)}/{self.target} views, "
                f"coverage {self.coverage * 100:.0f}%, "
                f"tilt {lo:.0f}-{hi:.0f} deg")


def fit_intrinsics(object_points: list[np.ndarray], image_points: list[np.ndarray],
                   width: int, height: int, rational: bool = False):
    """Fit K and distortion. Returns a dict, or None if the solve fails."""
    if len(object_points) < 3:
        return None

    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    obj = [np.asarray(o, dtype=np.float32).reshape(-1, 1, 3) for o in object_points]
    img = [np.asarray(i, dtype=np.float32).reshape(-1, 1, 2) for i in image_points]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, (width, height), None, None, flags=flags)

    return {
        "rms": float(rms),
        "K": np.asarray(K, dtype=float),
        "dist": np.asarray(dist, dtype=float).ravel(),
        "rvecs": [np.asarray(r, dtype=float).ravel() for r in rvecs],
        "tvecs": [np.asarray(t, dtype=float).ravel() for t in tvecs],
        "width": int(width),
        "height": int(height),
        "rational": bool(rational),
        "n_views": len(obj),
    }


def per_view_error(object_points, image_points, K, dist, rvec, tvec) -> float:
    """RMS reprojection error for one view, in pixels."""
    proj, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
        rvec, tvec, K, dist)
    diff = proj.reshape(-1, 2) - np.asarray(image_points, dtype=float).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def holdout_error(object_points, image_points, K, dist) -> tuple[float, list[float]]:
    """Reprojection error on views that took no part in the fit.

    Each held-out view gets its own pose from PnP, then the error is measured with
    the fitted K. This is the number that reflects how the camera model will
    actually behave, unlike the fit RMS which always looks better.
    """
    errors = []
    for obj, img in zip(object_points, image_points):
        obj_a = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
        img_a = np.asarray(img, dtype=np.float64).reshape(-1, 2)
        if len(obj_a) < 4:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj_a, img_a, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        errors.append(per_view_error(obj_a, img_a, K, dist, rvec, tvec))
    if not errors:
        return float("nan"), []
    return float(np.sqrt(np.mean(np.square(errors)))), errors


def fov_from_K(K: np.ndarray, width: int, height: int) -> dict[str, float]:
    """Field of view implied by a fitted K.

    Stage 3 needs this to work out how far the head can pan while keeping the
    board in view, so it is a real output of this stage rather than a curiosity.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    return {
        "fovx_deg": float(np.degrees(2 * np.arctan(width / (2 * fx)))),
        "fovy_deg": float(np.degrees(2 * np.arctan(height / (2 * fy)))),
        "fx": fx, "fy": fy,
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
    }


def sanity_check(result: dict) -> list[str]:
    """Flag physically implausible intrinsics that still fit the data."""
    problems = []
    K, w, h = result["K"], result["width"], result["height"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if not (0.2 * w < fx < 8.0 * w):
        problems.append(f"fx {fx:.1f} is implausible for a {w}px wide image")
    if abs(fx - fy) / max(fx, fy) > 0.15:
        problems.append(f"fx {fx:.1f} and fy {fy:.1f} differ by more than 15%, "
                        f"unusual for a square-pixel sensor")
    if abs(cx - w / 2) > 0.25 * w or abs(cy - h / 2) > 0.25 * h:
        problems.append(f"principal point ({cx:.0f}, {cy:.0f}) is far from the "
                        f"image centre ({w / 2:.0f}, {h / 2:.0f})")
    k1 = result["dist"][0] if len(result["dist"]) else 0.0
    if abs(k1) > 2.0:
        problems.append(f"k1 {k1:.3f} is extreme; check the board geometry")
    return problems
