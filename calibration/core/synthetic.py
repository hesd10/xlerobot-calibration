"""Synthetic camera for testing, rendering a ChArUco board through a known model.

Used to verify calibration code against ground truth, which a real capture cannot
do. Lens distortion is applied per pixel via an inverse map: a homography alone
cannot represent it, because board -> ideal image is projective but ideal ->
distorted image is not. Getting that wrong makes a correct calibration look
broken, so check_renderer.py verifies this module against cv2.projectPoints.
"""

from __future__ import annotations

import cv2
import numpy as np

from core import charuco

WIDTH, HEIGHT = 640, 480

# A plausible cheap wide-angle module at 640x480.
K_TRUE = np.array([[505.0, 0.0, 318.0],
                   [0.0, 503.0, 244.0],
                   [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.35, 0.14, 0.0010, -0.0007, -0.03])

MARGIN_MM = 8.0
# High enough that the board image is oversampled at every distance used.
DPI = 400
# Render this many times larger, then average down. Point-sampling a sharp
# chessboard aliases badly, which biases detected corners by a few tenths of a
# pixel and would show up as a fake calibration error.
SUPERSAMPLE = 3


class SyntheticCamera:
    """Renders board views through a fixed, known camera."""

    def __init__(self, spec: charuco.BoardSpec | None = None,
                 width: int = WIDTH, height: int = HEIGHT,
                 K: np.ndarray | None = None, dist: np.ndarray | None = None):
        self.spec = spec or charuco.BoardSpec(
            squares_x=7, squares_y=5, square_mm=34.9, marker_mm=26.1,
            name="synthetic", measured=True)
        self.width, self.height = width, height
        self.K = K_TRUE.copy() if K is None else np.asarray(K, dtype=float)
        self.dist = DIST_TRUE.copy() if dist is None else np.asarray(dist, float)

        self.board_img = charuco.render(self.spec, dpi=DPI, margin_mm=MARGIN_MM)
        bh_px, bw_px = self.board_img.shape[:2]
        bw_mm, bh_mm = self.spec.size_mm
        self._mm_per_px_x = (bw_mm + 2 * MARGIN_MM) / bw_px
        self._mm_per_px_y = (bh_mm + 2 * MARGIN_MM) / bh_px

        # Supersampled destination grid. Sample at sub-pixel centres so that
        # averaging the block down reproduces a proper box filter.
        s = SUPERSAMPLE
        self._ss = s
        self._sw, self._sh = width * s, height * s
        vv, uu = np.mgrid[0:self._sh, 0:self._sw].astype(np.float64)
        # Map supersampled index -> real pixel coordinate.
        uu = (uu + 0.5) / s - 0.5
        vv = (vv + 0.5) / s - 0.5
        dest = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64)
        dest = dest.reshape(-1, 1, 2)
        # Undistort the destination grid: this applies the lens model per pixel,
        # which a single homography cannot do.
        self._ideal = cv2.undistortPoints(dest, self.K, self.dist,
                                          P=self.K).reshape(-1, 2)
        self._ideal_h = np.column_stack([self._ideal, np.ones(len(self._ideal))])

    def object_points(self) -> np.ndarray:
        return self.spec.corner_positions()

    def render(self, rvec, tvec) -> np.ndarray:
        """Image of the board at the given pose, as BGR."""
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))
        t = np.asarray(tvec, dtype=float).reshape(3)
        # Board z=0, so board (X, Y) -> ideal pixels is exactly K [r1 r2 t].
        Hm = self.K @ np.column_stack([R[:, 0], R[:, 1], t])
        try:
            Hinv = np.linalg.inv(Hm)
        except np.linalg.LinAlgError:
            return np.full((self.height, self.width, 3), 120, np.uint8)

        board = (Hinv @ self._ideal_h.T).T
        w = board[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            board = board[:, :2] / w
        # Behind the camera or on the vanishing line: no valid preimage.
        bad = ~np.isfinite(board).all(axis=1) | (w[:, 0] <= 0)
        board[bad] = -1e6

        map_x = ((board[:, 0] * 1000.0 + MARGIN_MM) / self._mm_per_px_x)
        map_y = ((board[:, 1] * 1000.0 + MARGIN_MM) / self._mm_per_px_y)
        big = cv2.remap(
            self.board_img,
            map_x.reshape(self._sh, self._sw).astype(np.float32),
            map_y.reshape(self._sh, self._sw).astype(np.float32),
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=120)
        # Box-average each supersample block down to one output pixel.
        s = self._ss
        scene = big.reshape(self.height, s, self.width, s).mean(axis=(1, 3))
        return cv2.cvtColor(scene.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def project(self, rvec, tvec) -> np.ndarray:
        """Where the board corners should land, per cv2.projectPoints."""
        pts, _ = cv2.projectPoints(self.object_points(), np.asarray(rvec, float),
                                   np.asarray(tvec, float), self.K, self.dist)
        return pts.reshape(-1, 2)

    def fov(self) -> dict[str, float]:
        from core import intrinsics
        return intrinsics.fov_from_K(self.K, self.width, self.height)
