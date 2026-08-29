"""Stage 1: camera intrinsics -- capture rules and the fit.

Holds the acquisition logic (what counts as a usable view) and the fitting of K
and distortion, scored on held-out views. `stage1_web.py` is the interface that
drives it; run that, or:

    python calibration/run.py --stage 1

Why the guidance matters: a stack of centred, frontal views gives a very low
reprojection error and a distortion model that is pure extrapolation at the frame
edges, which is exactly where distortion is largest. `Capture` therefore tracks
coverage and tilt and reports what is still missing.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

STAGES = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGES))

import common  # noqa: E402
from core import charuco, gates, intrinsics, solver, storage  # noqa: E402

ROLES = ("head", "left_wrist", "right_wrist")
TARGET_VIEWS = 30


def camera_label(role: str, mounting: str | None = None) -> str:
    """Name a camera by the side the operator can point at.

    The wrist cameras are bolted to the arms and turn with them, so used
    back-to-front the camera stored as right_wrist is the one on the operator's
    left. The stored role stays the key for the saved intrinsics; only the words
    shown on screen follow the robot.
    """
    if role == "head":
        return "Head camera"
    try:
        import frames
        if mounting is None:
            mounting = frames.declared_mounting()
        side = "left" if frames.named_camera("left", mounting) == role else "right"
    except Exception:  # noqa: BLE001 - a label must never stop the stage
        side = "left" if role == "left_wrist" else "right"
    return f"{side.capitalize()} wrist camera"


class FrameSource:
    """One camera, opened only while it is being calibrated.

    Intrinsics are solved one camera at a time, so only one needs to stream. That
    also keeps the load off the USB bus: these modules draw 500mA each and the
    wrist pair sits behind a hub, where running all three provokes dropouts.
    """

    def __init__(self, role: str):
        self.role = role
        self._cap = None
        self.device: str | None = None
        self.width = self.height = 0

    def open(self) -> None:
        cap, device = common.open_camera(self.role)
        self._cap = cap
        self.device = device
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def check_resolution(self, width: int, height: int) -> None:
        """Intrinsics only apply at the resolution they were solved at."""
        if (self.width, self.height) != (width, height):
            raise RuntimeError(
                f"{camera_label(self.role)} is streaming "
                f"{self.width}x{self.height} but this "
                f"stage expects {width}x{height}. Intrinsics are "
                f"resolution-specific, so these must match.")

    def read(self):
        """Next frame, or None if the read failed."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            common.release_camera(self.role)

    def describe(self) -> str:
        return f"{self.device} at {self.width}x{self.height}"


class Capture:
    """Streams one camera, detects the board, and applies the capture rules.

    Holds a thread rather than subclassing Thread: the natural attribute names
    here (_stop, _started) collide with Thread's own internals.
    """

    def __init__(self, role: str, spec: charuco.BoardSpec, target: int):
        self._thread = threading.Thread(target=self._loop,
                                        name=f"capture-{role}", daemon=True)
        self.role = role
        self.label = camera_label(role)
        self.spec = spec
        self.detector = charuco.BoardDetector(spec, min_corners=gates.PNP_MIN_CORNERS)
        self.target = target

        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stop = threading.Event()
        self._grab = threading.Event()

        self.device: str | None = None
        self.width = self.height = 0
        self.error: str | None = None
        self.source: FrameSource | None = None
        self.guide: intrinsics.CaptureGuide | None = None
        self.stored: list[dict] = []
        self.last_message = "starting"
        self.n_corners = 0
        self.sharpness = 0.0
        self.fps = 0.0
        self.session: storage.CaptureSession | None = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout: float | None = None):
        self._thread.join(timeout)

    def request_grab(self):
        self._grab.set()

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def _loop(self):
        source = FrameSource(self.role)
        try:
            source.open()
        except Exception as exc:
            self.error = str(exc)
            return

        try:
            from config.cameras import HEIGHT, WIDTH

            source.check_resolution(WIDTH, HEIGHT)
        except Exception as exc:
            self.error = str(exc)
            source.close()
            return

        self.source = source
        self.device = source.device
        self.width, self.height = source.width, source.height
        self.guide = intrinsics.CaptureGuide(self.width, self.height, self.target)

        path = storage.session_path("stage1_intrinsics", self.role)
        moved = storage.archive_session(path)
        if moved is not None:
            print(f"  Previous {self.label} capture kept as {moved.name}")
        self.session = storage.CaptureSession(
            path,
            storage.SessionMeta(
                stage="1", purpose="camera intrinsics", camera_role=self.role,
                board_name=self.spec.name, width=self.width, height=self.height,
                notes={"device": source.device, "board": vars(self.spec)}))

        self.last_message = "ready"
        last, frames = time.time(), 0
        try:
            while not self._stop.is_set():
                frame = source.read()
                if frame is None:
                    time.sleep(0.02)
                    continue

                frames += 1
                now = time.time()
                if now - last >= 1.0:
                    self.fps = frames / (now - last)
                    last, frames = now, 0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                detection = self.detector.detect(gray)
                self.n_corners = detection["n"] if detection else 0

                view = None
                if detection is not None:
                    view = intrinsics.describe_view(
                        detection["corners"], self.width, self.height,
                        self.sharpness)
                    obj, img = self.detector.match_points(
                        detection["corners"], detection["ids"])
                    pose = None
                    if obj is not None and len(obj) >= 6:
                        K0 = np.array([[self.width, 0, self.width / 2],
                                       [0, self.width, self.height / 2],
                                       [0, 0, 1]], dtype=float)
                        ok, rvec, tvec = cv2.solvePnP(
                            np.asarray(obj, dtype=float), np.asarray(img, dtype=float),
                            K0, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
                        if ok:
                            R, _ = cv2.Rodrigues(rvec)
                            T = np.eye(4)
                            T[:3, :3], T[:3, 3] = R, tvec.ravel()
                            pose = T
                    intrinsics.with_tilt(view, pose)

                if self._grab.is_set():
                    self._grab.clear()
                    self._try_store(frame, detection, view)

                self._draw(frame, detection, view)
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
        finally:
            source.close()

    def _try_store(self, frame, detection, view) -> None:
        if detection is None or view is None:
            self.last_message = "no board detected"
            return
        accepted, why = self.guide.judge(view, gates.PNP_MIN_CORNERS)
        if not accepted:
            self.last_message = f"rejected: {why}"
            return

        obj, img = self.detector.match_points(detection["corners"], detection["ids"])
        if obj is None or len(obj) < gates.PNP_MIN_CORNERS:
            self.last_message = "rejected: could not match corners to the board"
            return

        self.guide.add(view)
        self.stored.append({"object": obj, "image": img,
                            "corners": detection["corners"],
                            "sharpness": view.sharpness,
                            "tilt_deg": view.tilt_deg})
        self.session.add(frame, detection=detection)
        self.last_message = f"stored view {len(self.stored)}"

    def _draw(self, frame, detection, view) -> None:
        h, w = frame.shape[:2]
        grid = intrinsics.GRID
        covered = self.guide.covered if self.guide else set()

        # Shade cells that still need a corner in them.
        overlay = frame.copy()
        for gx in range(grid):
            for gy in range(grid):
                if (gx, gy) in covered:
                    continue
                x0, y0 = int(gx * w / grid), int(gy * h / grid)
                x1, y1 = int((gx + 1) * w / grid), int((gy + 1) * h / grid)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (40, 40, 130), -1)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        for gx in range(1, grid):
            x = int(gx * w / grid)
            cv2.line(frame, (x, 0), (x, h), (70, 70, 70), 1)
        for gy in range(1, grid):
            y = int(gy * h / grid)
            cv2.line(frame, (0, y), (w, y), (70, 70, 70), 1)

        if detection is not None:
            for (px, py) in detection["corners"]:
                cv2.circle(frame, (int(px), int(py)), 3, (0, 240, 0), -1)

        cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
        colour = (0, 240, 0) if self.n_corners >= gates.PNP_MIN_CORNERS else (0, 170, 240)
        cv2.putText(frame, f"{self.label}  corners {self.n_corners}  "
                           f"sharp {self.sharpness:.0f}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    def stats(self) -> dict:
        guide = self.guide
        return {
            "role": self.role,
            "device": self.device,
            "error": self.error,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 1),
            "corners": self.n_corners,
            "sharpness": round(self.sharpness, 1),
            "stored": len(self.stored),
            "target": self.target,
            "coverage": round(guide.coverage * 100) if guide else 0,
            "tilt": list(guide.tilt_range()) if guide else [0, 0],
            "advice": guide.advice() if guide else "starting",
            "message": self.last_message,
            "ready": len(self.stored) >= gates.INTRINSICS_MIN_VIEWS,
        }

    def drop_last(self) -> bool:
        """Undo the most recent capture.

        Coverage is what the guidance is steering by, so a view stored from a
        blurred or badly placed board has to be removable or it keeps a cell
        marked as covered that really is not.
        """
        if not self.stored:
            return False
        self.stored.pop()
        if self.session is not None:
            self.session.drop_last()
        # The guide has no undo, so rebuild it from the views that remain.
        self.guide = intrinsics.CaptureGuide(self.width, self.height, self.target)
        for entry in self.stored:
            view = intrinsics.describe_view(entry["corners"], self.width,
                                            self.height, entry.get("sharpness", 0.0))
            view.tilt_deg = entry.get("tilt_deg")
            self.guide.add(view)
        self.last_message = f"removed the last view, {len(self.stored)} left"
        return True


def solve_camera(capture: Capture,
                 save: bool = False) -> tuple[dict | None, list[gates.GateResult]]:
    """Fit intrinsics for one camera and grade the outcome.

    Saving is opt-in so that tests, which drive this with synthetic or replayed
    captures, cannot write a result the rest of the calibration would then treat
    as real.
    """
    stored = capture.stored
    n = len(stored)
    checks: list[gates.GateResult] = []

    checks.append(gates.lower_bound(
        "view count", n, gates.INTRINSICS_MIN_VIEWS, warn_at=capture.target))
    if n < 6:
        return None, checks

    all_corners = np.vstack([s["corners"] for s in stored])
    coverage = gates.coverage_fraction(all_corners, capture.width, capture.height)
    checks.append(gates.lower_bound("corner coverage", coverage,
                                    gates.INTRINSICS_MIN_COVERAGE, warn_at=0.75))

    fit_idx, hold_idx = solver.split_holdout(
        n, fraction=0.25, seed=0, minimum=gates.INTRINSICS_MIN_HOLDOUT)

    fit = intrinsics.fit_intrinsics(
        [stored[i]["object"] for i in fit_idx],
        [stored[i]["image"] for i in fit_idx],
        capture.width, capture.height)
    if fit is None:
        checks.append(gates.GateResult("fit", False,
                                       detail="calibrateCamera failed"))
        return None, checks

    hold_rms, _ = intrinsics.holdout_error(
        [stored[i]["object"] for i in hold_idx],
        [stored[i]["image"] for i in hold_idx],
        fit["K"], fit["dist"])

    checks.append(gates.upper_bound("fit RMS", fit["rms"],
                                    gates.INTRINSICS_RMS_MAX_PX, " px",
                                    warn_at=gates.INTRINSICS_RMS_GOOD_PX))
    checks.append(gates.upper_bound("holdout RMS", hold_rms,
                                    gates.INTRINSICS_RMS_MAX_PX, " px",
                                    warn_at=gates.INTRINSICS_RMS_GOOD_PX))
    ratio = hold_rms / fit["rms"] if fit["rms"] > 0 else float("inf")
    checks.append(gates.upper_bound(
        "holdout / fit ratio", ratio, gates.INTRINSICS_HOLDOUT_RATIO_MAX,
        detail="a large gap means the views were too alike"))

    for problem in intrinsics.sanity_check(fit):
        checks.append(gates.GateResult("plausibility", False, detail=problem))

    fov = intrinsics.fov_from_K(fit["K"], capture.width, capture.height)
    result = {
        "camera_role": capture.role,
        "device": capture.device,
        "width": capture.width,
        "height": capture.height,
        "K": fit["K"],
        "dist": fit["dist"],
        "fit_rms_px": fit["rms"],
        "holdout_rms_px": hold_rms,
        "n_views_total": n,
        "n_views_fit": len(fit_idx),
        "n_views_holdout": len(hold_idx),
        "coverage": coverage,
        "fov": fov,
        "board": vars(capture.spec),
        "distortion_model": "rational" if fit["rational"] else "standard 5-term",
    }
    result["gates"] = [{"name": c.name, "passed": c.passed, "value": c.value,
                        "threshold": c.threshold, "detail": c.detail}
                       for c in checks]

    # Only a result that clears every gate is written. Later stages load these
    # without re-checking, so a failed fit left on disk would be used silently.
    if save and all(c.passed for c in checks):
        storage.save_result(f"intrinsics_{capture.role}", result)

    return result, checks


def report_camera(result: dict) -> None:
    fov = result["fov"]
    common.heading(f"{camera_label(result['camera_role'])}: fitted intrinsics")
    print(f"  resolution     {result['width']}x{result['height']}")
    print(f"  fx, fy         {fov['fx']:.2f}, {fov['fy']:.2f}")
    print(f"  cx, cy         {fov['cx']:.2f}, {fov['cy']:.2f}")
    print("  distortion     " + ", ".join(f"{v:+.5f}" for v in result["dist"]))
    print(f"  field of view  {fov['fovx_deg']:.1f} deg horizontal, "
          f"{fov['fovy_deg']:.1f} deg vertical")
    print(f"  fit RMS        {result['fit_rms_px']:.4f} px "
          f"({result['n_views_fit']} views)")
    print(f"  holdout RMS    {result['holdout_rms_px']:.4f} px "
          f"({result['n_views_holdout']} views)")
    print(f"  coverage       {result['coverage'] * 100:.0f}% of the frame")
    if result["camera_role"] == "head":
        print("\n  The horizontal field of view feeds stage 3: it sets how far")
        print("  the head can pan while keeping the board in sight.")



if __name__ == "__main__":
    # The browser interface is the normal way in: one page, one camera at a time,
    # no second terminal. This module keeps the capture rules and the fitting,
    # which that interface imports.
    from stage1_web import main as web_main

    try:
        raise SystemExit(web_main())
    except common.Aborted:
        print("\n  Stopped. Nothing further was saved.")
        raise SystemExit(130)
