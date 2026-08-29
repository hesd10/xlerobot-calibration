"""Stage 2-3: world frame, head mechanism and head camera extrinsics.

    python calibration/run.py --stage 2-3
    then open http://127.0.0.1:8423

Merged from the document's stages 2 and 3, because stage 2 on its own solves
nothing: W = the board frame is a definition, and the only thing it could store is
"I clamped it". The clamping is now a checklist here.

Head postures are set BY HAND. Torque stays off, you move the head, the page
tells you where you have been and where the coverage is thin, and space captures.
Nothing is driven, so the head cannot be commanded into itself or into a cable.

What is solved, and what is not
-------------------------------
12 parameters: T_W^B and the head camera mount. The joint zeros and the pan axis
position are fixed by convention, because a camera watching one fixed board cannot
separate a head rotation from a base rotation. See core/head_model.py; the null
space is five-dimensional and no amount of capture removes it.

Fixing them costs nothing measurable: T_W^B absorbs the offset and predictions are
unaffected to machine precision. Stage 5b later re-defines the zero from the arm
symmetry, in closed form, with no recapture.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

STAGES = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGES))

import common  # noqa: E402
import frames  # noqa: E402
from core import (charuco, gates, head_model, head_solve, servos,  # noqa: E402
                  storage, zeros)

HOST, PORT = "127.0.0.1", 8423

# Where the accuracy curve flattens, measured in check_stage23_synthetic.py:
# 15 views puts T_W^B out by ~4.4 mm, 45 by ~2.0 mm, and beyond that little
# changes. Real captures showed that 20--24 views are also very sensitive to the
# holdout split. Keep the solver's mathematical minimum separate from this more
# conservative capture policy.
TARGET_VIEWS = 45
CAPTURE_MIN_VIEWS = 30
CAPTURE_MIN_PAN_SWEEP_DEG = 55.0
CAPTURE_MIN_TILT_SWEEP_DEG = 30.0
TILT_LAYER_COUNT = 3
CAPTURE_MIN_VIEWS_PER_TILT_LAYER = 8
CAPTURE_MIN_PAN_PER_TILT_LAYER_DEG = 50.0

# Postures nearer than this to one already stored add nothing but time.
MIN_PAN_SEPARATION_DEG = 3.0
MIN_TILT_SEPARATION_DEG = 3.0

# A board this close to the pan axis says nothing about which side it is on, so
# the gate below declines to judge rather than guess. Well outside anything a
# real setup does: the head cannot focus at 50 mm, and every archived capture
# sat between 280 and 500 mm.
BOARD_SIDE_MIN_MM = 50.0


def solve_capture(stored: list[dict],
                  senses: tuple[float, float]) -> tuple[dict | None, list]:
    """Fit the head and world frame from a capture, and grade the result.

    Split out of the session so that an offline replay drives the same code an
    operator does. Reimplementing the conversion below in a test harness proved
    worthless: the harness got it right, so the harness passed, while the stage
    itself could be broken and no test noticed.

    `stored` holds the views as the capture recorded them; `senses` is the
    measured (pan, tilt) direction pair from stage 3.
    """
    # The captured pan is measured from the posture the operator set as the
    # zero, so it is an offset from "facing the board" and says nothing about
    # where the chassis points. The fit needs model angles, and which model
    # angle faces the board is fixed by the mounting alone: chassis-front on a
    # normal robot (q = 0), chassis-back on one used back-to-front (q = pi).
    # Adding it here, at the one place a captured angle becomes a model angle,
    # is what makes T_W_B come out in the model's own frame.
    #
    # Without it a back-to-front robot fits a T_W_B that puts the board in
    # front of the chassis when it is really behind, and every stage reading
    # that frame inherits the half turn -- stage 6 lands the arm roots on each
    # other's sides, which is what new_calibration_3 did.
    seat = head_model.mounting_pan_offset()
    pans = np.array([s["pan_rad"] + seat for s in stored])
    tilts = np.array([s["tilt_rad"] for s in stored])
    observed = [s["T_cam_board"] for s in stored]

    result = head_solve.fit(pans, tilts, observed, senses=senses)
    checks = head_solve.grade(result)
    if result:
        checks.append(_board_side_gate(result["T_W_B"]))
    return result, checks


def _board_side_gate(T_W_B) -> gates.GateResult:
    """Is the board on the side of the chassis the mounting says it must be?

    Every other gate here is a residual, and residuals cannot see this. The
    head mount rotation and the pan zero are two halves of one gauge freedom
    and only their sum is observable, so a frame half a turn from the model's
    fits its own capture exactly as well as the right one. new_calibration_3
    passed every gate in this stage at 2.51 mm holdout while putting the board
    492 mm in FRONT of a chassis that was standing back-to-front. Stage 6 then
    solved both arm roots onto each other's sides, and the fault only surfaced
    at stage 8, three stages and an hour of the operator's time later.

    The board is always physically in front of the operator, because nothing
    would see it otherwise. So which side of the chassis that is depends only
    on how the robot stands: facing it when normal, away from it when used
    back-to-front. That makes the sign checkable here, against the mounting
    alone, with nothing read from another stage's results.
    """
    board = np.linalg.inv(np.asarray(T_W_B, float))[:3, 3]
    forward_mm = frames.forward_of(board) * 1000.0
    mounting = frames.declared_mounting()
    want_front = mounting != frames.FLIPPED

    if abs(forward_mm) < BOARD_SIDE_MIN_MM:
        return gates.GateResult(
            "board side", True, warning=True, value=forward_mm, unit=" mm",
            detail="the board sits almost on the pan axis, so which side it is "
                   "on cannot be told; this is not a normal setup")

    where = "in front of" if forward_mm > 0 else "behind"
    should = "in front of" if want_front else "behind"
    return gates.GateResult(
        "board side", (forward_mm > 0) == want_front,
        value=forward_mm, unit=" mm",
        detail=f"the solved frame puts the board {where} the chassis, but the "
               f"robot is set up as {mounting}, so it must come out {should} "
               f"it. The fit is fine, so this is not a data problem: the frame "
               f"is half a turn from the model's. Check the mounting matches "
               f"how the robot is really standing, and that the head was "
               f"facing the board when its zero was set.")


class PostureGuide:
    """Tracks which head postures have been covered, and what is still missing.

    The pan sweep is what pins the vertical axis, so coverage is judged along pan
    first: a stack of views near one pan angle fits beautifully and determines
    almost nothing. Tilt matters less but still separates the mount rotation.
    """

    def __init__(self, pan_budget_deg: float, bins: int = 7,
                 pan_sense: float = -1.0):
        self.budget = max(pan_budget_deg, 1.0)
        self.bins = bins
        self.pan_sense = float(pan_sense)
        self.pans: list[float] = []
        self.tilts: list[float] = []
        self.corners: list[int] = []

    def side_of_bin(self, index: int) -> str:
        """Which way the operator turns the head to reach this pan band.

        The bands are indexed by encoder travel, and which way that travel
        swings the gaze is a wiring fact -- the pan sense -- not something the
        stage can assume.         Measured against head_model for both senses and both
        mountings: positive travel turns toward the operator's left when the
        sense is -1 and toward their right when it is +1, so the LOW bands lie
        the other way in each case.

        The mounting does NOT enter. Turning the robot around swaps which way
        the model's left points AND puts the seat at q = pi, and those two
        cancel exactly, so the operator's view of the turn is unchanged. This
        was checked rather than assumed; a version that also flipped on the
        mounting would send half the operators the wrong way.
        """
        low = "right" if self.pan_sense < 0 else "left"
        high = "left" if self.pan_sense < 0 else "right"
        return low if index < self.bins // 2 else high

    def bin_of(self, pan_deg: float) -> int:
        """Which pan bin a posture falls in, clamped to the budget."""
        t = (pan_deg + self.budget) / (2 * self.budget)
        return int(np.clip(t * self.bins, 0, self.bins - 1))

    @property
    def covered(self) -> set[int]:
        return {self.bin_of(p) for p in self.pans}

    def counts(self) -> list[int]:
        out = [0] * self.bins
        for p in self.pans:
            out[self.bin_of(p)] += 1
        return out

    def add(self, pan_deg: float, tilt_deg: float, corners: int = 0) -> None:
        self.pans.append(float(pan_deg))
        self.tilts.append(float(tilt_deg))
        self.corners.append(int(corners))

    def thin_views(self) -> int:
        """How many stored views are corner-starved.

        Worth surfacing because a sparse view is not obviously bad on screen:
        reprojection error stays low precisely because there is little to
        disagree with, while the pose it yields is several mm out.
        """
        return sum(1 for c in self.corners if c < gates.PNP_GOOD_CORNERS)

    def too_close(self, pan_deg: float, tilt_deg: float) -> bool:
        """Is this posture a near-duplicate of one already stored?"""
        for p, t in zip(self.pans, self.tilts):
            if (abs(pan_deg - p) < MIN_PAN_SEPARATION_DEG
                    and abs(tilt_deg - t) < MIN_TILT_SEPARATION_DEG):
                return True
        return False

    def sweep(self) -> tuple[float, float]:
        if not self.pans:
            return 0.0, 0.0
        return (max(self.pans) - min(self.pans),
                max(self.tilts) - min(self.tilts) if self.tilts else 0.0)

    def tilt_layers(self) -> list[dict]:
        """Split the measured tilt range into low, middle and high layers."""
        if not self.tilts:
            return []
        low, high = min(self.tilts), max(self.tilts)
        if high - low < CAPTURE_MIN_TILT_SWEEP_DEG:
            return []
        edges = np.linspace(low, high, TILT_LAYER_COUNT + 1)
        layers = []
        for index in range(TILT_LAYER_COUNT):
            members = [i for i, tilt in enumerate(self.tilts)
                       if edges[index] <= tilt <= edges[index + 1]
                       and (index == TILT_LAYER_COUNT - 1
                            or tilt < edges[index + 1])]
            layer_pans = [self.pans[i] for i in members]
            layers.append({
                "name": ("low", "middle", "high")[index],
                "count": len(members),
                "pan_sweep_deg": (max(layer_pans) - min(layer_pans)
                                  if len(layer_pans) >= 2 else 0.0),
            })
        return layers

    def blocking_gaps(self) -> list[str]:
        """The only reasons the solve is refused outright.

        The view count is the one gate the solver genuinely cannot do without:
        below it there is nothing left over for a holdout split, so the fit
        cannot be checked and a reported pass would mean nothing.

        Every other coverage figure is a recommendation. They describe the
        capture that fits best, not the capture that fits at all, and an
        operator working around a fixed board or a restricted mount may not be
        able to reach them. Refusing to solve in that case throws away good
        data; saying the fit is likely to be weak, and letting them look at the
        residuals, does not.
        """
        n = len(self.pans)
        if n < CAPTURE_MIN_VIEWS:
            return [f"need {CAPTURE_MIN_VIEWS - n} more views "
                    f"({n} of {CAPTURE_MIN_VIEWS})"]
        return []

    def advisory_gaps(self) -> list[str]:
        """Coverage that is recommended but not required.

        Shown so the operator can weigh the risk before solving and can read the
        residuals afterwards knowing what was thin.
        """
        pan_sweep, tilt_sweep = self.sweep()
        gaps = []
        if pan_sweep < CAPTURE_MIN_PAN_SWEEP_DEG:
            gaps.append(f"pan sweep is {pan_sweep:.0f} deg; "
                        f"{CAPTURE_MIN_PAN_SWEEP_DEG:.0f} deg is recommended")
        if tilt_sweep < CAPTURE_MIN_TILT_SWEEP_DEG:
            gaps.append(f"tilt sweep is {tilt_sweep:.0f} deg; "
                        f"{CAPTURE_MIN_TILT_SWEEP_DEG:.0f} deg is recommended")
            # The layers are cut from the tilt range, so with the range this
            # narrow they describe nothing worth reporting separately.
            return gaps
        for layer in self.tilt_layers():
            if layer["count"] < CAPTURE_MIN_VIEWS_PER_TILT_LAYER:
                gaps.append(f'{layer["name"]} tilt layer has {layer["count"]} '
                            f'views; {CAPTURE_MIN_VIEWS_PER_TILT_LAYER} is '
                            'recommended')
            elif layer["pan_sweep_deg"] < CAPTURE_MIN_PAN_PER_TILT_LAYER_DEG:
                gaps.append(f'{layer["name"]} tilt layer spans '
                            f'{layer["pan_sweep_deg"]:.0f} deg of pan; '
                            f'{CAPTURE_MIN_PAN_PER_TILT_LAYER_DEG:.0f} deg is '
                            'recommended')
        return gaps

    def capture_gaps(self) -> list[str]:
        """Everything still short of the recommended capture, blocking first."""
        return self.blocking_gaps() + self.advisory_gaps()

    def ready_to_solve(self) -> bool:
        return not self.blocking_gaps()

    def advice(self) -> str:
        n = len(self.pans)
        if n == 0:
            return ("Point the head at the board and hold it steady, then press "
                    "space. Start near the middle.")

        pan_sweep, tilt_sweep = self.sweep()
        counts = self.counts()
        empty = [i for i, c in enumerate(counts) if c == 0]

        # Coaching order is what to reach for next, which is not the same as
        # what blocks the solve: spread is worth more per view than raw count,
        # so it is suggested first even though only the count is a gate.
        if pan_sweep < CAPTURE_MIN_PAN_SWEEP_DEG:
            return (f"Pan sweep is {pan_sweep:.0f} deg, under the "
                    f"{CAPTURE_MIN_PAN_SWEEP_DEG:.0f} deg recommended. Turn the "
                    "head further left and right while keeping the whole board "
                    "visible.")

        if empty:
            side = self.side_of_bin(min(empty))
            return (f"{len(empty)} of {self.bins} pan bands are still empty. "
                    f"Turn the head further {side}.")

        if tilt_sweep < CAPTURE_MIN_TILT_SWEEP_DEG:
            return (f"Pan is well covered. Now add low and high views: tilt sweep "
                    f"is {tilt_sweep:.0f} deg, under the "
                    f"{CAPTURE_MIN_TILT_SWEEP_DEG:.0f} deg recommended.")

        for layer in self.tilt_layers():
            if layer["count"] < CAPTURE_MIN_VIEWS_PER_TILT_LAYER:
                return (f'The {layer["name"]} tilt layer has {layer["count"]} views; '
                        f'{CAPTURE_MIN_VIEWS_PER_TILT_LAYER} is recommended, spread '
                        'from left to right.')
            if layer["pan_sweep_deg"] < CAPTURE_MIN_PAN_PER_TILT_LAYER_DEG:
                return (f'The {layer["name"]} tilt layer spans only '
                        f'{layer["pan_sweep_deg"]:.0f} deg of pan. Add views at both '
                        'left and right edges at this height.')

        if n < CAPTURE_MIN_VIEWS:
            return (f"Coverage looks good. {CAPTURE_MIN_VIEWS - n} more views "
                    "to make the fit and holdout split stable.")

        thin = self.thin_views()
        if thin > n // 3:
            return (f"{thin} of {n} views see fewer than "
                    f"{gates.PNP_GOOD_CORNERS} corners. Those carry several times "
                    f"the pose error of a full view, so bring the board closer or "
                    f"square it up to the lens and capture some more.")

        if n < TARGET_VIEWS:
            return (f"Ready to solve. Another {TARGET_VIEWS - n} well-spread views "
                    "improve robustness; gains flatten beyond that.")

        return "Well covered. Solving now is reasonable."


class Capture:
    """Streams the head camera, reads the head servos, applies the capture rules.

    Holds a thread rather than subclassing Thread: the natural attribute names
    here collide with Thread's internals.
    """

    def __init__(self, spec: charuco.BoardSpec, intrinsics: dict,
                 target: int = TARGET_VIEWS, pan_sense: float = -1.0):
        self._thread = threading.Thread(target=self._loop, name="capture-head",
                                        daemon=True)
        # Only the coaching needs this: which way "turn the head left" moves
        # the encoder depends on how the servo is wired.
        self.pan_sense = float(pan_sense)
        self.spec = spec
        self.detector = charuco.BoardDetector(spec,
                                             min_corners=gates.PNP_MIN_CORNERS)
        self.K = np.array(intrinsics["K"], dtype=float)
        self.dist = np.array(intrinsics["dist"], dtype=float)
        self.fovx = float(intrinsics["fov"]["fovx_deg"])
        self.target = target

        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._halt = threading.Event()
        self._grab = threading.Event()

        self.device: str | None = None
        self.width = self.height = 0
        self.error: str | None = None
        self.robot: servos.RawRobot | None = None
        self.servo_error: str | None = None

        self.guide: PostureGuide | None = None
        self.stored: list[dict] = []
        self.session: storage.CaptureSession | None = None
        self.zero_set = zeros.ZeroSet()

        self.last_message = "starting"
        self.n_corners = 0
        self.fps = 0.0
        # Live state, updated every frame so the page can steer the operator.
        self.pan_deg = self.tilt_deg = 0.0
        self.raw = {"head_motor_1": None, "head_motor_2": None}
        self.servo_sample_ok = False
        self._last_raw: dict[str, int] = {}
        self._continuous_counts: dict[str, float] = {}
        self.distance_mm = 0.0
        self.reproj_px = 0.0
        self.board_visible = False
        self.board_inside = False
        self.pan_budget_deg = 0.0

    def start(self):
        self._thread.start()

    def stop(self):
        self._halt.set()

    def join(self, timeout: float | None = None):
        self._thread.join(timeout)

    def request_grab(self):
        self._grab.set()

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def set_zero(self) -> tuple[bool, str]:
        """Record the current posture as the head's zero.

        The zero is a convention, so 'wherever the head is now' is a legitimate
        choice; what matters is that the count is recorded and travels with the
        T_W^B solved against it.
        """
        if self.robot is None:
            return False, "no servo connection"
        raw = {n: self.robot.read_raw(n) for n in head_model.HEAD_MOTORS}
        if any(v is None for v in raw.values()):
            return False, "could not read the head servos"
        self._last_raw = {}
        self._continuous_counts = {}
        for name, value in raw.items():
            self.zero_set.add(name, int(value), source="mechanical",
                              note="operator set this posture as zero")
            self._last_raw[name] = int(value)
            self._continuous_counts[name] = 0.0
        if self.stored:
            # Angles are measured from the zero, so moving it invalidates them.
            self.stored.clear()
            self.guide = PostureGuide(self.pan_budget_deg, pan_sense=self.pan_sense)
            return True, ("zero set; stored views were discarded because their "
                          "angles were measured from the old zero")
        return True, f"zero set at {raw}"

    def _open_servos(self) -> None:
        try:
            robot = servos.RawRobot()
        except Exception as exc:
            self.servo_error = str(exc)
            return
        problems = [p for p in robot.verify()
                    if any(m in p for m in head_model.HEAD_MOTORS)]
        if problems:
            self.servo_error = "; ".join(problems)
        self.robot = robot

    def _loop(self):
        try:
            cap, device = common.open_camera("head")
        except Exception as exc:
            self.error = str(exc)
            return

        self.device = device
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._open_servos()

        board_w = self.spec.size_mm[0] / 1000.0
        self.pan_budget_deg = head_model.pan_budget(self.fovx, board_w, 0.6)
        self.guide = PostureGuide(self.pan_budget_deg, pan_sense=self.pan_sense)

        self.session = storage.CaptureSession(
            self._session_path(),
            storage.SessionMeta(
                stage="2-3", purpose="world frame and head mechanism",
                camera_role="head", board_name=self.spec.name,
                width=self.width, height=self.height,
                notes={"device": device, "board": vars(self.spec),
                       "fovx_deg": self.fovx}))

        self.last_message = "ready"
        last, frames = time.time(), 0
        try:
            while not self._halt.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                frames += 1
                now = time.time()
                if now - last >= 1.0:
                    self.fps = frames / (now - last)
                    last, frames = now, 0

                self._read_servos()
                detection, pose = self._analyse(frame)
                if self._grab.is_set():
                    self._grab.clear()
                    self._try_store(frame, detection, pose)

                self._draw(frame, detection, pose)
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
        finally:
            cap.release()
            common.release_camera("head")
            if self.robot is not None:
                self.robot.close()

    def _session_path(self) -> Path:
        path = storage.session_path("stage23_head")
        moved = storage.archive_session(path)
        if moved is not None:
            print(f"  Previous capture kept as {moved.name}")
        return path

    def _read_servos(self) -> None:
        self.servo_sample_ok = False
        if self.robot is None:
            return
        current = {name: self.robot.read_raw(name)
                   for name in head_model.HEAD_MOTORS}
        if any(value is None for value in current.values()):
            return
        self.servo_sample_ok = True
        for name, value in current.items():
            self.raw[name] = int(value)

        for name, value in self.raw.items():
            if name not in self.zero_set.joints or value is None:
                continue
            value = int(value)
            if name not in self._last_raw:
                zero = self.zero_set.joints[name].raw
                self._continuous_counts[name] = float(
                    servos.unwrap_delta(value - zero))
            else:
                self._continuous_counts[name] += servos.unwrap_delta(
                    value - self._last_raw[name])
            self._last_raw[name] = value

        scale = 360.0 / servos.COUNTS_PER_TURN
        if "head_motor_1" in self._continuous_counts:
            self.pan_deg = self._continuous_counts["head_motor_1"] * scale
        if "head_motor_2" in self._continuous_counts:
            self.tilt_deg = self._continuous_counts["head_motor_2"] * scale

    def _analyse(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = self.detector.detect(gray)
        self.n_corners = detection["n"] if detection else 0
        self.board_visible = detection is not None
        if detection is None:
            self.board_inside = False
            self.distance_mm = self.reproj_px = 0.0
            return None, None

        pose = self.detector.solve_pose(detection, self.K, self.dist)
        if pose is None:
            self.distance_mm = self.reproj_px = 0.0
            return detection, None

        self.distance_mm = float(np.linalg.norm(pose["tvec"]) * 1000)
        self.reproj_px = self.detector.reprojection_error(
            detection, self.K, self.dist, pose["rvec"], pose["tvec"])
        # Fully inside matters because a board clipped by the frame edge loses the
        # corners that constrain it most.
        pts = np.asarray(detection["corners"], float)
        m = 8
        self.board_inside = bool(
            pts[:, 0].min() > m and pts[:, 1].min() > m
            and pts[:, 0].max() < self.width - m
            and pts[:, 1].max() < self.height - m)
        return detection, pose

    def _try_store(self, frame, detection, pose) -> None:
        if not self.zero_set.joints:
            self.last_message = "set the head zero first"
            return
        if detection is None:
            self.last_message = "rejected: no board detected"
            return
        if pose is None:
            self.last_message = "rejected: could not solve the board pose"
            return
        if not self.board_inside:
            self.last_message = ("rejected: the board touches the frame edge, "
                                 "which loses the corners that constrain it most")
            return
        if self.reproj_px > gates.PNP_MAX_REPROJ_PX:
            self.last_message = (f"rejected: reprojection {self.reproj_px:.2f} px "
                                 f"exceeds {gates.PNP_MAX_REPROJ_PX:.1f}, so the "
                                 f"frame is probably smeared")
            return
        if not self.servo_sample_ok:
            self.last_message = "rejected: could not read both head servos in this frame"
            return
        if self.guide.too_close(self.pan_deg, self.tilt_deg):
            self.last_message = ("rejected: too close to a posture already "
                                 "stored; move the head further")
            return

        self.guide.add(self.pan_deg, self.tilt_deg, int(detection["n"]))
        self.stored.append({
            "pan_rad": float(np.deg2rad(self.pan_deg)),
            "tilt_rad": float(np.deg2rad(self.tilt_deg)),
            "T_cam_board": pose["T_cam_board"],
            "raw": dict(self.raw),
            "n_corners": int(detection["n"]),
            "reproj_px": float(self.reproj_px),
        })
        self.session.add(frame, servos=dict(self.raw), detection=detection,
                         extra={"pan_deg": self.pan_deg,
                                "tilt_deg": self.tilt_deg,
                                "reproj_px": float(self.reproj_px),
                                "distance_mm": self.distance_mm})
        self.last_message = (f"stored view {len(self.stored)} at pan "
                             f"{self.pan_deg:+.1f}, tilt {self.tilt_deg:+.1f}")

    def drop_last(self) -> bool:
        """Undo the most recent capture, freeing the pan band it claimed."""
        if not self.stored:
            return False
        self.stored.pop()
        if self.session is not None:
            self.session.drop_last()
        self.guide = PostureGuide(self.pan_budget_deg, pan_sense=self.pan_sense)
        for entry in self.stored:
            self.guide.add(np.rad2deg(entry["pan_rad"]),
                           np.rad2deg(entry["tilt_rad"]),
                           entry.get("n_corners", 0))
        self.last_message = f"removed the last view, {len(self.stored)} left"
        return True

    def _draw(self, frame, detection, pose) -> None:
        h, w = frame.shape[:2]
        if detection is not None:
            colour = (0, 240, 0) if self.board_inside else (0, 170, 240)
            for (px, py) in detection["corners"]:
                cv2.circle(frame, (int(px), int(py)), 3, colour, -1)

        # The optical axis, at the calibrated principal point rather than the
        # frame centre. On these modules the two differ by 17 and 27 px, which is
        # nearly 4.5 degrees -- aiming by the frame centre would be misleading.
        cx, cy = int(round(self.K[0, 2])), int(round(self.K[1, 2]))
        cv2.line(frame, (cx - 14, cy), (cx + 14, cy), (90, 160, 255), 1)
        cv2.line(frame, (cx, cy - 14), (cx, cy + 14), (90, 160, 255), 1)

        cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
        if self.zero_set.joints:
            label = (f"pan {self.pan_deg:+6.1f}  tilt {self.tilt_deg:+6.1f}  "
                     f"corners {self.n_corners}  {self.distance_mm:.0f}mm")
        else:
            label = "zero not set - press Z with the head where you want zero"
        cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 240, 0) if self.zero_set.joints else (0, 200, 255), 1)

    def stats(self) -> dict:
        guide = self.guide
        pan_sweep, tilt_sweep = guide.sweep() if guide else (0.0, 0.0)
        ready = guide.ready_to_solve() if guide else False
        return {
            "device": self.device,
            "error": self.error,
            "servo_error": self.servo_error,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 1),
            "zero_set": bool(self.zero_set.joints),
            "zeros": self.zero_set.to_dict(),
            "raw": dict(self.raw),
            "pan_deg": round(self.pan_deg, 2),
            "tilt_deg": round(self.tilt_deg, 2),
            "corners": self.n_corners,
            "board_visible": self.board_visible,
            "board_inside": self.board_inside,
            "distance_mm": round(self.distance_mm),
            "reproj_px": round(self.reproj_px, 3),
            "stored": len(self.stored),
            "target": self.target,
            "pan_budget_deg": round(self.pan_budget_deg, 1),
            "pan_sweep_deg": round(pan_sweep, 1),
            "tilt_sweep_deg": round(tilt_sweep, 1),
            "bins": guide.counts() if guide else [],
            "tilt_layers": guide.tilt_layers() if guide else [],
            "advice": guide.advice() if guide else "starting",
            "message": self.last_message,
            "ready": ready,
            "capture_gaps": guide.capture_gaps() if guide else [],
            "advisory_gaps": guide.advisory_gaps() if guide else [],
            "min_views": CAPTURE_MIN_VIEWS,
            "min_sweep_deg": CAPTURE_MIN_PAN_SWEEP_DEG,
        }


# The old stage 2, reduced to what it actually was: confirmations.
CLAMP_CHECKS = [
    ("base", "The robot base is clamped and cannot shift or rotate.",
     "Every result is expressed relative to the board. If the base moves after "
     "this, the calibration is void and nothing downstream can detect it."),
    ("board", "The ChArUco board is fixed and will not be moved.",
     "The board IS the world frame. Moving it later redefines the world."),
    ("torque", "Head servo torque is off, so the head can be moved by hand.",
     "Postures are set by hand in this stage; nothing is driven."),
    ("reach", "The head can see the board across its full comfortable pan range.",
     "The pan sweep is what determines the vertical axis. If the board leaves "
     "the frame early, move the robot closer to square-on before starting."),
]


class Session:
    """The stage's state: clamp checks, then capture, then solve."""

    def __init__(self, spec: charuco.BoardSpec, intrinsics: dict, target: int,
                 senses: tuple[float, float] | None = None):
        self.spec = spec
        self.intrinsics = intrinsics
        self.target = target
        # Measured in stage 2. Carried explicitly rather than read from module
        # state so the value used by a solve is the value that gets recorded.
        self.senses = senses
        self.lock = threading.Lock()
        self.confirmed: set[str] = set()
        self.capture: Capture | None = None
        self.solving = False
        self.last_solve: dict | None = None
        self.last_error: str | None = None

    @property
    def clamped(self) -> bool:
        return len(self.confirmed) == len(CLAMP_CHECKS)

    def confirm(self, key: str, value: bool) -> None:
        with self.lock:
            if value:
                self.confirmed.add(key)
            else:
                self.confirmed.discard(key)

    def start(self) -> tuple[bool, str]:
        with self.lock:
            if not self.clamped:
                return False, "confirm the clamping checklist first"
            if self.solving:
                return False, "still solving"
            if self.capture is not None:
                return True, "already running"

            capture = Capture(
                self.spec, self.intrinsics, self.target,
                pan_sense=(self.senses[0] if self.senses
                           else head_model.PAN_SENSE))
            capture.start()
            for _ in range(80):
                if capture.error or capture.device:
                    break
                time.sleep(0.1)
            if capture.error:
                capture.stop()
                self.last_error = capture.error
                return False, capture.error
            if not capture.device:
                capture.stop()
                self.last_error = "the head camera did not start in time"
                return False, self.last_error

            self.capture = capture
            self.last_error = None
            self.last_solve = None
            return True, f"head camera live on {capture.device}"

    def _stop_locked(self) -> None:
        if self.capture is not None:
            self.capture.stop()
            self.capture.join(timeout=3.0)
            self.capture = None

    def stop(self) -> None:
        with self.lock:
            self._stop_locked()

    def solve(self) -> tuple[bool, str]:
        with self.lock:
            capture = self.capture
            if capture is None:
                return False, "capture is not running"
            if self.solving:
                return False, "already solving"
            if capture.guide is None:
                return False, "capture is still starting"
            blocking = capture.guide.blocking_gaps()
            if blocking:
                return False, "capture is not ready: " + "; ".join(blocking)
            self.solving = True

        def work():
            try:
                summary = self._solve_now(capture)
                with self.lock:
                    self.last_solve = summary
            except Exception as exc:
                # Swallowing the reason would turn a bug here into "the
                # calibration did not pass", sending the operator off to
                # recapture data that was fine.
                import traceback
                detail = traceback.format_exc()
                print(f"\n  Solving raised {type(exc).__name__}: {exc}")
                print("  This is a fault in the tool, not in the captured data.")
                print(detail)
                with self.lock:
                    self.last_solve = {"passed": False, "internal": True,
                                       "stamp": time.time(),
                                       "error": f"{type(exc).__name__}: {exc}"}
            finally:
                with self.lock:
                    self.solving = False

        threading.Thread(target=work, daemon=True).start()
        return True, "solving"

    def _solve_now(self, capture: Capture) -> dict:
        stored = capture.stored
        result, checks = solve_capture(stored, self.senses)
        passed = bool(result) and all(c.passed for c in checks)

        summary = {
            "passed": passed,
            # Identifies this solve, so the page can tell a fresh result from one
            # the operator has already read and dismissed.
            "stamp": time.time(),
            "gates": [{"name": c.name, "passed": c.passed, "line": c.line()}
                      for c in checks],
        }
        if result:
            result["camera_role"] = "head"
            result["device"] = capture.device
            result["board"] = vars(capture.spec)
            result["gauge"] = {
                "pan_zero": "fixed by convention at the recorded posture",
                "tilt_zero": "fixed by convention at the recorded posture",
                "pan_axis_origin_m": head_model.PAN_ORIGIN.tolist(),
                "note": "These are unobservable from one fixed board; T_W_B "
                        "absorbs them. See core/head_model.py.",
            }
            # The zero is stored exactly as the operator set it: the raw count of
            # the posture facing the board. It is not moved to meet the model.
            # The mounting offset lives in the conversion above and in the
            # matching conversion each later stage does for itself, so no reader
            # has to know what this stage decided, and none can double it.
            result["zeros"] = capture.zero_set.to_dict()
            summary.update({k: result[k] for k in (
                "fit_rms_mm", "holdout_rms_mm", "fit_max_mm", "holdout_max_mm",
                "fit_rms_deg", "holdout_rms_deg", "n_views_total",
                "n_views_fit", "n_views_holdout", "pan_sweep_deg",
                "tilt_sweep_deg", "condition_number",
                "camera_from_tilt_joint_mm", "lever_arm_mm",
                "lever_arm_nominal_mm")})
            report(result)
            if passed:
                storage.save_result("head", result)
                if capture.session is not None:
                    capture.session.finish(solved=True,
                                           holdout_rms_mm=result["holdout_rms_mm"])
        return summary

    def status(self) -> dict:
        with self.lock:
            capture = self.capture
            out = {
                "board": {"name": self.spec.name,
                          "squares": f"{self.spec.squares_x}x{self.spec.squares_y}",
                          "square_mm": self.spec.square_mm,
                          "size_mm": list(self.spec.size_mm)},
                "fovx_deg": round(self.intrinsics["fov"]["fovx_deg"], 1),
                "checklist": [{"key": k, "text": t, "why": w,
                               "done": k in self.confirmed}
                              for k, t, w in CLAMP_CHECKS],
                "clamped": self.clamped,
                "running": capture is not None,
                "solving": self.solving,
                "last_solve": self.last_solve,
                "last_error": self.last_error,
                "already_solved": storage.load_result("head") is not None,
            }
        out["capture"] = capture.stats() if capture is not None else None
        return out


def report(result: dict) -> None:
    common.heading("Head mechanism and camera mount")
    T = np.array(result["T_W_B"])
    print(f"  T_W_B translation   {T[0, 3] * 1000:+.1f}, {T[1, 3] * 1000:+.1f}, "
          f"{T[2, 3] * 1000:+.1f} mm")
    cam = result["camera_from_tilt_joint_mm"]
    print(f"  camera from tilt jt {cam[0]:+.1f}, {cam[1]:+.1f}, {cam[2]:+.1f} mm")
    print(f"    lever arm          {result['lever_arm_mm']:.1f} mm, model says "
          f"{result['lever_arm_nominal_mm']:.1f} mm")
    print(f"  views               {result['n_views_total']} "
          f"({result['n_views_fit']} fit, {result['n_views_holdout']} holdout)")
    print(f"  pan sweep           {result['pan_sweep_deg']:.1f} deg total")
    print(f"  tilt sweep          {result['tilt_sweep_deg']:.1f} deg total")
    print(f"  fit error           {result['fit_rms_mm']:.2f} mm rms, "
          f"{result['fit_max_mm']:.2f} mm worst")
    print(f"  holdout error       {result['holdout_rms_mm']:.2f} mm rms, "
          f"{result['holdout_max_mm']:.2f} mm worst")
    print(f"  condition number    {result['condition_number']:.1f}")


PAGE_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Stage: head, world frame and head camera</title>
<style>
  * { box-sizing: border-box; }
  body { background:#14161a; color:#e6e6e6; font-family:system-ui,sans-serif;
         margin:0; padding:20px; }
  h1 { font-size:19px; margin:0 0 3px; }
  h2 { font-size:15px; margin:0 0 8px; }
  .sub { color:#9aa0a6; font-size:13px; margin:0 0 16px; }
  .layout { display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; }
  .panel { background:#1d2025; border:1px solid #2c3038; border-radius:9px;
           padding:14px; }
  .side { width:330px; } .main { flex:1; min-width:680px; }
  img#feed { display:block; width:640px; height:480px; background:#000;
             border-radius:6px; }
  .row { display:flex; gap:10px; margin-top:12px; flex-wrap:wrap; }
  button.act { background:#2f6fb5; border:0; color:#fff; padding:10px 18px;
               border-radius:6px; cursor:pointer; font-size:14px;
               font-family:inherit; }
  button.act:hover:not(:disabled) { background:#3a83d0; }
  button.act:disabled { background:#3a3f48; color:#7b8190; cursor:not-allowed; }
  button.sec { background:#3a3f48; }
  button.sec:hover:not(:disabled) { background:#474d58; }
  button.go { background:#2e7d32; }
  button.go:hover:not(:disabled) { background:#388e3c; }
  .stats { display:grid; grid-template-columns:auto auto; gap:5px 16px;
           font-size:13px; font-variant-numeric:tabular-nums; }
  .stats dt { color:#9aa0a6; }
  .advice { margin-top:12px; padding:10px 12px; background:#20262e;
            border-left:3px solid #2f6fb5; border-radius:4px; font-size:13px;
            line-height:1.5; }
  .msg { margin-top:8px; font-size:12px; color:#9aa0a6; min-height:16px; }
  .msg.bad { color:#e0a0a0; } .msg.good { color:#9fd9a3; }
  .check { display:flex; gap:9px; align-items:flex-start; margin-bottom:11px;
           font-size:13px; line-height:1.45; cursor:pointer; }
  .check input { margin-top:2px; flex:none; width:15px; height:15px;
                 accent-color:#2e7d32; cursor:pointer; }
  .check .why { display:block; color:#7b8190; font-size:11px; margin-top:2px; }
  .bins { display:flex; gap:3px; margin-top:6px; }
  .bins i { flex:1; height:22px; border-radius:3px; background:#24272d;
            display:flex; align-items:center; justify-content:center;
            font-size:10px; font-style:normal; color:#7b8190; }
  .bins i.hit { background:#2b5c33; color:#c8e6c9; }
  .bar { height:7px; background:#24272d; border-radius:4px; overflow:hidden;
         margin-top:5px; }
  .bar i { display:block; height:100%; background:#2f6fb5; width:0; }
  .idle { color:#9aa0a6; font-size:14px; padding:30px 20px; line-height:1.7; }
  table.g { border-collapse:collapse; margin-top:10px; font-size:12px;
            width:100%; }
  table.g td { padding:4px 8px; border-top:1px solid #2c3038; }
  .pass { color:#9fd9a3; } .fail { color:#e0a0a0; }
  .warn { background:#3a2f20; border-left:3px solid #a5651f; padding:9px 12px;
          border-radius:4px; font-size:12px; margin-bottom:14px; line-height:1.5; }
  code { background:#24272d; padding:1px 5px; border-radius:3px; font-size:12px; }
  kbd { background:#2c3038; border:1px solid #3d434d; border-radius:3px;
        padding:1px 5px; font-size:11px; font-family:inherit; }
  .pill { display:inline-block; padding:1px 7px; border-radius:9px;
          font-size:11px; background:#24272d; color:#9aa0a6; }
  .pill.ok { background:#1f2c22; color:#9fd9a3; }
  .pill.no { background:#2e2224; color:#e0a0a0; }
</style></head><body>
<h1>Stage: head, world frame and head camera</h1>
<p class="sub" id="sub">loading...</p>
<div id="pollwarn"></div>
<div id="notes"></div>
<div class="layout">
  <div class="panel side" id="side"></div>
  <div class="panel main" id="main"></div>
</div>
"""

PAGE_SCRIPT = """
<script>
let running = false, pollFails = 0;

// Which solve result the operator has dismissed, so a stale one does not keep
// covering the live view after they go back to capturing.
let solveSeen = null, last = null;

function post(path, body) {
  return fetch(path, {method:'POST',
    headers: body ? {'Content-Type':'application/json'} : {},
    body: body ? JSON.stringify(body) : undefined})
    .then(r => r.json()).catch(() => null);
}

async function tick(key, value) { await post('/confirm', {key, value}); refresh(); }
async function begin()  { const r = await post('/start');
                          if (r && !r.ok) setMsg(r.message, 'bad'); refresh(); }
async function grab()   { await post('/grab'); refresh(); }
async function undo()   { await post('/undo'); refresh(); }
async function setZero(){ const r = await post('/zero');
                          if (r && !r.ok) setMsg(r.message, 'bad'); refresh(); }
async function solve()  { const r = await post('/solve');
                          if (r && !r.ok) setMsg(r.message, 'bad'); refresh(); }
async function halt()   { await post('/stop'); running = false; refresh(); }

function setMsg(text, cls) {
  const el = document.getElementById('msg');
  if (el) { el.textContent = text || ''; el.className = 'msg ' + (cls || ''); }
}

function checklist(s) {
  return `<h2>Clamp everything</h2>
    <p class="sub" style="font-size:12px;margin-bottom:12px">
      The document's stage 2. It solves nothing, so it is a checklist.</p>
    ${s.checklist.map(c => `
      <label class="check">
        <input type="checkbox" ${c.done ? 'checked' : ''}
          ${s.running ? 'disabled' : ''}
          onchange="tick('${c.key}', this.checked)">
        <span>${c.text}<span class="why">${c.why}</span></span>
      </label>`).join('')}
    ${s.running ? '' : `<button class="act go" style="width:100%;margin-top:4px"
      onclick="begin()" ${s.clamped ? '' : 'disabled'}>
      ${s.clamped ? 'Start capture' : 'Confirm all four to start'}</button>`}`;
}

function liveSide(s) {
  const c = s.capture;
  const pct = Math.min(100, 100 * c.stored / Math.max(1, c.target));
  return `<h2>Head posture</h2>
    <dl class="stats">
      <dt>pan</dt><dd>${c.zero_set ? c.pan_deg.toFixed(1) + '&deg;' : '&mdash;'}</dd>
      <dt>tilt</dt><dd>${c.zero_set ? c.tilt_deg.toFixed(1) + '&deg;' : '&mdash;'}</dd>
      <dt>raw counts</dt>
      <dd>${c.raw.head_motor_1 ?? '?'} / ${c.raw.head_motor_2 ?? '?'}</dd>
      <dt>board</dt><dd>${c.board_visible
        ? (c.board_inside ? '<span class="pill ok">fully in frame</span>'
                          : '<span class="pill no">touching the edge</span>')
        : '<span class="pill no">not detected</span>'}</dd>
      <dt>corners</dt><dd>${c.corners}</dd>
      <dt>distance</dt><dd>${c.distance_mm ? c.distance_mm + ' mm' : '&mdash;'}</dd>
      <dt>reprojection</dt><dd>${c.reproj_px ? c.reproj_px.toFixed(2) + ' px' : '&mdash;'}</dd>
      <dt>rate</dt><dd>${c.fps} fps</dd>
    </dl>
    <h2 style="margin-top:16px">Coverage</h2>
    <dl class="stats">
      <dt>views</dt><dd><b>${c.stored}</b> / ${c.target}</dd>
      <dt>pan sweep</dt>
      <dd>${c.pan_sweep_deg.toFixed(0)}&deg; of ${(2*c.pan_budget_deg).toFixed(0)}&deg;
          usable</dd>
      <dt>tilt sweep</dt><dd>${c.tilt_sweep_deg.toFixed(0)}&deg;</dd>
    </dl>
    <div class="bar"><i style="width:${pct}%"></i></div>
    <div class="bins">${c.bins.map(n =>
      `<i class="${n ? 'hit' : ''}">${n || ''}</i>`).join('')}</div>
    <p class="sub" style="font-size:11px;margin:6px 0 0">
      Pan bands, left to right. Empty bands are what the fit lacks.</p>
    ${c.tilt_layers.length ? `<h2 style="margin-top:14px">Tilt layers</h2>
      <dl class="stats">${c.tilt_layers.map(x =>
        `<dt>${x.name}</dt><dd>${x.count} views / ${x.pan_sweep_deg.toFixed(0)}&deg; pan</dd>`
      ).join('')}</dl>` : ''}`;
}

function gateTable(g) {
  if (!g || !g.length) return '';
  return '<table class="g">' + g.map(x =>
    `<tr><td class="${x.passed ? 'pass' : 'fail'}">${x.passed ? 'PASS' : 'FAIL'}</td>
     <td>${x.line}</td></tr>`).join('') + '</table>';
}

function solveView(s) {
  const r = s.last_solve;
  if (r.error) {
    return `<h2 class="fail">Solve failed</h2><p class="fail">${r.error}</p>
      ${r.internal ? `<p class="sub">This is a fault in the tool, not in what you
      captured. The views are still on disk; nothing needs recapturing. The
      terminal has the traceback.</p>` : ''}
      ${dismissButton(s)}`;
  }
  return `<h2>${r.passed ? '<span class="pass">Passed</span>'
                         : '<span class="fail">Did not pass</span>'}</h2>
    <dl class="stats">
      <dt>views</dt><dd>${r.n_views_total}
        (${r.n_views_fit} fit, ${r.n_views_holdout} holdout)</dd>
      <dt>holdout error</dt><dd>${r.holdout_rms_mm.toFixed(2)} mm rms,
        ${r.holdout_max_mm.toFixed(2)} mm worst</dd>
      <dt>fit error</dt><dd>${r.fit_rms_mm.toFixed(2)} mm rms</dd>
      <dt>pan sweep</dt><dd>${r.pan_sweep_deg.toFixed(0)}&deg; total</dd>
      <dt>camera from tilt joint</dt>
      <dd>${r.camera_from_tilt_joint_mm.map(v => v.toFixed(1)).join(', ')} mm
        (${r.lever_arm_mm.toFixed(1)} mm out, model says
        ${r.lever_arm_nominal_mm.toFixed(1)})</dd>
      <dt>conditioning</dt><dd>${r.condition_number.toFixed(1)}</dd>
    </dl>
    ${gateTable(r.gates)}
    <p class="sub" style="margin-top:12px">${r.passed
      ? 'Saved. Next: <code>python calibration/run.py --stage 5</code>'
      : 'Not saved. Capture more views, or check the gate that failed.'}</p>
    ${dismissButton(s)}`;
}

function dismissButton(s) {
  // Only offered while capture is still up, since that is the only case where
  // there is a live view to go back to.
  if (!s.capture) return '';
  return `<div class="row" style="margin-top:12px">
    <button class="act sec" onclick="dismiss()">Back to capturing</button></div>`;
}

function dismiss() {
  solveSeen = (last && last.last_solve) ? last.last_solve.stamp : null;
  refresh();
}

function liveMain(s) {
  const c = s.capture;
  const enough = c.ready;
  // Solving with the recommended coverage unmet is allowed, so the button says
  // what will happen rather than refusing. The warning is what makes that an
  // informed choice instead of a silent downgrade.
  const advisories = c.advisory_gaps || [];
  const solveLabel = !enough ? `${c.min_views - c.stored} more views to solve`
    : advisories.length ? 'Solve anyway' : 'Solve';
  return `<img id="feed" src="/feed?t=${Date.now()}">
    <div class="row">
      <button class="act" onclick="grab()" ${c.zero_set ? '' : 'disabled'}>
        Capture <kbd>space</kbd></button>
      <button class="act sec" onclick="setZero()">Set zero here <kbd>z</kbd></button>
      <button class="act sec" onclick="undo()" ${c.stored ? '' : 'disabled'}>
        Undo last</button>
      <button class="act go" onclick="solve()" ${enough ? '' : 'disabled'}>
        ${solveLabel}</button>
      <button class="act sec" onclick="halt()">Stop</button>
    </div>
    <div class="msg" id="msg">${c.message || ''}</div>
    <div class="advice">${c.advice}</div>
    <div id="advisories">${advisoryNote(c)}</div>`;
}

function advisoryNote(c) {
  const gaps = c.advisory_gaps || [];
  if (!c.ready || !gaps.length) return '';
  return `<div class="warn">You have enough views to solve. These are short of
    the recommended coverage, which usually means a weaker fit rather than a
    wrong one:<ul>${gaps.map(g => `<li>${g}</li>`).join('')}</ul>
    You can solve now and judge it by the residuals, or capture more first.</div>`;
}

function notes(s) {
  let out = '';
  if (!s.running && s.already_solved) {
    out += `<div class="warn">This stage has already been solved. Capturing
      again and solving will replace that result. Anything downstream that was
      solved against it stays valid only if the head zero is unchanged.</div>`;
  }
  const c = s.capture;
  if (c && c.servo_error) {
    out += `<div class="warn">Head servos: ${c.servo_error}<br>
      Postures are read from the encoders, so this must be working before any
      view is worth storing. Check the bus and that torque is off.</div>`;
  }
  if (c && !c.zero_set) {
    out += `<div class="warn">No head zero recorded yet. Put the head where you
      want "zero" to mean &mdash; roughly level and facing forward &mdash; then
      press <kbd>z</kbd>. The exact posture does not matter: it is a convention,
      and stage 7, body frame and zero conventions, refines it later from the
      arm symmetry. Angles cannot be measured until it is set.</div>`;
  }
  return out;
}

async function refresh() {
  let s;
  try {
    const r = await fetch('/status');
    if (!r.ok) throw new Error('status ' + r.status);
    s = await r.json();
    last = s;
    pollFails = 0;
    document.getElementById('pollwarn').innerHTML = '';
  } catch (e) {
    if (++pollFails >= 4) {
      document.getElementById('pollwarn').innerHTML =
        `<div class="warn">Lost contact with the server (${e.message}).
         What you see below is stale. The terminal has the reason.</div>`;
    }
    return;
  }

  document.getElementById('sub').textContent =
    `board ${s.board.name}: ${s.board.squares} squares, ` +
    `${s.board.size_mm[0].toFixed(0)}x${s.board.size_mm[1].toFixed(0)} mm, ` +
    `head camera ${s.fovx_deg} deg horizontal`;

  const set = (id, html) => {
    const el = document.getElementById(id);
    if (el && el.dataset.html !== html) { el.innerHTML = html; el.dataset.html = html; }
  };
  set('notes', notes(s));

  const side = document.getElementById('side');
  const main = document.getElementById('main');

  if (s.solving) {
    set('side', checklist(s));
    set('main', `<div class="idle"><h2>Solving...</h2>
      <p>Fitting 12 parameters and scoring them on views held out of the fit.</p>
      </div>`);
    running = false;
    solveSeen = null;
  } else if (s.last_solve && solveSeen !== s.last_solve.stamp) {
    // A finished solve outranks the live view. Capture keeps running, so
    // without this the page would fall through to the branch below and the
    // result would never be shown at all.
    set('side', checklist(s));
    set('main', solveView(s));
    running = false;
  } else if (s.capture) {
    // Rebuilding every poll would restart the stream and eat clicks, so the
    // live panels are built once and then updated in place.
    if (!running) {
      side.innerHTML = liveSide(s); side.dataset.html = '';
      main.innerHTML = liveMain(s); main.dataset.html = '';
      running = true;
    } else {
      side.innerHTML = liveSide(s);
      updateMain(s);
    }
  } else {
    running = false;
    set('side', checklist(s));
    if (s.last_solve) set('main', solveView(s));
    else set('main', `<div class="idle">
      <h2>How this works</h2>
      <p>You move the head by hand; nothing is driven. The page shows where you
      have been and where the coverage is thin.</p>
      <p>Twelve parameters are solved: where the robot is in the world, and where
      the camera sits on the head. The joint zeros and the pan axis are fixed by
      convention &mdash; a camera watching one fixed board cannot tell a head
      rotation from a base rotation, and no amount of capture changes that.
      Fixing them costs no accuracy.</p>
      <p>Confirm the checklist on the left to begin.</p></div>`);
  }
}

function updateMain(s) {
  const c = s.capture;
  const msg = document.getElementById('msg');
  if (msg) {
    msg.textContent = c.message || '';
    msg.className = 'msg ' + (/reject|fail|could not/i.test(c.message || '')
      ? 'bad' : /stored|removed|zero set/i.test(c.message || '') ? 'good' : '');
  }
  const adv = document.querySelector('#main .advice');
  if (adv) adv.textContent = c.advice;
  const btns = document.querySelectorAll('#main button.act');
  if (btns.length >= 4) {
    btns[0].disabled = !c.zero_set;
    btns[2].disabled = !c.stored;
    btns[3].disabled = !c.ready;
    const advisories = c.advisory_gaps || [];
    btns[3].textContent = !c.ready ? `${c.min_views - c.stored} more views to solve`
      : advisories.length ? 'Solve anyway' : 'Solve';
  }
  const adv2 = document.getElementById('advisories');
  if (adv2) adv2.innerHTML = advisoryNote(c);
}

document.addEventListener('keydown', e => {
  if (!running) return;
  if (e.code === 'Space') { e.preventDefault(); grab(); }
  if (e.code === 'KeyZ') { e.preventDefault(); setZero(); }
});
setInterval(refresh, 500);
refresh();
</script></body></html>
"""


def page() -> str:
    return PAGE_HEAD + PAGE_SCRIPT


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    session: Session = None

    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type="text/html; charset=utf-8"):
        payload = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload, code=200):
        # Transforms come back as ndarrays. Without the converter this raises
        # inside the handler, /status stops answering, and the page freezes on
        # whatever it last saw -- which looks like a hang, not an error.
        body = json.dumps(payload, default=storage.json_default)
        self._send(code, body, "application/json")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, page())
        elif path == "/status":
            self._json(self.session.status())
        elif path == "/feed":
            self._stream()
        else:
            self._send(404, "not found")

    def do_POST(self):
        path = self.path.split("?")[0]
        capture = self.session.capture

        if path == "/confirm":
            body = self._body()
            self.session.confirm(str(body.get("key")), bool(body.get("value")))
            self._json({"ok": True})
        elif path == "/start":
            ok, message = self.session.start()
            self._json({"ok": ok, "message": message})
        elif path == "/grab":
            if capture is None:
                self._json({"ok": False, "message": "not running"}, 409)
                return
            capture.request_grab()
            self._json({"ok": True})
        elif path == "/undo":
            if capture is None:
                self._json({"ok": False, "message": "not running"}, 409)
                return
            self._json({"ok": capture.drop_last()})
        elif path == "/zero":
            if capture is None:
                self._json({"ok": False, "message": "not running"}, 409)
                return
            ok, message = capture.set_zero()
            self._json({"ok": ok, "message": message})
        elif path == "/solve":
            ok, message = self.session.solve()
            self._json({"ok": ok, "message": message})
        elif path == "/stop":
            self.session.stop()
            self._json({"ok": True})
        else:
            self._send(404, "not found")

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                capture = self.session.capture
                if capture is None:
                    break
                jpeg = capture.snapshot()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpeg)).encode()
                                 + b"\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2-3: world frame and head mechanism")
    parser.add_argument("--target", type=int, default=TARGET_VIEWS,
                        help="views to aim for")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    try:
        common.require_results("board", "intrinsics_head", "senses")
        spec = common.load_board()
    except common.Aborted:
        return 1

    intrinsics = storage.load_result("intrinsics_head")

    # From stage 2, never assumed. Both senses fit this data to 3.6mm, so getting
    # it from a measurement rather than a default is the whole point of stage 2.
    senses = head_model.load_senses()

    common.heading("Stage 3: world frame, head mechanism and head camera")
    print(f"  Board: {spec.name}, {spec.squares_x}x{spec.squares_y} squares, "
          f"{spec.size_mm[0]:.0f}x{spec.size_mm[1]:.0f} mm")
    print(f"  Head camera: {intrinsics['fov']['fovx_deg']:.1f} deg horizontal, "
          f"holdout {intrinsics['holdout_rms_px']:.3f} px")
    print(f"  Joint senses: pan {senses[0]:+.0f}, tilt {senses[1]:+.0f} "
          f"(measured in stage 2)")

    budget = head_model.pan_budget(intrinsics["fov"]["fovx_deg"],
                                   spec.size_mm[0] / 1000.0, 0.6)
    print(f"  At 60 cm that leaves about +/-{budget:.0f} deg of pan with the "
          f"board staying in frame.")
    print(f"  The gate wants {gates.HEAD_PAN_SWEEP_MIN_DEG:.0f} deg of total "
          f"sweep, so this is comfortable."
          if budget * 2 > gates.HEAD_PAN_SWEEP_MIN_DEG else
          f"  That is under the {gates.HEAD_PAN_SWEEP_MIN_DEG:.0f} deg the gate "
          f"wants; move the board further away.")

    print("\n  Solving 12 parameters: T_W_B and the head camera mount.")
    print("  Joint zeros and the pan axis are fixed by convention: a camera")
    print("  watching one fixed board cannot separate a head rotation from a")
    print("  base rotation. This costs no accuracy; see core/head_model.py.")
    print("\n  Head postures are set BY HAND. Leave torque off.")

    if storage.load_result("head"):
        print("\n  NOTE: this stage already has a result. Solving again "
              "replaces it.")

    session = Session(spec, intrinsics, args.target, senses)
    Handler.session = session
    try:
        server = Server((HOST, args.port), Handler)
    except OSError as exc:
        print(f"\n  Cannot listen on {HOST}:{args.port}: {exc}")
        print(f"  Another copy may already be running. Open "
              f"http://{HOST}:{args.port}")
        return 1

    print(f"\n  Open http://{HOST}:{args.port}")
    print("  Ctrl-C here when you are done.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping...")
    finally:
        session.stop()
        server.shutdown()
        server.server_close()

    # Show what happened in THIS session, not just what's on disk
    result = storage.load_result("head")
    common.heading("Stage 2-3 summary")
    
    # Check if this session produced a successful solve
    if hasattr(session, 'last_solve') and session.last_solve:
        summary = session.last_solve
        if summary.get("passed"):
            print(f"  ✓ This session: Solved and saved")
            print(f"    holdout {summary['holdout_rms_mm']:.2f} mm, "
                  f"pan sweep {summary['pan_sweep_deg']:.0f} deg")
            print("  Next: python calibration/run.py --stage 5")
            return 0
        else:
            print(f"  ✗ This session: Solve attempted but FAILED gates, not saved")
            print(f"    lever {summary.get('lever_arm_mm', '?'):.1f} mm, "
                  f"holdout {summary.get('holdout_rms_mm', '?'):.2f} mm")
            if result:
                print(f"  Previous calibration on disk: "
                      f"holdout {result['holdout_rms_mm']:.2f} mm (still valid)")
            else:
                print(f"  No previous calibration on disk.")
            print("  Tip: Collect more views or check the failed gates.")
            return 1
    
    # No solve attempted this session, check disk
    if result:
        print(f"  Previous calibration on disk: "
              f"holdout {result['holdout_rms_mm']:.2f} mm, "
              f"pan sweep {result['pan_sweep_deg']:.0f} deg")
        print("  Next: python calibration/run.py --stage 5")
        return 0
    print("  Not solved yet. Rerun this stage when ready.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except common.Aborted:
        print("\n  Stopped. Nothing further was saved.")
        raise SystemExit(130)
