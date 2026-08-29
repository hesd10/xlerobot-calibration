"""Stage 5 Fusion: arm zeros + mounting + camera mount from wrist views.

Replaces the contact-based stage 5 with a vision-only approach that uses the
wrist camera to observe the ChArUco board from varied arm poses. This breaks
the wrist_flex ↔ touch_point degeneracy that plagued contact calibration.

What this solves (per arm)
---------------------------
- Arm mounting T_B_A (5 DoF, yaw held by gauge)
- 4 joint zeros: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex
- Wrist camera mount T_wrist_cam (6 DoF)

Total: 15 parameters per arm, from 12-15 diverse wrist camera views.

Collection guidance
-------------------
Two requirements matter more than the rest, and for different reasons.

**wrist_flex diversity**: sweep -60° to +60° with at least 4 distinct angles.
This is what breaks the wrist_flex ↔ touch_point degeneracy.

**wrist_roll spread**: at least 60°, ideally 90°. The optimiser holds the roll
zero fixed, which is why the joint was once left out of the coverage bars
altogether -- but holding a zero is not the same as not needing the motion. The
roll turns the camera, and its spread is what separates the camera mount's
orientation from the arm's. Without it the solve is refused; see
WRIST_ROLL_SPAN_DEG for the measurements behind the number.

Also vary camera height (lift/elbow) and lateral reach (pan). The UI shows
real-time coverage bars for all joints and warns when a new view is too similar
to existing ones.

Run:  python calibration/run.py --stage 5f
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CALIB = os.path.dirname(_HERE)
if _CALIB not in sys.path:
    sys.path.insert(0, _CALIB)

import frames  # noqa: E402
import model_map  # noqa: E402
from core import (arm_fusion_solve, arm_model, charuco, gates,  # noqa: E402
                  ranges, se3, senses as senses_mod, servos, storage,
                  zeros as zeros_mod)
from stages import common  # noqa: E402


MIN_VIEWS = 10
TARGET_VIEWS = 14
MAX_CONDITION_NUMBER = 1e6
MAX_RMS_MM = 8.0
HOST, PORT = "0.0.0.0", 5005

# How far wrist_roll must be spread across the views.
#
# The optimiser holds the wrist_roll zero fixed, so it was left out of the
# coverage bars -- but the roll still turns the camera, and it is what
# separates the camera mount's orientation from the arm's. Without spread the
# mount can rotate about the roll axis and be paid for elsewhere, so those
# directions stop being identifiable.
#
# Measured on the four archived workspaces. Holding the view count fixed at 10
# and choosing the widest- and narrowest-spread subsets of the same capture --
# so only the spread differs -- the condition number rose on all 8 arms without
# exception:
#
#   span 162.7 -> 30.6 deg   cond   62.6 ->  123.5
#   span 146.2 -> 55.5 deg   cond   61.6 ->  174.8
#   span 105.9 -> 19.7 deg   cond   64.1 ->  237.0
#   span  96.1 -> 37.2 deg   cond   83.6 ->   97.9   (and four more alike)
#
# Grouped by spread: above 70 deg the condition number stays under 90, between
# 40 and 70 its median is 82, and below 40 it is 120 with a worst case of 237.
# Every arm that solved well in practice spread the roll by at least 61 deg.
#
# The hold-out RMS does eventually notice, but only faintly and not in a fixed
# direction, because a narrow roll set is easier to fit as well as less
# informative -- one collapsed case fitted to 1.36 mm while holding out at
# 11.15 mm, with a condition number of 6966. Waiting for that means the
# operator learns after capturing everything. The bar tells them during.
WRIST_ROLL_SPAN_DEG = 60.0
WRIST_ROLL_TARGET_DEG = 90.0

# How many corners a view should carry, and how many views should carry that
# many. Advice only: nothing below refuses a capture or a solve.
#
# The fit solves 15 parameters per arm. Views cost seconds, so operators stop
# near MIN_VIEWS, and the two arms of new_calibration_flipped were captured
# with 11 and 10. Both passed every gate -- roll spread 91 and 107 deg,
# condition number 53 and 80, reprojection 0.13-0.56 px throughout -- and both
# still put their wrist camera over the limit at stage 8 (right 10.92 mm, left
# 8.82 mm, against an 8 mm gate).
#
# Re-running the real fit over that capture says the shortfall is the capture
# size and the corner counts, not any one bad view:
#
#   * Resampling the right arm's split 40 times gives a hold-out median of
#     6.92 mm ranging 1.72-11.90 mm. The single number the stage prints is one
#     draw of a seed=0 split, so it is mostly luck.
#   * The learning curve is still falling where the data ends: fitting on 8
#     views holds out at 7.23 mm, on 9 at 6.48 mm. Nothing suggests 10 or 11
#     is where it flattens.
#   * Leave-one-out clears every individual view. Dropping the worst still
#     leaves the rest fitting at 2.4-3.8 mm.
#   * Corner count is what separates them. Across both arms, views under 40
#     corners held out at a median 9.85 mm against 3.93 mm for the rest
#     (r = -0.41), and a third of the capture was under 40 -- the sparsest had
#     19, barely above the 12 that PnP itself demands.
#
# So RICH_VIEW_CORNERS is set at the 40 the measurement points to, and the
# targets ask for a capture where most views clear it. These are suggestions
# because the evidence is a correlation over one workspace: a sparse view is
# worth less, not worthless, and an operator who cannot get closer to the board
# is better off with the view than without it.
RICH_VIEW_CORNERS = 40
RICH_VIEW_TARGET_FRACTION = 0.7


class Refused(RuntimeError):
    """A gate turned the capture down, and its message is for the operator.

    Separated from the unexpected failures so the page can show the wording as
    written. A refusal is a finished answer -- too few views, too little roll,
    an arm reaching the wrong way -- and appending a traceback to it only
    buries the sentence that tells the operator what to do next.
    """


def wrist_roll_span_deg(arm: str, views: list[dict]) -> float:
    """How widely a set of views spreads the wrist roll, in degrees."""
    jn = f"{arm}_wrist_roll"
    values = [view["angles"][jn] for view in views
              if jn in view.get("angles", {})]
    if len(values) < 2:
        return 0.0
    return float(np.rad2deg(max(values) - min(values)))


def rich_view_count(views: list[dict]) -> int:
    """How many views carry enough corners to be worth their place in the fit.

    Counting rather than averaging on purpose: a couple of very close views
    would lift a mean while leaving the sparse ones just as uninformative, and
    it is the sparse ones that the hold-out error tracks.
    """
    return sum(1 for view in views
               if int(view.get("n_corners", 0)) >= RICH_VIEW_CORNERS)


def capture_advice(views: list[dict]) -> str:
    """What would most improve this capture, or "" if nothing stands out.

    Advice, never a refusal. It is returned to the page and shown beside the
    coverage bars; no caller gates on it. The stage's actual refusals live in
    validate_result and wrist_roll_complaint, and this deliberately stays out
    of their way -- it speaks only about how much was captured and how well the
    board filled the frame, which are the operator's to judge against the room
    they have.
    """
    if not views:
        return ""
    rich = rich_view_count(views)
    wanted = int(len(views) * RICH_VIEW_TARGET_FRACTION)

    if len(views) < TARGET_VIEWS:
        return (f"{len(views)} views so far. The solve fits 15 parameters per "
                f"arm, and on a capture this size its error was still falling "
                f"as views were added, so {TARGET_VIEWS} or more is worth the "
                f"extra minute.")
    if rich < wanted:
        return (f"{len(views) - rich} of {len(views)} views show fewer than "
                f"{RICH_VIEW_CORNERS} corners. Views that sparse carried "
                f"roughly 2.5x the error of the rest, so filling more of the "
                f"frame with the board on the next few would help.")
    return ""


def wrist_roll_complaint(arm: str, views: list[dict]) -> str:
    """Why these views cannot support a solve yet, or "" if they can."""
    span = wrist_roll_span_deg(arm, views)
    if span >= WRIST_ROLL_SPAN_DEG:
        return ""
    return (
        f"The wrist roll only varies by {span:.0f} deg across these views; "
        f"at least {WRIST_ROLL_SPAN_DEG:.0f} deg is needed "
        f"({WRIST_ROLL_TARGET_DEG:.0f} is better).\n"
        f"  The roll turns the camera, so without spread the camera mount's "
        f"orientation cannot be told apart from the arm's, and the solve "
        f"has no way to separate them.\n"
        f"  Rotate the gripper about its own axis between views and take a "
        f"few more. Nothing already captured is lost.")


def check_the_arm_reaches_outward(sim, arm: str, result: dict) -> None:
    """At its own zero the arm must reach away from the centreline.

    Every residual gate is blind to this fault. The mount yaw and the
    shoulder-pan zero are two halves of one gauge freedom, and only their sum
    is observable, so a frame half a turn out splits that sum as (mount yawed
    180) + (pan zero at 180). The two cancel in every prediction: the fit is
    just as good, and the arm root even stays on its correct side, because
    turning about the pan axis moves the root hardly at all.

    What does move is the tip. A half-turned arm reaches across the
    centreline, which is the export where both arms fold into each other. So
    the tip is the one thing worth testing here.
    """
    T_B_A = np.asarray(result["T_B_A"], float)
    sim.set_joints({})
    tip, _ = sim.body_pose_in_chassis(model_map.TIP_BODIES[arm])
    tip = (T_B_A @ np.append(np.asarray(tip, float), 1.0))[:3]

    if not frames.is_on_expected_side(arm, tip):
        mounting = frames.declared_mounting()
        side = frames.physical_side(arm, mounting)
        raise Refused(
            f"At its zero pose the {side} arm reaches across the robot's "
            f"centreline ({frames.lateral(tip) * 1000:+.1f} mm toward the "
            f"robot's left), so the two arms would fold into each other.\n"
            f"The fit itself is fine, so this is not a data problem: the "
            f"base frame is half a turn from the model's.\n"
            f"Check that the mounting is set to how the robot is really "
            f"standing (currently {mounting}), then redo the head stage "
            f"with the head facing the board.")


def validate_result(sim, arm: str, result: dict) -> None:
    """Every gate the stage puts between a converged solve and a saved one."""
    if not result.get("success"):
        raise Refused(
            f"The solver did not converge: {result.get('message', 'unknown')}")
    if not np.isfinite(result["condition_number"]):
        raise Refused("The condition number is invalid; the data still "
                           "has unidentifiable directions")
    if result["condition_number"] > MAX_CONDITION_NUMBER:
        raise Refused(
            f"Condition number {result['condition_number']:.1e} is too high; "
            f"vary the joint poses more")
    if result["holdout_rms_mm"] > MAX_RMS_MM:
        raise Refused(
            f"Hold-out RMS {result['holdout_rms_mm']:.2f} mm is too high; "
            f"check for blurred or bad views")
    if result["holdout_rms_deg"] > 3.0:
        raise Refused(
            f"Hold-out rotation RMS {result['holdout_rms_deg']:.2f} deg is "
            f"too high; vary the poses more")
    check_the_arm_reaches_outward(sim, arm, result)


def corrected_zeros(result: dict, rough_zeros: dict[str, int],
                    senses: dict[str, int]) -> dict[str, int]:
    """Stage 4's rough zeros with this stage's corrections written over them.

    Split out because the correction is what stage 7 actually reads, and an
    offline replay that skips it normalises a mount solved against corrected
    zeros while holding the uncorrected ones. The two disagree by a few
    degrees per joint -- enough to put the wrist cameras 20 mm out at stage 8.
    """
    out = dict(rough_zeros)
    for joint_name, correction_deg in result["zeros_deg"].items():
        out[joint_name] = servos.zero_with_angle_correction(
            rough_zeros[joint_name], np.deg2rad(correction_deg),
            senses[joint_name])
    return out


def solve_arm(sim, arm: str, views: list[dict], intrinsics: dict,
              T_W_B, rough_zeros: dict[str, int],
              senses: dict[str, int]) -> tuple[dict, dict[str, int]]:
    """Fit one arm from its captured views, gates and all.

    The whole path from views to a result the stage would accept: view count,
    roll spread, the solve, and every gate. Callers that are not the live
    session -- the offline replay, above all -- go through this so that what
    they exercise is the stage rather than a second copy of its arithmetic.

    Raises Refused, carrying the operator-facing wording, when a gate turns
    the capture down. Returns the solved result and the zeros stage 7 sees.
    """
    if len(views) < MIN_VIEWS:
        raise Refused(f"At least {MIN_VIEWS} views are required; "
                           f"there are {len(views)}")

    # Refused before the solve rather than after, because a narrow roll set
    # fits BETTER while meaning less: the collapsed case measured on
    # new_calibration_4 fitted to 1.36 mm and held out at 11.15 mm. The
    # residual gates cannot be trusted to catch what geometry never
    # constrained.
    complaint = wrist_roll_complaint(arm, views)
    if complaint:
        raise Refused(complaint)

    result = arm_fusion_solve.fit(sim, arm, views, intrinsics, T_W_B,
                                  rough_zeros, senses)
    validate_result(sim, arm, result)
    return result, corrected_zeros(result, rough_zeros, senses)


class ArmSession:
    """One arm's capture state: views taken, live pose, solve result."""

    def __init__(self, arm: str, spec, intrinsics: dict, sim, robot,
                 rough_zeros: dict[str, int], senses: dict[str, int],
                 measured_ranges: ranges.RangeSet,
                 T_W_B: np.ndarray, bus_lock: threading.Lock):
        self.arm = arm
        self.spec = spec
        self.K = np.array(intrinsics["K"], float)
        self.dist = np.array(intrinsics["dist"], float)
        self.intrinsics_width = int(intrinsics["width"])
        self.intrinsics_height = int(intrinsics["height"])
        self.sim = sim
        self.robot = robot
        self.rough_zeros = rough_zeros
        self.senses = senses
        self.measured_ranges = measured_ranges
        for name in rough_zeros:
            if (name not in measured_ranges.travels
                    or name not in measured_ranges.zero_raw):
                raise RuntimeError(f"Stage 4 has no continuous travel range for {name}")
            if measured_ranges.travels[name].span_counts >= servos.COUNTS_PER_TURN:
                raise RuntimeError(f"{name} travels more than one full turn, so the encoder branch is not unique")
        self.T_W_B = np.asarray(T_W_B, float)

        self.taken: list[dict] = []
        self._lock = threading.RLock()
        self._bus_lock = bus_lock
        self._angle_tracker = ranges.RangeAngleTracker(
            self.rough_zeros, self.measured_ranges)
        self._last_sample_time = 0.0

        # This stage solves 15 parameters per arm from a dozen or so views, and
        # until now kept none of what it saw: the frames and corner pixels were
        # discarded the moment PnP had run, leaving only the 4x4 poses in
        # touch.json. That is enough to re-run the fit, but not to ask whether a
        # view was motion-blurred, whether the corners were badly distributed,
        # or what the poses become under corrected intrinsics -- the questions
        # that actually come up. Answering them meant an hour of re-capture with
        # the robot. The head and validation stages have recorded this all
        # along; this brings the arm stage in line with them.
        path = storage.session_path("stage6_arms", arm)
        storage.archive_session(path)
        self.session = storage.CaptureSession(path, storage.SessionMeta(
            stage="6", purpose="arm mount, joint zeros and wrist camera",
            camera_role=f"{arm.split('_')[0]}_wrist",
            board_name=getattr(spec, "name", None),
            width=self.intrinsics_width, height=self.intrinsics_height,
            notes={"arm": arm}))

        self.solving = False
        self.solve_result: dict | None = None
        self.last_error: str | None = None

    def sample_joints(self) -> tuple[dict[str, int], dict[str, float], float]:
        """Read and continuously unwrap all arm joints in one serialised sample."""
        with self._bus_lock:
            raw = {n: self.robot.read_raw(n) for n in self.rough_zeros}
        missing = [n for n, value in raw.items() if value is None]
        if missing:
            raise RuntimeError(f"Cannot read motors: {', '.join(missing)}")

        now = time.monotonic()
        with self._lock:
            try:
                self._angle_tracker.update(raw)
            except ValueError as exc:
                raise RuntimeError(f"Cannot localise the current motor readings: {exc}") from exc
            angles = self._angle_tracker.angles(self.senses)
            self._last_sample_time = now
        return {n: int(v) for n, v in raw.items()}, angles, now

    def _joint_coverage(self) -> dict[str, dict]:
        # wrist_roll is here even though the optimiser holds its zero: the
        # joint still turns the camera, and its spread is what makes the
        # camera mount identifiable. See WRIST_ROLL_SPAN_DEG.
        jnames = [f"{self.arm}_{j}" for j in
                  ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                   "wrist_roll"]]
        coverage = {jn: {"min": None, "max": None, "count": 0, "values": []}
                    for jn in jnames}

        for view in self.taken:
            angles = view.get("angles", {})
            for jn in jnames:
                if jn in angles:
                    val = angles[jn]
                    coverage[jn]["values"].append(val)
                    coverage[jn]["count"] += 1
                    if coverage[jn]["min"] is None:
                        coverage[jn]["min"] = coverage[jn]["max"] = val
                    else:
                        coverage[jn]["min"] = min(coverage[jn]["min"], val)
                        coverage[jn]["max"] = max(coverage[jn]["max"], val)

        return coverage

    def _wrist_roll_span_deg(self) -> float:
        """How widely the captured views spread the wrist roll, in degrees."""
        return wrist_roll_span_deg(self.arm, self.taken)

    def _wrist_roll_complaint(self) -> str:
        """Why this capture cannot support a solve yet, or "" if it can."""
        return wrist_roll_complaint(self.arm, self.taken)

    def _camera_height_span(self) -> float:
        if len(self.taken) < 2:
            return 0.0
        zs = []
        for view in self.taken:
            T_cam_board = np.asarray(view["T_cam_board"], float).reshape(4, 4)
            T_W_cam = se3.invert(T_cam_board)
            zs.append(T_W_cam[2, 3])
        return float(max(zs) - min(zs))

    def _duplicate_view(self, T_cam_board: np.ndarray) -> bool:
        T_new = np.asarray(T_cam_board, float).reshape(4, 4)
        p_new = T_new[:3, 3]
        R_new = T_new[:3, :3]

        for view in self.taken:
            T_old = np.asarray(view["T_cam_board"], float).reshape(4, 4)
            trans_diff = np.linalg.norm(p_new - T_old[:3, 3])
            R_rel = R_new.T @ T_old[:3, :3]
            ang = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
            if trans_diff < 0.03 and ang < np.deg2rad(5):
                return True
        return False

    def capture(self, pose: dict | None) -> dict:
        if pose is None:
            return {"ok": False, "error": "No board detected; aim at the ChArUco board"}

        pose_time = float(pose.get("timestamp", 0.0))
        if pose_time <= 0.0 or time.monotonic() - pose_time > 1.0:
            return {"ok": False, "error": "The board pose is stale; keep the board visible and retry"}
        if int(pose.get("n_corners", 0)) < 12:
            return {"ok": False, "error": "Fewer than 12 corners detected; bring more of the board into view"}
        if float(pose.get("reproj_px", float("inf"))) > 1.5:
            return {"ok": False, "error": "Reprojection error is too high; hold the arm steady and retry"}

        try:
            raw = {n: int(v) for n, v in pose["raw"].items()}
            angles = {n: float(v) for n, v in pose["angles"].items()}
            sample_time = float(pose["sample_time"])
            if abs(sample_time - pose_time) > 0.1:
                return {"ok": False, "error": "The image and motor timestamps are out of sync; hold the arm steady and retry"}
            T_cam_board = np.asarray(pose["T_cam_board"], float).reshape(4, 4)

            if self._duplicate_view(T_cam_board):
                return {"ok": False, "error": "This pose is too similar to an existing view; change the arm angle"}

            view = {
                "raw": raw,
                "angles": angles,
                "sample_time": sample_time,
                "pose_time": float(pose.get("timestamp", 0.0)),
                "T_cam_board": T_cam_board.tolist(),
                "reproj_px": float(pose.get("reproj_px", 0.0)),
                "n_corners": int(pose.get("n_corners", 0)),
            }

            frame = pose.get("frame")
            if frame is not None:
                # Written before the view is accepted, so the stored capture and
                # the solved-from list stay the same length and the same order.
                # A failure here is worth stopping for: silently solving from
                # views that were never recorded is how this stage became
                # unreplayable in the first place.
                self.session.add(
                    frame, servos=raw, detection=pose.get("detection"),
                    extra={"angles": angles,
                           "sample_time": sample_time,
                           "pose_time": float(pose.get("timestamp", 0.0)),
                           "T_cam_board": T_cam_board.tolist(),
                           "reproj_px": float(pose.get("reproj_px", 0.0))})

            with self._lock:
                self.taken.append(view)
                count = len(self.taken)
                # The failure belonged to the view set that produced it. Adding
                # to that set is the operator answering the complaint, so the
                # red banner must go: leaving it up reads as "still broken"
                # while the capture counter climbs behind it.
                self.last_error = None
            return {"ok": True, "count": count}

        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def undo(self) -> dict:
        with self._lock:
            if not self.taken:
                return {"ok": False, "error": "There is no view to undo"}
            self.taken.pop()
            # The stored capture has to drop it too, or the archive ends up
            # holding a view the solve never used -- which is worse than
            # holding none, because it reads as evidence.
            self.session.drop_last()
            # Same reasoning as capture: the view set the error described is
            # gone, so the error no longer describes anything.
            self.last_error = None
            return {"ok": True, "count": len(self.taken)}

    def _persist_result(self, result: dict, taken: list[dict]) -> None:
        """Atomically update the Stage 5 contract consumed by later stages."""
        stage4 = storage.load_result("zeros")
        if not stage4:
            raise RuntimeError("The Stage 4 zero result disappeared while solving")
        recorded = zeros_mod.ZeroSet.from_dict(stage4.get("zeros"))
        corrected = corrected_zeros(result, self.rough_zeros, self.senses)
        for joint_name, correction_deg in result["zeros_deg"].items():
            recorded.add(
                joint_name, corrected[joint_name],
                source="stage5_fusion",
                note=f"vision fusion: {correction_deg:+.2f} deg from rough pose")

        zeros_payload = dict(stage4)
        zeros_payload["zeros"] = recorded.to_dict()
        storage.save_result("zeros", zeros_payload)

        existing = storage.load_result("touch") or {"arms": {}, "captures": {}}
        existing.pop("saved_at", None)
        existing.pop("git_revision", None)
        existing.setdefault("arms", {})[self.arm] = result
        existing.setdefault("captures", {})[self.arm] = taken
        existing["method"] = "wrist_camera_fusion"
        existing["zeros_used"] = {
            name: int(value) for name, value in self.rough_zeros.items()}
        existing.setdefault("senses_used", {}).update(self.senses)
        existing["complete"] = all(
            arm_name in existing["arms"] for arm_name in ("left_arm", "right_arm"))
        storage.save_result("touch", existing)

    def solve(self) -> None:
        def work():
            try:
                with self._lock:
                    taken = list(self.taken)

                intrinsics = {"K": self.K.tolist(), "dist": self.dist.tolist()}
                result, _ = solve_arm(self.sim, self.arm, taken, intrinsics,
                                      self.T_W_B, self.rough_zeros, self.senses)

                self._persist_result(result, taken)
                # Marks the capture as the one that produced the stored result,
                # so a directory found later is not mistaken for an abandoned
                # attempt. Never fatal: the solve is already saved by here, and
                # losing it over a metadata write would be absurd.
                try:
                    self.session.finish(solved=True,
                                        rms_mm=result.get("rms_mm"),
                                        views=len(taken))
                except Exception:
                    pass
                with self._lock:
                    self.solve_result = result
                    self.last_error = None
                    self.solving = False

            except Refused as exc:
                # A gate's wording is the whole answer; a traceback under it
                # only pushes the instruction off the operator's screen.
                with self._lock:
                    self.last_error = str(exc)
                    self.solve_result = None
                    self.solving = False

            except Exception as exc:
                import traceback
                with self._lock:
                    self.last_error = str(exc) + "\n" + traceback.format_exc()
                    self.solve_result = None
                    self.solving = False

        with self._lock:
            if self.solving:
                return
            self.solving = True
            self.solve_result = None
            self.last_error = None

        threading.Thread(target=work, daemon=True).start()

    def status(self) -> dict:
        with self._lock:
            coverage = self._joint_coverage()
            height_span = self._camera_height_span()

            try:
                _, angles_live, _ = self.sample_joints()
            except Exception:
                angles_live = {}

            return {
                "arm": self.arm,
                "count": len(self.taken),
                "coverage": {jn: {"min_deg": np.rad2deg(c["min"]) if c["min"] is not None else None,
                                  "max_deg": np.rad2deg(c["max"]) if c["max"] is not None else None,
                                  "span_deg": np.rad2deg(c["max"] - c["min"]) if c["min"] is not None else 0,
                                  "count": c["count"]}
                             for jn, c in coverage.items()},
                "height_span_m": height_span,
                # Advice, not a gate: the page shows these, and nothing here
                # or on the page prevents a capture or a solve because of them.
                "rich_views": rich_view_count(self.taken),
                "capture_advice": capture_advice(self.taken),
                "live_angles_deg": {jn: np.rad2deg(v) for jn, v in angles_live.items()},
                "solving": self.solving,
                "solve_result": self.solve_result,
                "last_error": self.last_error,
            }


class CameraFeed:
    """ChArUco detection + MJPEG stream for one wrist camera."""

    def __init__(self, cam_name: str, role: str, spec, intrinsics: dict,
                 session: ArmSession):
        self.cam_name = cam_name
        self.role = role
        self.spec = spec
        self.K = np.array(intrinsics["K"], float)
        self.dist = np.array(intrinsics["dist"], float)
        self.session = session
        self.running = False
        # Why the feed stopped, in the operator's terms. Without this a camera
        # that never opened and a camera that opened at the wrong size both look
        # to the browser exactly like one that is merely slow, and the only
        # symptom is a timeout that blames the connection.
        self.error = None
        self._thread = None
        self._lock = threading.Lock()
        self._snapshot = None
        self._last_pose = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def last_pose(self):
        with self._lock:
            return self._last_pose

    def _loop(self):
        width = int(self.session.intrinsics_width)
        height = int(self.session.intrinsics_height)
        cap_result = common.open_camera(self.role, width=width, height=height)
        if cap_result is None:
            self.error = (
                f"Could not open the {self.role.replace('_', ' ')} camera. "
                "The usual cause is another copy of this stage, or the camera "
                "identification tool, still holding the device. Close it and "
                "start this stage again.")
            print(f"\n  {self.error}")
            self.running = False
            return
        cap, device = cap_result
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (width, height):
            self.error = (
                f"The {self.role.replace('_', ' ')} camera ({device}) opened at "
                f"{actual_w}x{actual_h}, but the intrinsics were measured at "
                f"{width}x{height}. Poses solved from the wrong resolution would "
                "be wrong, so the feed was stopped.")
            print(f"\n  {self.error}")
            cap.release()
            common.release_camera(self.role)
            self.running = False
            return

        detector = charuco.BoardDetector(self.spec, min_corners=gates.PNP_MIN_CORNERS)

        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                try:
                    detected = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    if detected is not None and detected["n"] >= gates.PNP_MIN_CORNERS:
                        # Kept before the overlay is drawn: a stored frame is
                        # for re-running detection later, and green dots baked
                        # into the image would corrupt exactly that.
                        clean = frame.copy()
                        for i, (px, py) in enumerate(detected["corners"]):
                            cid = int(detected["ids"][i])
                            cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)
                            cv2.putText(frame, str(cid), (int(px) + 8, int(py) - 8),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        solved = detector.solve_pose(detected, self.K, self.dist)
                        if solved is not None:
                            reproj = detector.reprojection_error(
                                detected, self.K, self.dist,
                                solved["rvec"], solved["tvec"])
                            raw, angles, sample_time = self.session.sample_joints()
                            with self._lock:
                                self._last_pose = {
                                    "T_cam_board": solved["T_cam_board"],
                                    "reproj_px": float(reproj),
                                    "n_corners": detected["n"],
                                    "timestamp": sample_time,
                                    "sample_time": sample_time,
                                    "raw": raw,
                                    "angles": angles,
                                    # Carried so a captured view can be written
                                    # to disk with the image and corners that
                                    # produced it.
                                    "frame": clean,
                                    "detection": {
                                        "corners": detected["corners"],
                                        "ids": detected["ids"],
                                        "n": detected["n"],
                                    },
                                }
                except Exception:
                    pass

                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with self._lock:
                    self._snapshot = jpg.tobytes()

        finally:
            cap.release()
            common.release_camera(self.role)
            self.running = False


def make_handler(sessions, feeds, feed_lock, first_arm=None):
    # Which arm's feed is currently running. It has to agree with the arm the
    # page opens on, or /switch will stop a feed that was never started and
    # leave the running one orphaned.
    active = {"arm": first_arm or next(iter(sessions))}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, payload, code=200):
            body = json.dumps(payload, default=storage.json_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _html(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE_HTML.encode())

        def do_GET(self):
            path = self.path.split("?")[0]

            if path == "/":
                self._html()
            elif path.startswith("/status/"):
                arm = path.split("/")[-1]
                if arm in sessions:
                    status = sessions[arm].status()
                    # The feed is a separate stream, so a camera fault would
                    # otherwise reach the browser only as a missing image.
                    feed = feeds.get(f"{arm}_wrist")
                    status["camera_error"] = feed.error if feed else None
                    status["camera_live"] = bool(feed and feed.running)
                    self._json(status)
                else:
                    self.send_error(404)
            elif path.startswith("/feed/"):
                cam = path.split("/")[-1]
                if cam not in feeds:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while feeds[cam].running:
                        snap = feeds[cam].snapshot()
                        if snap:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(snap)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            path = self.path.split("?")[0]
            parts = path.strip("/").split("/")
            action = parts[0] if parts else ""
            arm = parts[1] if len(parts) > 1 else None

            if action == "capture" and arm in sessions:
                cam = f"{arm}_wrist"
                with feed_lock:
                    cam_feed = feeds.get(cam)
                pose = cam_feed.last_pose() if cam_feed else None
                self._json(sessions[arm].capture(pose))
            elif action == "switch" and arm in sessions:
                with feed_lock:
                    old_arm = active["arm"]
                    if old_arm != arm:
                        old_feed = feeds.get(f"{old_arm}_wrist")
                        if old_feed:
                            old_feed.stop()
                        new_feed = feeds.get(f"{arm}_wrist")
                        if new_feed:
                            new_feed.start()
                        active["arm"] = arm
                self._json({"ok": True, "arm": arm})
            elif action == "undo" and arm in sessions:
                self._json(sessions[arm].undo())
            elif action == "solve" and arm in sessions:
                session = sessions[arm]
                if session.solving:
                    self._json({"ok": False, "error": "This arm is already being solved"}, code=409)
                elif len(session.taken) < MIN_VIEWS:
                    self._json({
                        "ok": False,
                        "error": f"At least {MIN_VIEWS} views are required; there are {len(session.taken)}",
                    }, code=400)
                else:
                    session.solve()
                    self._json({"ok": True, "solving": True})
            else:
                self.send_error(404)

    return Handler


PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 5 Fusion</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:system-ui,-apple-system,sans-serif; background:#1a1d24; color:#e8eaed; padding:20px; }
.container { max-width:1200px; margin:0 auto; }
h1 { font-size:22px; margin-bottom:6px; }
h2 { font-size:17px; margin-bottom:14px; color:#8ab4f8; }
.subtitle { color:#9aa0a6; font-size:14px; margin-bottom:18px; }
.tabs { display:flex; gap:8px; margin-bottom:16px; }
.tab { padding:10px 26px; background:#24272d; border:1px solid #2c3038; border-radius:6px; cursor:pointer; font-size:14px; color:#cdd2d8; }
.tab.active { background:#2f6fb5; color:#fff; border-color:#2f6fb5; }
.layout { display:flex; gap:20px; align-items:flex-start; }
.main-panel { background:#24272f; border-radius:8px; padding:18px; }
.side-panel { width:400px; background:#24272f; border-radius:8px; padding:18px; }
.video-box { position:relative; background:#000; border-radius:6px; overflow:hidden; width:640px; height:480px; margin-bottom:14px; }
.video-box img { width:100%; height:100%; object-fit:contain; }
.video-loading { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#9aa0a6; font-size:13px; }
.coverage-item { margin-bottom:12px; }
.coverage-label { font-size:13px; color:#9aa0a6; margin-bottom:4px; display:flex; justify-content:space-between; }
.coverage-bar { height:8px; background:#2d3139; border-radius:4px; position:relative; }
.coverage-fill { height:100%; background:linear-gradient(90deg,#1976d2,#42a5f5); border-radius:4px; transition:width 0.3s; }
.coverage-live { position:absolute; top:-2px; width:2px; height:12px; background:#ffa726; }
.stats { display:grid; grid-template-columns:auto auto; gap:4px 12px; font-size:13px; margin:14px 0; }
.stats dt { color:#9aa0a6; }
.stats dd { font-variant-numeric:tabular-nums; }
.buttons { display:flex; gap:8px; margin-bottom:10px; }
button { background:#2f6fb5; border:0; color:#fff; padding:10px 18px; border-radius:6px; cursor:pointer; font-size:14px; }
button:hover:not(:disabled) { background:#3a83d0; }
button:disabled { background:#3a3f48; color:#7b8190; cursor:not-allowed; }
button.sec { background:#3a3f48; }
button.sec:hover:not(:disabled) { background:#474d58; }
button.go { background:#2e7d32; }
button.go:hover:not(:disabled) { background:#388e3c; }
.msg { padding:9px 12px; border-radius:4px; font-size:13px; min-height:20px; }
.msg.good { background:#1b5e20; color:#a5d6a7; }
.msg.bad { background:#b71c1c; color:#ef9a9a; }
.result { background:#20262e; border-radius:6px; padding:14px; font-size:13px; margin-top:12px; }
.advice { background:#20262e; border-left:3px solid #8ab4f8; border-radius:4px; padding:10px 12px; font-size:12px; line-height:1.5; color:#c8ccd0; margin-top:12px; }
.result h3 { font-size:15px; margin-bottom:10px; color:#8ab4f8; }
.result pre { background:#1a1d24; padding:8px; border-radius:4px; overflow-x:auto; font-size:12px; line-height:1.5; white-space:pre-wrap; }
</style>
</head>
<body>
<div class="container">
  <h1>Stage 5 Fusion &mdash; arm fusion calibration</h1>
  <p class="subtitle">Watch the ChArUco board with the wrist camera to calibrate arm mount, joint zeros and camera mount together</p>
  <div class="tabs" id="tabs"></div>
  <div class="layout">
    <div class="main-panel">
      <div class="video-box">
        <img id="feed" src="/feed/__FIRST_ARM___wrist">
        <div class="video-loading" id="loading">Loading camera&hellip;</div>
      </div>
      <div class="buttons">
        <button id="btn-capture" onclick="capture()">Capture (space)</button>
        <button class="sec" id="btn-undo" onclick="undo()">Undo</button>
        <button class="go" id="btn-solve" onclick="solve()">Solve</button>
      </div>
      <div class="msg" id="msg"></div>
    </div>
    <div class="side-panel" id="side"></div>
  </div>
</div>
<script>
// Names the arm the operator can point at. Used back-to-front the flanges are
// turned, so the model's left_arm is the one on their right; a tab labelled by
// the stored name would send them to the wrong arm.
const ARM_LABELS = __ARM_LABELS__;
// Left first, by the operator's left. Back-to-front that is the model's
// right_arm, so the tab order comes from the mounting, not the model.
const ARM_ORDER = __ARM_ORDER__;
let currentArm = ARM_ORDER[0];
let state = {};
let notices = {};

async function post(url) {
  try {
    const r = await fetch(url, {method:'POST', cache:'no-store'});
    const text = await r.text();
    if (!r.ok) return {ok:false, error:'The server returned HTTP ' + r.status + ': ' + text};
    return text ? JSON.parse(text) : {ok:true};
  } catch (err) {
    return {ok:false, error:'Capture request failed: ' + err.message};
  }
}

async function refresh() {
  for (const arm of ['left_arm', 'right_arm']) {
    const r = await fetch('/status/' + arm).then(x => x.json()).catch(() => null);
    if (r) state[arm] = r;
  }
  render();
}

async function switchArm(arm) {
  const r = await post('/switch/' + arm);
  if (!r || !r.ok) return;
  currentArm = arm;
  document.getElementById('feed').src = '/feed/' + arm + '_wrist?t=' + Date.now();
  const loading = document.getElementById('loading');
  if (loading) { loading.style.display = 'block'; loading.textContent = 'Loading camera\u2026'; }
  monitorFeed();
  render();
}

function render() {
  const tabsHtml = ARM_ORDER.map(arm => {
    const label = ARM_LABELS[arm];
    const cls = arm === currentArm ? 'tab active' : 'tab';
    const cnt = (state[arm] || {}).count || 0;
    return '<button class="' + cls + '" data-arm="' + arm + '">' + label + ' (' + cnt + ')</button>';
  }).join('');
  const tabsEl = document.getElementById('tabs');
  tabsEl.innerHTML = tabsHtml;
  tabsEl.querySelectorAll('[data-arm]').forEach(tab => {
    tab.addEventListener('click', () => switchArm(tab.dataset.arm));
  });

  const s = state[currentArm] || {};
  const cov = s.coverage || {};
  const live = s.live_angles_deg || {};
  const count = s.count || 0;

  const joints = [
    {key: currentArm + '_shoulder_pan', label: 'Shoulder Pan', range: 60},
    {key: currentArm + '_shoulder_lift', label: 'Shoulder Lift', range: 90},
    {key: currentArm + '_elbow_flex', label: 'Elbow Flex', range: 120},
    {key: currentArm + '_wrist_flex', label: 'Wrist Flex \u26a0\ufe0f', range: 120},
    {key: currentArm + '_wrist_roll', label: 'Wrist Roll \u26a0\ufe0f',
     range: ROLL_TARGET_DEG, required: ROLL_SPAN_DEG},
  ];

  const bars = joints.map(j => {
    const c = cov[j.key] || {};
    const span = c.span_deg || 0;
    const pct = Math.min(100, (span / j.range) * 100);
    const liveVal = live[j.key] || 0;
    // The other bars are advice, and 60% of target has always been their
    // "nearly there". The roll is the one the solve actually refuses on, so
    // its middle mark is the refusal threshold instead.
    const nearly = j.required || j.range * 0.6;
    const status = span >= j.range ? '\u2713' : (span >= nearly ? '~' : '\u2717');
    return '<div class="coverage-item"><div class="coverage-label"><span>' +
      j.label + ' ' + status + '</span><span>' + span.toFixed(1) + '\u00b0 / ' +
      j.range + '\u00b0 (now ' + liveVal.toFixed(1) + '\u00b0)</span></div>' +
      '<div class="coverage-bar"><div class="coverage-fill" style="width:' + pct + '%"></div></div></div>';
  }).join('');

  const heightSpan = ((s.height_span_m || 0) * 1000).toFixed(0);
  const heightOk = (s.height_span_m || 0) > 0.15 ? '\u2713' : '\u2717';

  let resultHtml = '';
  if (s.solve_result) {
    const res = s.solve_result;
    const cond = res.condition_number.toFixed(1);
    const condOk = res.condition_number < 1000 ? '\u2713' : '\u2717';
    const rms = res.holdout_rms_mm.toFixed(2);
    const zeros = Object.entries(res.zeros_deg).map(kv => '  ' + kv[0] + ': ' + kv[1].toFixed(2) + '\u00b0').join('\\n');
    resultHtml = '<div class="result"><h3>Solve result</h3>' +
      '<dl class="stats"><dt>Condition Number ' + condOk + '</dt><dd>' + cond + '</dd>' +
      '<dt>RMS error</dt><dd>' + rms + ' mm</dd></dl>' +
      '<pre>Joint zero correction (relative to Stage 4):\\n' + zeros + '</pre></div>';
  }

  const armLabel = ARM_LABELS[currentArm];
  const rich = s.rich_views || 0;
  const richOk = count === 0 ? '' : (rich >= count * RICH_FRACTION ? '\u2713' : '~');
  const advice = s.capture_advice || '';
  // Marked '~' at worst, never '\u2717'. These two lines are suggestions: the
  // capture and solve buttons ignore them entirely, and an operator who cannot
  // get nearer the board should still take the view.
  const adviceHtml = advice
    ? '<div class="advice">Suggestion: ' + advice + '</div>' : '';
  document.getElementById('side').innerHTML =
    '<h2>' + armLabel + ' (' + count + ' views)</h2>' + bars +
    '<div class="coverage-item"><div class="coverage-label"><span>Camera height variation ' +
    heightOk + '</span><span>' + heightSpan + ' mm (need > 150mm)</span></div></div>' +
    '<div class="coverage-item"><div class="coverage-label"><span>Board fills the frame ' +
    richOk + '</span><span>' + rich + ' of ' + count + ' views \u2265 ' +
    RICH_CORNERS + ' corners</span></div></div>' +
    '<dl class="stats"><dt>Minimum views</dt><dd>MIN_VIEWS</dd>' +
    '<dt>Recommended views</dt><dd>TARGET_VIEWS</dd></dl>' + adviceHtml + resultHtml;

  let msgHtml = '';
  let msgCls = 'msg';
  if (s.solving) { msgHtml = 'Solving\u2026'; }
  // A notice is the result of the operator's most recent action, so it wins
  // over last_error, which describes the solve they have already moved past.
  else if (notices[currentArm]) {
    msgHtml = notices[currentArm].text;
    msgCls = 'msg ' + notices[currentArm].kind;
  }
  else if (s.last_error) { msgHtml = s.last_error; msgCls = 'msg bad'; }
  const msgEl = document.getElementById('msg');
  msgEl.className = msgCls;
  msgEl.textContent = msgHtml;

  document.getElementById('btn-capture').disabled = s.solving || count >= 30;
  document.getElementById('btn-undo').disabled = s.solving || count === 0;
  document.getElementById('btn-solve').disabled = s.solving || count < MIN_VIEWS;
}

async function capture() {
  const msgEl = document.getElementById('msg');
  msgEl.className = 'msg';
  msgEl.textContent = 'Capturing\u2026';
  const r = await post('/capture/' + currentArm);
  if (r && !r.ok && r.error) {
    notices[currentArm] = {kind:'bad', text:r.error};
  } else if (r && r.ok) {
    notices[currentArm] = {kind:'good', text:'Captured view ' + r.count};
  }
  refresh();
}

async function undo() { await post('/undo/' + currentArm); refresh(); }
async function solve() {
  notices[currentArm] = null;
  const msgEl = document.getElementById('msg');
  msgEl.className = 'msg';
  msgEl.textContent = 'Solving\u2026';
  const r = await post('/solve/' + currentArm);
  if (r && !r.ok && r.error) {
    notices[currentArm] = {kind:'bad', text:r.error};
  }
  await refresh();
}

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && !e.repeat) {
    e.preventDefault();
    capture();
  }
});

function monitorFeed() {
  const feedImg = document.getElementById('feed');
  const loading = document.getElementById('loading');
  let checkCount = 0;
  const iv = setInterval(() => {
    checkCount++;
    if (feedImg.naturalWidth > 0) {
      clearInterval(iv);
      if (loading) loading.style.display = 'none';
      return;
    }
    // The backend knows why a camera failed; a bare timeout would send the
    // operator looking at cables when the real cause is usually another copy
    // of this stage still holding the device. Opening a camera can also take
    // a couple of seconds, so wait long enough not to accuse a slow one.
    const err = state[currentArm] && state[currentArm].camera_error;
    if (err) {
      clearInterval(iv);
      if (loading) { loading.innerHTML = err; loading.style.color = '#c00'; }
    } else if (checkCount > 150) {
      clearInterval(iv);
      if (loading) {
        loading.textContent = 'No frames from the camera after 15 seconds. '
          + 'Check that nothing else is using it, then restart this stage.';
        loading.style.color = '#c00';
      }
    }
  }, 100);
}

monitorFeed();
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
""".replace("MIN_VIEWS", str(MIN_VIEWS)).replace(
    "TARGET_VIEWS", str(TARGET_VIEWS)).replace(
    "ROLL_TARGET_DEG", str(WRIST_ROLL_TARGET_DEG)).replace(
    "ROLL_SPAN_DEG", str(WRIST_ROLL_SPAN_DEG)).replace(
    "RICH_CORNERS", str(RICH_VIEW_CORNERS)).replace(
    "RICH_FRACTION", str(RICH_VIEW_TARGET_FRACTION)).replace(
    "__ARM_LABELS__", json.dumps({
        arm: f"{frames.physical_side(arm, frames.declared_mounting()).title()} arm"
        for arm in ("left_arm", "right_arm")})).replace(
    "__ARM_ORDER__", json.dumps(
        list(frames.working_order(frames.declared_mounting())))).replace(
    "__FIRST_ARM__", frames.working_order(frames.declared_mounting())[0])


def main():
    try:
        results = common.require_results(
            "senses", "zeros", "head",
            "intrinsics_left_wrist", "intrinsics_right_wrist")
    except common.Aborted:
        return 1

    spec = common.load_board()
    T_W_B = np.asarray(results["head"]["T_W_B"], float)
    stage4 = results["zeros"]
    measured_ranges = {
        arm: ranges.RangeSet.from_dict(data)
        for arm, data in (stage4.get("ranges") or {}).items()
    }

    got_senses = senses_mod.load()
    if got_senses is None:
        print("\n  No joint sense data found; run Stage 2 first")
        return 1

    print("\n  !!  The board and the base must stay exactly where they were in Stage 3")
    if not common.confirm("Board and base have not moved", False):
        return 1

    sim = model_map.SimModel()
    try:
        robot = servos.RawRobot()
    except Exception as exc:
        print(f"\n  Cannot connect to the robot: {exc}")
        return 1

    sessions = {}
    feeds = {}
    bus_lock = threading.Lock()
    feed_lock = threading.Lock()

    CAM_ROLE = {"left_arm": "left_wrist", "right_arm": "right_wrist"}
    INTR_RESULT = {"left_arm": "intrinsics_left_wrist",
                   "right_arm": "intrinsics_right_wrist"}

    with robot:
        for arm in ("left_arm", "right_arm"):
            cam_name = f"{arm}_wrist"
            role = CAM_ROLE[arm]
            intr_key = INTR_RESULT[arm]

            if intr_key not in results:
                print(f"\n  No intrinsics found for camera {cam_name}; skipping")
                continue

            intrinsics = results[intr_key]

            jnames = [f"{arm}_{j}" for j in
                      ["shoulder_pan", "shoulder_lift", "elbow_flex",
                       "wrist_flex", "wrist_roll"]]

            arm_ranges = measured_ranges.get(arm)
            rough_zeros = {
                jn: int(arm_ranges.zero_raw[jn]) for jn in jnames
                if arm_ranges is not None and jn in arm_ranges.zero_raw
            }
            if len(rough_zeros) != len(jnames):
                miss = [jn for jn in jnames if jn not in rough_zeros]
                print(f"\n  Stage 4 range record has no raw rough zero for: {', '.join(miss)}; skipping {arm}")
                continue

            senses_dict = {jn: got_senses.sign(jn) for jn in jnames}
            weak_senses = [jn for jn in jnames
                           if got_senses.senses[jn].weak]
            if weak_senses:
                print(f"\n  Stage 2 sense measurement too weak for: {', '.join(weak_senses)}; skipping {arm}")
                continue
            missing_ranges = [jn for jn in jnames
                              if arm_ranges is None or jn not in arm_ranges.travels]
            if missing_ranges:
                print(f"\n  Stage 4 has no travel range for: {', '.join(missing_ranges)}; skipping {arm}")
                continue

            sessions[arm] = ArmSession(
                arm, spec, intrinsics, sim, robot, rough_zeros, senses_dict,
                arm_ranges, T_W_B, bus_lock)
            feeds[cam_name] = CameraFeed(cam_name, role, spec, intrinsics,
                                         sessions[arm])

        if not sessions:
            print("\n  No camera intrinsics found for either arm")
            return 1

        # The page opens on the arm the operator can point at, which
        # back-to-front is not the first one built above. The feed that gets
        # started, the arm the handler thinks is active, and the arm the page
        # requests all have to be this same one.
        first_arm = next((arm for arm in frames.working_order(
            frames.declared_mounting()) if arm in sessions),
            next(iter(sessions)))

        handler = make_handler(sessions, feeds, feed_lock, first_arm)
        # Claim the port before touching a camera. A previous copy of this stage
        # that is still alive holds both the port and the video devices, and the
        # camera is the one that fails silently -- binding first turns that into
        # a plain "something else is already running" instead of a feed that
        # never appears.
        try:
            server = ThreadingHTTPServer((HOST, PORT), handler)
        except OSError as exc:
            print(f"\n  Cannot listen on {HOST}:{PORT}: {exc}")
            print("  Another copy of this stage is probably still running, and")
            print("  it is holding the wrist cameras too. Close it (or its")
            print(f"  browser tab at http://{HOST}:{PORT}) and start again.")
            return 1
        server.daemon_threads = True

        feeds[f"{first_arm}_wrist"].start()
        print(f"\n  Web UI: http://localhost:{PORT}")
        print("  Press Ctrl+C to stop\n")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n\n  Stopping\u2026")
        finally:
            for feed in feeds.values():
                feed.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
