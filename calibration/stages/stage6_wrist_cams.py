"""Stage 6: wrist camera extrinsics and the wrist roll zero, web interface.

One arm at a time. The wrist camera watches the ChArUco board while the arm is
posed in many orientations. Per arm this solves 7 parameters: the camera mount
on the Fixed_Jaw body (6) and the correction to stage 4's rough wrist_roll zero
(1). The mount's roll about the roll axis is a gauge freedom shared with the
roll zero, so it is held at the XML nominal and the roll zero absorbs the rest;
see core/wrist_model.py.

    python calibration/run.py --stage 6
    then open http://127.0.0.1:8091

What matters is a WIDE wrist_roll sweep. Rolling the wrist is the only motion
that separates the roll zero from the camera mount rotation, exactly the coupling
that makes the head's tilt zero unsolvable. The camera sits ~24mm off the roll
axis, so the lever arm is usable, but only if the roll actually moves.

Each capture records the five arm encoder counts and a PnP board pose from the
wrist camera. The arm angles use stage 5's zeros (shoulder/lift/elbow/wrist_pitch)
and stage 4's rough zero for wrist_roll; the solved roll zero is a correction to
that. T_W_B comes from stage 3, T_B_A per arm from stage 5.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

STAGES = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGES))
sys.path.insert(0, str(STAGES.parent))

import common  # noqa: E402
import frames  # noqa: E402
import model_map  # noqa: E402
from core import (charuco, gates, ranges, senses as senses_mod, servos,  # noqa: E402
                  storage, wrist_model, zeros as zeros_mod)
# Use the axis-aware solver: it measures the real roll axis (direction + lateral
# position) instead of trusting the XML geometry, which is off by ~7deg / ~15mm.
from core import wrist_solve_axis as wrist_solve  # noqa: E402

HOST, PORT = "127.0.0.1", 8091
# Model order, for keying stored results. Anything the operator sees goes
# through frames.working_order(), which puts the arm on THEIR left first.
ARMS = ("left_arm", "right_arm")
TARGET_VIEWS = 30

# Roles the intrinsics live under, and the camera role open_camera knows.
CAM_ROLE = {"left_arm": "left_wrist", "right_arm": "right_wrist"}
INTR_RESULT = {"left_arm": "intrinsics_left_wrist",
               "right_arm": "intrinsics_right_wrist"}

# A view is a near-duplicate only if it matches a stored one on ALL of these:
# camera position, camera orientation, AND wrist_roll. Any one differing means
# the camera sees the board from a new viewpoint, which the solve wants.
MIN_ROLL_SEPARATION_DEG = 6.0
DUP_TRANS_M = 0.010   # 10 mm: closer than this the camera is in the same spot
DUP_ROT_DEG = 3.0     # 3 deg: closer than this the camera points the same way


class ArmSession:
    """One arm's capture state: stored views, solve, save.

    The heavy fields (sim, robot, T_W_B, T_B_A) are shared references; only the
    per-arm capture list and solve result are owned here.
    """

    def __init__(self, arm: str, spec, intrinsics: dict, sim, robot,
                 zero_raw: dict[str, int], signs: dict[str, int],
                 rough_roll_zero_rad: float, measured_ranges: ranges.RangeSet,
                 T_W_B: np.ndarray, T_B_A: np.ndarray):
        self.arm = arm
        self.spec = spec
        self.K = np.array(intrinsics["K"], float)
        self.dist = np.array(intrinsics["dist"], float)
        self.sim = sim
        self.robot = robot
        self.zero_raw = zero_raw
        self.signs = signs
        self.rough_roll_zero_rad = rough_roll_zero_rad
        self.measured_ranges = measured_ranges
        self._angle_tracker = ranges.RangeAngleTracker(zero_raw, measured_ranges)
        self.T_W_B = np.asarray(T_W_B, float)
        self.T_B_A = np.asarray(T_B_A, float)
        self.roll_motor = wrist_model.ARM_JOINT_NAMES[arm][4]

        self._lock = threading.RLock()  # reentrant: status() calls roll_sweep_deg()
        self.taken: list[dict] = []
        self.solving = False
        self.last_solve: dict | None = None
        self.last_error: str | None = None

        # Continuous joint tracking. A single-turn absolute encoder aliases any
        # motion beyond half a turn, so reading it once per capture mis-reads a
        # wrist_roll that has swept past ±180°. Instead a background thread polls
        # every ~30ms and unwraps against the previous reading, accumulating the
        # true multi-turn position. Capture then uses this continuous position,
        # not a naive single-shot unwrap.
        self._ranges = ranges.RangeSet()
        self._bus_lock = threading.Lock()   # serialise robot reads
        self._track_halt = threading.Event()
        self._active = False   # only poll while this arm's camera is live
        self._track_thread = threading.Thread(target=self._track_loop, daemon=True)
        self._track_thread.start()

    def set_active(self, active: bool) -> None:
        """Track only while this arm is the one on camera (its motors may move).

        On (re)activation, re-seed the tracker's last_raw baseline so a large
        jump that happened while inactive is not mistaken for continuous travel.
        The accumulated span is preserved; only the unwrap reference is refreshed.
        """
        if active and not self._active:
            try:
                with self._bus_lock:
                    raw = {n: self.robot.read_raw(n) for n in self.zero_raw}
                with self._lock:
                    self._angle_tracker.reseed(raw)
                    for name, r in raw.items():
                        if r is None:
                            continue
                        travel = self._ranges.travels.get(name)
                        if travel is not None:
                            travel.last_raw = int(r)  # refresh sweep reference
            except (Exception, ValueError) as exc:
                self.last_error = f"cannot locate joints in Stage 4 ranges: {exc}"
        self._active = active

    def stop_tracking(self) -> None:
        self._track_halt.set()

    def _track_loop(self) -> None:
        while not self._track_halt.is_set():
            if not self._active:
                time.sleep(0.05)
                continue
            try:
                with self._bus_lock:
                    raw = {n: self.robot.read_raw(n) for n in self.zero_raw}
                with self._lock:
                    self._ranges.update(raw)
                    self._angle_tracker.update(raw)
            except Exception:
                pass  # transient bus hiccup; next tick retries
            time.sleep(0.03)

    def _continuous_angles(self, raw: dict[str, int]) -> dict[str, float]:
        """Joint angles from the shared Stage 4 range-aware tracker."""
        self._angle_tracker.update(raw)
        return self._angle_tracker.angles(self.signs)

    def roll_sweep_deg(self) -> float:
        # Prefer the tracker's measured span (wrap-aware, multi-turn) when active.
        # Caller must hold self._lock (status() and advice() both do).
        travel = self._ranges.travels.get(self.roll_motor)
        if travel is not None and travel.steps > 1:
            return float(travel.span_deg)
        rolls = [t["angles"][self.roll_motor] for t in self.taken
                 if self.roll_motor in t.get("angles", {})]
        if len(rolls) < 2:
            return 0.0
        return float(np.rad2deg(max(rolls) - min(rolls)))

    def _duplicate_view(self, pose: dict, roll_rad: float) -> bool:
        """A view adds nothing only if the camera sees the board from essentially
        the same viewpoint AS WELL AS the same wrist_roll.

        What the solve needs is variety in where the camera sits relative to the
        board. Two captures at the same wrist_roll but different arm postures put
        the camera in different places, so they ARE informative and must be kept.
        We therefore reject only near-duplicates: the camera pose (position and
        orientation relative to the board) is nearly identical to a stored view
        AND the wrist_roll barely moved. Either one differing keeps the view.
        """
        T_new = np.asarray(pose["T_cam_board"], float)
        p_new = T_new[:3, 3]
        R_new = T_new[:3, :3]
        for t in self.taken:
            T_old = np.asarray(t["T_cam_board"], float)
            trans_diff = np.linalg.norm(p_new - T_old[:3, 3])
            R_rel = R_new.T @ T_old[:3, :3]
            ang = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
            rot_diff_deg = np.rad2deg(ang)
            other_roll = t["angles"].get(self.roll_motor)
            roll_diff = (abs(np.rad2deg(roll_rad - other_roll))
                         if other_roll is not None else 999.0)
            # Near-duplicate: same viewpoint AND same roll.
            if (trans_diff < DUP_TRANS_M and rot_diff_deg < DUP_ROT_DEG
                    and roll_diff < MIN_ROLL_SEPARATION_DEG):
                return True
        return False

    def capture(self, pose: dict | None) -> dict:
        """Record one view: the current arm angles and the live board pose.

        `pose` is the detector's last good result, carried from the feed thread
        so the capture uses the same frame the operator saw.
        """
        if pose is None:
            return {"ok": False, "error": "no board pose right now; square the "
                    "camera to the board until the overlay is green"}
        try:
            with self._bus_lock:
                raw = {n: self.robot.read_raw(n) for n in self.zero_raw}
            missing = [n for n, v in raw.items() if v is None]
            if missing:
                return {"ok": False,
                        "error": f"no reading from {', '.join(missing)}"}
            angles = self._continuous_angles(raw)

            roll = angles[self.roll_motor]
            if self._duplicate_view(pose, roll):
                return {"ok": False, "error": "this is nearly identical to a "
                        "stored view (same camera viewpoint and wrist_roll); "
                        "move the arm to a different posture or roll further"}

            view = {
                "raw": raw,
                "angles": angles,
                "T_cam_board": np.asarray(pose["T_cam_board"], float).tolist(),
                "reproj_px": float(pose.get("reproj_px", 0.0)),
                "n_corners": int(pose.get("n_corners", 0)),
            }
            with self._lock:
                self.taken.append(view)
                count = len(self.taken)
            return {"ok": True, "count": count}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def undo(self) -> dict:
        with self._lock:
            if not self.taken:
                return {"ok": False, "error": "nothing to undo"}
            self.taken.pop()
            return {"ok": True, "count": len(self.taken)}

    def solve(self) -> None:
        def work():
            session_dir = None
            try:
                with self._lock:
                    taken = list(self.taken)
                    ranges_dict = self._ranges.to_dict()
                
                # Save raw session data for debugging, win or lose
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                session_dir = storage.DATA_DIR / f"stage6_wrist_{self.arm}.{timestamp}"
                session_dir.mkdir(parents=True, exist_ok=True)
                
                session_data = {
                    "arm": self.arm,
                    "timestamp": timestamp,
                    "n_views": len(taken),
                    "views": taken,
                    "zero_raw": self.zero_raw,
                    "signs": self.signs,
                    "rough_roll_zero_rad": self.rough_roll_zero_rad,
                    "T_W_B": self.T_W_B.tolist(),
                    "T_B_A": self.T_B_A.tolist(),
                    "ranges": ranges_dict,
                }
                (session_dir / "session.json").write_text(
                    json.dumps(session_data, indent=2, default=storage.json_default)
                )
                
                if len(taken) < wrist_solve.MIN_VIEWS:
                    error_msg = (f"need at least {wrist_solve.MIN_VIEWS} "
                                f"views, have {len(taken)}")
                    (session_dir / "error.txt").write_text(error_msg)
                    with self._lock:
                        self.last_error = error_msg
                        self.solving = False
                    return

                result = wrist_solve.fit(
                    self.sim, self.arm,
                    [t["angles"] for t in taken],
                    [np.asarray(t["T_cam_board"], float) for t in taken],
                    self.rough_roll_zero_rad, self.T_W_B, self.T_B_A)

                with self._lock:
                    if result is None:
                        error_msg = "fit returned None"
                        (session_dir / "error.txt").write_text(error_msg)
                        self.last_error = error_msg
                    else:
                        # grade() fills result["gates"] (list of dicts with
                        # name/passed/line) and result["passed"] in place.
                        result = wrist_solve.grade(result)
                        result["stamp"] = time.time()
                        
                        # Save solve result (passed or failed gates)
                        (session_dir / "result.json").write_text(
                            json.dumps(result, indent=2, default=storage.json_default)
                        )
                        
                        self.last_solve = result
                        self.last_error = None
                        report(result)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(tb)
                if session_dir:
                    (session_dir / "exception.txt").write_text(tb)
                with self._lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    self.solving = False

        with self._lock:
            if self.solving:
                return
            self.solving = True
            self.last_solve = None
            self.last_error = None
        threading.Thread(target=work, daemon=True).start()

    def status(self) -> dict:
        with self._lock:
            return {
                "arm": self.arm,
                "count": len(self.taken),
                "target": TARGET_VIEWS,
                "min_views": wrist_solve.MIN_VIEWS,
                "roll_sweep_deg": round(self.roll_sweep_deg(), 1),
                "min_roll_sweep_deg": gates.WRIST_MIN_ROLL_SWEEP_DEG,
                "solving": self.solving,
                "last_solve": self.last_solve,
                "last_error": self.last_error,
                "advice": self.advice(),
            }

    def advice(self) -> str:
        n = len(self.taken)
        if n == 0:
            return ("Pose the arm so its wrist camera sees the board, then press "
                    "space. Vary the whole arm between captures.")
        sweep = self.roll_sweep_deg()
        if sweep < gates.WRIST_MIN_ROLL_SWEEP_DEG:
            return (f"wrist_roll has swept {sweep:.0f} deg. Below "
                    f"{gates.WRIST_MIN_ROLL_SWEEP_DEG:.0f} the roll zero cannot be "
                    f"separated from the camera mount. Roll the wrist further.")
        if n < wrist_solve.MIN_VIEWS:
            return (f"Roll coverage is good. {wrist_solve.MIN_VIEWS - n} more views "
                    f"to reach the minimum.")
        if n < TARGET_VIEWS:
            return (f"Enough to solve. Another {TARGET_VIEWS - n} varied views "
                    f"tightens the fit.")
        return "Well covered. Solving now is reasonable."


class WristFeed:
    """Streams ONE wrist camera at a time and runs live ChArUco + PnP.

    Only one arm's camera is open at once. Switching arms releases the current
    device and opens the other, which keeps within the USB bandwidth budget and
    avoids fighting for a V4L2 device that admits one opener.
    """

    def __init__(self, sessions: dict[str, "ArmSession"], spec):
        self.sessions = sessions
        self.spec = spec
        self.detector = charuco.BoardDetector(spec,
                                             min_corners=gates.PNP_MIN_CORNERS)
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._pose: dict | None = None  # last good pose, for capture()
        self.arm: str | None = None
        self.device: str | None = None
        self.error: str | None = None
        self.width = self.height = 0
        self.n_corners = 0
        self.distance_mm = 0.0
        self.reproj_px = 0.0
        self.board_visible = False

        self._switch_to: str | None = None
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self, arm: str) -> None:
        self._switch_to = arm
        self._thread.start()

    def request_switch(self, arm: str) -> None:
        if arm in self.sessions:
            self._switch_to = arm

    def stop(self) -> None:
        self._halt.set()
        self._thread.join(timeout=3.0)

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def current_pose(self) -> dict | None:
        with self._lock:
            return self._pose

    def _open(self, arm: str):
        import cv2
        cap, device = common.open_camera(CAM_ROLE[arm], width=640, height=480)
        self.arm = arm
        self.device = device
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.error = None
        return cap

    def _loop(self):
        import cv2
        cap = None
        try:
            while not self._halt.is_set():
                # Handle an arm switch.
                if self._switch_to is not None and self._switch_to != self.arm:
                    target = self._switch_to
                    # Deactivate tracking on the old arm, activate on the new
                    if self.arm and self.arm in self.sessions:
                        self.sessions[self.arm].set_active(False)
                    if target in self.sessions:
                        self.sessions[target].set_active(True)
                    if cap is not None:
                        cap.release()
                        common.release_camera(CAM_ROLE[self.arm])
                        cap = None
                    with self._lock:
                        self._jpeg = None
                        self._pose = None
                    try:
                        cap = self._open(target)
                    except Exception as exc:
                        self.error = str(exc)
                        self.arm = target  # remember intent so status is sane
                        time.sleep(0.5)
                        continue

                if cap is None:
                    time.sleep(0.05)
                    continue

                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                self._analyse_and_draw(frame)
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
        finally:
            if cap is not None:
                cap.release()
                if self.arm is not None:
                    common.release_camera(CAM_ROLE[self.arm])

    def _analyse_and_draw(self, frame) -> None:
        import cv2
        sess = self.sessions.get(self.arm)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = self.detector.detect(gray)
        self.n_corners = detection["n"] if detection else 0
        self.board_visible = detection is not None

        pose = None
        if detection is not None and sess is not None:
            solved = self.detector.solve_pose(detection, sess.K, sess.dist)
            if solved is not None:
                reproj = self.detector.reprojection_error(
                    detection, sess.K, sess.dist,
                    solved["rvec"], solved["tvec"])
                self.distance_mm = float(
                    np.linalg.norm(np.array(solved["T_cam_board"])[:3, 3]) * 1000)
                self.reproj_px = float(reproj)
                good = reproj <= gates.PNP_MAX_REPROJ_PX
                pose = {
                    "T_cam_board": solved["T_cam_board"],
                    "reproj_px": self.reproj_px,
                    "n_corners": int(detection["n"]),
                    "good": bool(good),
                }
            # Draw the detected corners.
            colour = (0, 240, 0) if (pose and pose["good"]) else (0, 170, 240)
            for (px, py) in detection["corners"]:
                cv2.circle(frame, (int(px), int(py)), 3, colour, -1)
        else:
            self.distance_mm = self.reproj_px = 0.0

        with self._lock:
            self._pose = pose if (pose and pose["good"]) else None

        # Header band.
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
        if self.board_visible:
            label = (f"{self.arm}  corners {self.n_corners}  "
                     f"{self.distance_mm:.0f}mm  reproj {self.reproj_px:.2f}px")
            colour = (0, 240, 0) if (pose and pose["good"]) else (0, 200, 255)
        else:
            label = f"{self.arm}  board not detected"
            colour = (0, 170, 240)
        cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1)

    def stats(self) -> dict:
        return {
            "arm": self.arm,
            "device": self.device,
            "error": self.error,
            "board_visible": self.board_visible,
            "corners": self.n_corners,
            "distance_mm": round(self.distance_mm),
            "reproj_px": round(self.reproj_px, 3),
            "pose_good": self.current_pose() is not None,
        }


def report(result: dict) -> None:
    common.heading(f"Wrist camera: {result['arm']}")
    t = result["mount_translation_mm"]
    r = result["mount_rotation_deg"]
    print(f"  mount translation   {t[0]:+.1f}, {t[1]:+.1f}, {t[2]:+.1f} mm")
    print(f"  mount rotation      {r[0]:+.1f}, {r[1]:+.1f}, {r[2]:+.1f} deg")
    print(f"  wrist_roll zero     {result['wrist_roll_zero_correction_deg']:+.2f} "
          f"deg correction to stage 4")
    if "optical_axis_at_xml_zero" in result:
        oa = result["optical_axis_at_xml_zero"]
        z = result["optical_z_at_xml_zero"]
        print(f"  optical @ XML zero  [{oa[0]:+.3f}, {oa[1]:+.3f}, {oa[2]:+.3f}] "
              f"(chassis; Z={z:+.4f}, should be ~0 = horizontal)")
    print(f"  views               {result['n_views_total']} "
          f"({result['n_views_fit']} fit, {result['n_views_holdout']} holdout)")
    print(f"  wrist_roll sweep    {result['roll_sweep_deg']:.0f} deg")
    print(f"  holdout error       {result['holdout_trans_rms_mm']:.2f} mm, "
          f"{result['holdout_rot_rms_deg']:.2f} deg")
    print(f"  condition number    {result['condition_number']:.1f}")


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_handler(sessions, feed, stage4, got_senses, recorded):

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, payload, code=200):
            body = json.dumps(payload, default=storage.json_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

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
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path.startswith("/status/"):
                arm = path.rsplit("/", 1)[-1]
                if arm not in sessions:
                    self._json({"error": "unknown arm"}, 404)
                    return
                out = sessions[arm].status()
                out["feed"] = feed.stats()
                self._json(out)
            elif path == "/feed":
                self._stream()
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?")[0]
            parts = path.strip("/").split("/")
            action = parts[0] if parts else ""
            arm = parts[1] if len(parts) > 1 else None

            if action == "switch" and arm in sessions:
                feed.request_switch(arm)
                self._json({"ok": True})
            elif action == "capture" and arm in sessions:
                self._json(sessions[arm].capture(feed.current_pose()))
            elif action == "undo" and arm in sessions:
                self._json(sessions[arm].undo())
            elif action == "solve" and arm in sessions:
                sessions[arm].solve()
                self._json({"ok": True})
            elif action == "save" and arm in sessions:
                self._json(self._save(arm))
            else:
                self._json({"error": "not found"}, 404)

        def _save(self, arm: str) -> dict:
            sess = sessions[arm]
            with sess._lock:
                result = sess.last_solve
                taken = list(sess.taken)
            if not result:
                return {"ok": False, "error": "nothing solved yet; press Solve first"}
            if not result.get("passed"):
                failed = [g["name"] for g in result.get("gates", [])
                          if not g["passed"]]
                return {"ok": False,
                        "error": f"gates failed: {', '.join(failed)}. "
                        f"Save is blocked; collect better views."}
            try:
                # Fold the solved wrist_roll correction into the stored zero, so
                # downstream stages read one absolute count as with every joint.
                roll_motor = sess.roll_motor
                correction_deg = result["wrist_roll_zero_correction_deg"]
                sense = got_senses.sign(roll_motor)
                rough_raw = stage4["zeros"]["joints"][roll_motor]["raw"]
                correction_counts = int(round(
                    correction_deg / 360.0 * 4096 * sense))
                new_raw = (rough_raw + correction_counts) % 4096
                recorded.add(roll_motor, new_raw, source="wrist-camera",
                             note=f"stage 6: {correction_deg:+.2f} deg from rough")

                zeros_payload = {
                    "zeros": recorded.to_dict(),
                    "ranges": stage4.get("ranges", {}),
                    "gauge": stage4.get("gauge", {}),
                }
                if "solved_joints" in stage4:
                    zeros_payload["solved_joints"] = stage4["solved_joints"]
                storage.save_result("zeros", zeros_payload)

                # Per-arm wrist result, keyed under its own result name.
                stored = {k: v for k, v in result.items() if k != "gates"}
                stored["body_frame_id"] = head["body_frame_id"]
                stored["stage5b_sources"] = head["stage5b_sources"]
                stored["captures"] = [
                    {kk: vv for kk, vv in t.items() if kk != "angles"}
                    for t in taken]
                # The stored key stays model-canonical -- it is what the later
                # stages load -- while the sentence names the arm the operator
                # is standing in front of.
                which = "wrist_left" if arm == "left_arm" else "wrist_right"
                storage.save_result(which, stored)
                side = frames.physical_side(arm, frames.declared_mounting())
                print(f"\n  Saved the {side} wrist camera calibration "
                      f"({which}).", flush=True)
                return {"ok": True, "message": f"saved the {side} wrist camera"}
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not feed._halt.is_set():
                    jpeg = feed.snapshot()
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode()
                        + b"\r\n\r\n" + jpeg + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


PAGE = ""  # rendered from PAGE_HTML in main()


def main() -> int:
    common.heading("Stage 6: wrist camera extrinsics and wrist roll zero")
    print("  Pose each arm so its wrist camera sees the board, in many")
    print("  orientations. A WIDE wrist_roll sweep is what makes the roll zero")
    print("  separable from the camera mount. One camera is live at a time;")
    print("  switch arms in the browser.")
    print(f"\n  Open http://{HOST}:{PORT} once it is up.")

    try:
        results = common.require_results(
            "senses", "zeros_zero", "head_zero", "touch_zero",
            "intrinsics_left_wrist", "intrinsics_right_wrist")
    except common.Aborted:
        return 1

    spec = common.load_board()
    common.warn_board_drift(spec)

    head = results["head_zero"]
    T_W_B = np.asarray(head["T_W_B"], float)
    stage4 = results["zeros_zero"]
    touch = results["touch_zero"]
    if head.get("body_frame_id") != touch.get("body_frame_id"):
        print("\n  head_zero and touch_zero use different body frames. Re-run Stage 5b.")
        return 1
    recorded = zeros_mod.ZeroSet.from_dict(stage4.get("zeros"))
    measured_ranges = {
        arm: ranges.RangeSet.from_dict(data)
        for arm, data in (stage4.get("ranges") or {}).items()
    }
    got_senses = senses_mod.load()
    if got_senses is None:
        print("\n  No joint senses recorded. Run stage 2 first.")
        return 1

    if not touch.get("arms"):
        print("\n  Stage 5 (touch) has no solved arms. Run stage 5 first.")
        return 1

    print("\n  The board and base must be untouched since stage 3, since the")
    print("  board is still the world frame and T_W_B is measured against it.")
    if not common.confirm("board and robot base both untouched since stage 3",
                          False):
        print("\n  Re-run stage 3 first to re-measure where the board sits.")
        return 1

    sim = model_map.SimModel()
    try:
        robot = servos.RawRobot()
    except Exception as exc:
        print(f"\n  Cannot reach the servos: {exc}")
        return 1

    sessions: dict[str, ArmSession] = {}
    mounting = frames.declared_mounting()
    with robot:
        # Working order, not model order: the operator is told to start with
        # the arm on their left, and back-to-front that is the one the model
        # calls right_arm. Iterating ARMS here would silently reverse the
        # instruction, since this order becomes the tab order below.
        for arm in frames.working_order(mounting):
            spoken = f"{frames.physical_side(arm, mounting)} arm"
            if arm not in touch["arms"]:
                print(f"  The {spoken} was not calibrated in stage 5; skipping.")
                continue
            names = list(model_map.ARM_JOINTS_NO_GRIPPER[arm])
            zero_raw = {n: recorded.joints[n].raw for n in names
                        if n in recorded.joints}
            if len(zero_raw) != len(names):
                miss = [frames.spoken_joint(n, mounting)
                        for n in names if n not in zero_raw]
                print(f"  Missing zeros for {', '.join(miss)}; "
                      f"skipping the {spoken}.")
                continue
            signs = {n: got_senses.sign(n) for n in names}
            arm_ranges = measured_ranges.get(arm)
            missing_ranges = [n for n in names
                              if arm_ranges is None or n not in arm_ranges.travels]
            if missing_ranges:
                spoken_missing = [frames.spoken_joint(n, mounting)
                                  for n in missing_ranges]
                print(f"  Missing Stage 4 ranges for "
                      f"{', '.join(spoken_missing)}; skipping the {spoken}.")
                continue
            rough_roll_zero_rad = 0.0  # zeros already fold stage 4's rough zero
            T_B_A = np.asarray(touch["arms"][arm]["T_B_A"], float)
            intr = results[INTR_RESULT[arm]]
            sessions[arm] = ArmSession(
                arm, spec, intr, sim, robot, zero_raw, signs,
                rough_roll_zero_rad, arm_ranges, T_W_B, T_B_A)

        if not sessions:
            print("\n  No arms ready. Check stages 4 and 5.")
            return 1

        first_arm = next(iter(sessions))
        feed = WristFeed(sessions, spec)
        sessions[first_arm].set_active(True)
        feed.start(first_arm)
        for _ in range(40):
            if feed.error or feed.device:
                break
            time.sleep(0.1)
        if feed.error and not feed.device:
            print(f"\n  {feed.error}")
            feed.stop()
            return 1

        global PAGE
        PAGE = PAGE_HTML.replace(
            "__ARMS__", json.dumps(list(sessions))).replace(
            "__ARM_LABELS__", json.dumps({
                arm: f"{frames.physical_side(arm, mounting).title()} arm"
                for arm in ARMS}))
        handler = make_handler(sessions, feed, stage4, got_senses, recorded)
        try:
            server = Server((HOST, PORT), handler)
        except OSError as exc:
            print(f"\n  Cannot listen on {HOST}:{PORT}: {exc}")
            feed.stop()
            return 1

        print(f"\n  Live on http://{HOST}:{PORT}")
        print("  Save each arm in the browser after it passes. Ctrl-C when done.\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopping...")
        finally:
            for s in sessions.values():
                s.stop_tracking()
            feed.stop()
            server.shutdown()
            server.server_close()

    # Named by the side the operator sees, in the order they worked through
    # them, so a half-finished run says which arm is still outstanding in
    # terms they can act on.
    stored_key = {"left_arm": "wrist_left", "right_arm": "wrist_right"}
    saved = [f"{frames.physical_side(arm, mounting)} wrist camera"
             for arm in frames.working_order(mounting)
             if storage.load_result(stored_key[arm]) is not None]
    common.heading("Stage 6 summary")
    if len(saved) == 2:
        print("  Both wrist cameras saved.")
        print("  Next: python calibration/run.py --stage 8")
        return 0
    if saved:
        print(f"  Saved: {', '.join(saved)}. The other arm still needs doing.")
        return 1
    print("  Nothing saved.")
    return 1


PAGE_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Stage 6: wrist cameras</title>
<style>
  * { box-sizing: border-box; }
  body { background:#14161a; color:#e6e6e6; font-family:system-ui,sans-serif;
         margin:0; padding:20px; }
  h1 { font-size:19px; margin:0 0 3px; }
  h2 { font-size:15px; margin:0 0 8px; }
  .sub { color:#9aa0a6; font-size:13px; margin:0 0 16px; }
  .tabs { display:flex; gap:8px; margin-bottom:16px; }
  .tab { padding:9px 20px; background:#24272d; border:1px solid #2c3038;
         border-radius:6px; cursor:pointer; font-size:14px; color:#cdd2d8;
         font-family:inherit; }
  .tab.active { background:#2f6fb5; color:#fff; border-color:#2f6fb5; }
  .layout { display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; }
  .panel { background:#1d2025; border:1px solid #2c3038; border-radius:9px;
           padding:14px; }
  .side { width:340px; } .main { flex:1; min-width:680px; }
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
  .bar { height:7px; background:#24272d; border-radius:4px; overflow:hidden;
         margin-top:5px; }
  .bar i { display:block; height:100%; background:#2f6fb5; width:0; }
  .bar i.ok { background:#2e7d32; }
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
  .idle { color:#9aa0a6; font-size:14px; padding:20px; line-height:1.7; }
</style></head><body>
<h1>Stage 6: wrist camera extrinsics and wrist roll zero</h1>
<p class="sub" id="sub">One arm at a time. Roll the wrist widely.</p>
<div class="tabs" id="tabs"></div>
<div id="notes"></div>
<div class="layout">
  <div class="panel main">
    <img id="feed" src="/feed">
    <div class="row">
      <button class="act" id="btn-capture" onclick="grab()" disabled>
        Capture <kbd>space</kbd></button>
      <button class="act sec" id="btn-undo" onclick="undo()" disabled>Undo last</button>
      <button class="act go" id="btn-solve" onclick="solve()" disabled>Solve</button>
    </div>
    <div class="msg" id="msg"></div>
    <div class="advice" id="advice"></div>
  </div>
  <div class="panel side" id="side"></div>
</div>

<script>
const ARMS = __ARMS__;
const ARM_LABELS = __ARM_LABELS__;
let currentArm = ARMS[0];
let last = null, solveSeen = null, pollFails = 0;

function post(path) {
  return fetch(path, {method:'POST'}).then(r => r.json()).catch(() => null);
}
async function switchArm(arm) {
  if (arm === currentArm) return;
  currentArm = arm;
  solveSeen = null;
  await post('/switch/' + arm);
  document.getElementById('feed').src = '/feed?t=' + Date.now();
  refresh();
}
async function grab()  { const r = await post('/capture/' + currentArm);
                         if (r && !r.ok) setMsg(r.error, 'bad');
                         else if (r && r.ok) setMsg('stored view ' + r.count, 'good');
                         refresh(); }
async function undo()  { await post('/undo/' + currentArm); refresh(); }
async function solve() { await post('/solve/' + currentArm); solveSeen = null;
                         refresh(); }
async function save()  { const r = await post('/save/' + currentArm);
                         if (r && r.ok) setMsg('saved ' + currentArm, 'good');
                         else if (r) setMsg(r.error, 'bad');
                         refresh(); }

function setMsg(text, cls) {
  const el = document.getElementById('msg');
  if (el) { el.textContent = text || ''; el.className = 'msg ' + (cls || ''); }
}

function tabs() {
  return ARMS.map(a =>
    `<div class="tab ${a === currentArm ? 'active' : ''}"
      onclick="switchArm('${a}')">${ARM_LABELS[a]}</div>`).join('');
}

function gateTable(g) {
  if (!g || !g.length) return '';
  return '<table class="g">' + g.map(x =>
    `<tr><td class="${x.passed ? 'pass' : 'fail'}">${x.passed ? 'PASS' : 'FAIL'}</td>
     <td>${x.line}</td></tr>`).join('') + '</table>';
}

function solveView(s) {
  const r = s.last_solve;
  const t = r.mount_translation_mm, rot = r.mount_rotation_deg;
  return `<h2>${r.passed ? '<span class="pass">Passed</span>'
                         : '<span class="fail">Did not pass</span>'}</h2>
    <dl class="stats">
      <dt>views</dt><dd>${r.n_views_total}
        (${r.n_views_fit} fit, ${r.n_views_holdout} holdout)</dd>
      <dt>roll sweep</dt><dd>${r.roll_sweep_deg.toFixed(0)}&deg;</dd>
      <dt>holdout error</dt>
      <dd>${r.holdout_trans_rms_mm.toFixed(2)} mm,
          ${r.holdout_rot_rms_deg.toFixed(2)}&deg;</dd>
      <dt>worst holdout</dt><dd>${r.holdout_trans_max_mm.toFixed(2)} mm</dd>
      <dt>mount xyz</dt>
      <dd>${t[0].toFixed(1)}, ${t[1].toFixed(1)}, ${t[2].toFixed(1)} mm</dd>
      <dt>mount rot</dt>
      <dd>${rot[0].toFixed(1)}, ${rot[1].toFixed(1)}, ${rot[2].toFixed(1)}&deg;</dd>
      <dt>wrist_roll zero</dt>
      <dd>${r.wrist_roll_zero_correction_deg.toFixed(2)}&deg; correction</dd>
      <dt>conditioning</dt><dd>${r.condition_number.toFixed(1)}</dd>
    </dl>
    ${gateTable(r.gates)}
    <div class="row">
      <button class="act go" onclick="save()" ${r.passed ? '' : 'disabled'}>
        ${r.passed ? 'Save this arm' : 'Blocked (gates failed)'}</button>
      <button class="act sec" onclick="dismiss()">Back to capturing</button>
    </div>
    <p class="sub" style="margin-top:10px">${r.passed
      ? 'Saving folds the roll zero into <code>zeros.json</code> and writes the mount.'
      : 'Not saved. Roll wider or collect more varied views, then Solve again.'}</p>`;
}

function dismiss() {
  solveSeen = (last && last.last_solve) ? last.last_solve.stamp : null;
  refresh();
}

function liveSide(s) {
  const f = s.feed || {};
  const pct = Math.min(100, 100 * s.count / Math.max(1, s.target));
  const sweepPct = Math.min(100, 100 * s.roll_sweep_deg / s.min_roll_sweep_deg);
  const sweepOk = s.roll_sweep_deg >= s.min_roll_sweep_deg;
  return `<h2>${ARM_LABELS[currentArm]}</h2>
    <dl class="stats">
      <dt>camera</dt><dd>${f.device || '&mdash;'}</dd>
      <dt>board</dt><dd>${f.board_visible
        ? (f.pose_good ? '<span class="pill ok">locked</span>'
                       : '<span class="pill no">weak pose</span>')
        : '<span class="pill no">not detected</span>'}</dd>
      <dt>corners</dt><dd>${f.corners ?? 0}</dd>
      <dt>distance</dt><dd>${f.distance_mm ? f.distance_mm + ' mm' : '&mdash;'}</dd>
      <dt>reprojection</dt>
      <dd>${f.reproj_px ? f.reproj_px.toFixed(2) + ' px' : '&mdash;'}</dd>
    </dl>
    <h2 style="margin-top:16px">Coverage</h2>
    <dl class="stats">
      <dt>views</dt><dd><b>${s.count}</b> / ${s.target}</dd>
      <dt>roll sweep</dt>
      <dd>${s.roll_sweep_deg.toFixed(0)}&deg; of ${s.min_roll_sweep_deg}&deg; min</dd>
    </dl>
    <div class="bar"><i style="width:${pct}%"></i></div>
    <p class="sub" style="font-size:11px;margin:6px 0 2px">views to target</p>
    <div class="bar"><i class="${sweepOk ? 'ok' : ''}"
      style="width:${sweepPct}%"></i></div>
    <p class="sub" style="font-size:11px;margin:6px 0 0">
      wrist_roll sweep &mdash; this is what makes the roll zero solvable</p>`;
}

function notes(s) {
  const f = s.feed || {};
  if (f.error) {
    return `<div class="warn">Camera: ${f.error}</div>`;
  }
  return '';
}

async function refresh() {
  let s;
  try {
    const r = await fetch('/status/' + currentArm);
    if (!r.ok) throw new Error('status ' + r.status);
    s = await r.json();
    last = s; pollFails = 0;
    document.getElementById('notes').innerHTML = notes(s);
  } catch (e) {
    if (++pollFails >= 4)
      document.getElementById('notes').innerHTML =
        `<div class="warn">Lost contact with the server (${e.message}).</div>`;
    return;
  }

  document.getElementById('tabs').innerHTML = tabs();

  const side = document.getElementById('side');
  const cap = document.getElementById('btn-capture');
  const undoBtn = document.getElementById('btn-undo');
  const solveBtn = document.getElementById('btn-solve');
  const advice = document.getElementById('advice');

  if (s.solving) {
    side.innerHTML = `<div class="idle"><h2>Solving...</h2>
      <p>Fitting the camera mount and roll zero, scoring on held-out views.</p></div>`;
    solveBtn.textContent = 'Solving...'; solveBtn.disabled = true;
    solveSeen = null;
  } else if (s.last_solve && solveSeen !== s.last_solve.stamp) {
    side.innerHTML = solveView(s);
  } else {
    side.innerHTML = liveSide(s);
  }

  advice.textContent = s.advice || '';
  cap.disabled = !(s.feed && s.feed.pose_good);
  undoBtn.disabled = !s.count;
  const enough = s.count >= s.min_views;
  solveBtn.disabled = !enough || s.solving;
  solveBtn.textContent = s.solving ? 'Solving...'
    : (enough ? 'Solve' : 'Solve (need ' + s.min_views + ')');

  if (s.last_error) setMsg(s.last_error, 'bad');
}

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && !e.repeat) { e.preventDefault(); grab(); }
});
setInterval(refresh, 500);
refresh();
</script></body></html>
"""


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except common.Aborted:
        print("\n  Stopped. Nothing further was saved.")
        raise SystemExit(130)
