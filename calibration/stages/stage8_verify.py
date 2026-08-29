"""Stage 8: passive three-camera validation on held-out poses.

The robot is never commanded. Torque must already be off; the operator manually
poses the head and arms while this process reads encoders and cameras.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_CALIB = _HERE.parent
if str(_CALIB) not in sys.path:
    sys.path.insert(0, str(_CALIB))

import model_map
from core import charuco, gates, head_model, ranges, senses as senses_mod
from core import servos, storage, validation
from stages import common

HOST, PORT = "0.0.0.0", 5008
MIN_CORNERS, MAX_REPROJ_PX, MIN_SAMPLES = 12, 1.5, 10
MAX_CAPTURE_AGE_S = 0.5
CAMERAS = ("head", "left_wrist", "right_wrist")


def _as_intrinsics(data: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    return (np.asarray(data["K"], float), np.asarray(data["dist"], float),
            int(data["width"]), int(data["height"]))


def _paired(results: dict) -> None:
    ids = {results[n].get("body_frame_id") for n in ("head_zero", "touch_zero", "zeros_zero")}
    if len(ids) != 1 or None in ids:
        raise RuntimeError("Stage 5b results do not share one body_frame_id")


def _zero_raw(zeros: dict) -> dict[str, int]:
    return {n: int(z["raw"]) for n, z in zeros["zeros"]["joints"].items()}


def _head_pan_seat() -> float:
    """The model pan angle of the posture recorded as the head zero.

    Delegates to head_model, which owns this because it is fixed by the
    mounting alone: back-to-front is q = pi, normal is q = 0. Nothing is read
    from the calibration results, so nothing upstream has to agree about it.
    """
    return head_model.mounting_pan_offset()


class HeadAngleTracker:
    """Head pan and tilt from raw encoder counts, incrementally and wrap-aware.

    The first reading is measured against the stored zero; every one after
    that accumulates a delta from the previous reading, so the angle keeps
    going past a wrap instead of jumping back. The pan then gets the mounting's
    seat added, which is what makes q = pi mean "facing the board" on a
    back-to-front robot.

    Held here as an object rather than inline in the session so that an
    offline replay can track a recorded capture through the same arithmetic.
    A copy of this loop living in a test harness is a copy that keeps passing
    after the real one changes.
    """

    def __init__(self, zero_raw: dict[str, int], seat: float | None = None):
        self._zero_raw = dict(zero_raw)
        self._seat = _head_pan_seat() if seat is None else float(seat)
        self._last: dict[str, int] = {}
        self._positions: dict[str, float] = {}

    def update(self, raw: dict[str, int]) -> tuple[float, float]:
        """Fold in one encoder reading; return the model's pan and tilt."""
        for name, value in raw.items():
            value = int(value)
            if name not in self._last:
                self._positions[name] = float(
                    servos.unwrap_delta(value - self._zero_raw[name]))
            else:
                self._positions[name] += float(
                    servos.unwrap_delta(value - self._last[name]))
            self._last[name] = value
        scale = 2.0 * np.pi / servos.COUNTS_PER_TURN
        return (self._positions["head_motor_1"] * scale + self._seat,
                self._positions["head_motor_2"] * scale)


def within_tolerance(role: str, position_rms_mm: float,
                     rotation_rms_deg: float) -> bool:
    """Does this camera meet the accuracy the stage asks of it?

    The limits live in one place because they are what "calibrated" means
    here, and an offline check that hard-codes its own copy would go on
    reporting a pass after the stage tightened them.
    """
    limit = 6.0 if role == "head" else 8.0
    return position_rms_mm <= limit and rotation_rms_deg <= 3.0


def _source_payload(results: dict) -> dict:
    return {
        "head_zero": storage.result_fingerprint(results["head_zero"]),
        "touch_zero": storage.result_fingerprint(results["touch_zero"]),
        "zeros_zero": storage.result_fingerprint(results["zeros_zero"]),
        "intrinsics_head": storage.result_fingerprint(results["intrinsics_head"]),
        "intrinsics_left_wrist": storage.result_fingerprint(results["intrinsics_left_wrist"]),
        "intrinsics_right_wrist": storage.result_fingerprint(results["intrinsics_right_wrist"]),
        "body_frame_id": results["head_zero"]["body_frame_id"],
    }


def _gate_summary(summary: dict, role: str) -> dict:
    limit = 6.0 if role == "head" else 8.0
    translation = summary.get("translation_rms_mm")
    rotation = summary.get("rotation_rms_deg")
    return {
        "position_rms_mm": bool(
            translation is not None and translation <= limit),
        "rotation_rms_deg": bool(
            rotation is not None and rotation <= 3.0),
        "minimum_samples": bool(summary["count"] >= MIN_SAMPLES),
    }


def arm_of(role: str) -> str:
    """Which arm a wrist camera rides on.

    The camera is bolted to the arm, so the two turn over together and this
    binding is the same under either mounting. Deliberately NOT routed through
    the mounting conversion: doing so would rename the pair in step and point
    the prediction at the arm the camera is not on.
    """
    return "left_arm" if role == "left_wrist" else "right_arm"


class ValidationSession:
    def __init__(self, role: str, spec, results: dict, sim, robot,
                 bus_lock: threading.Lock):
        self.role, self.spec, self.results = role, spec, results
        self.sim, self.robot, self.bus_lock = sim, robot, bus_lock
        key = {"head": "intrinsics_head", "left_wrist": "intrinsics_left_wrist",
               "right_wrist": "intrinsics_right_wrist"}[role]
        self.K, self.dist, self.width, self.height = _as_intrinsics(results[key])
        self.detector = charuco.BoardDetector(spec, min_corners=MIN_CORNERS)
        self.samples: list[dict] = []
        self.last: dict | None = None
        self.error: str | None = None
        self._lock = threading.RLock()
        self.session = None
        path = storage.session_path("stage8_validation", role)
        storage.archive_session(path)
        self.session = storage.CaptureSession(path, storage.SessionMeta(
            stage="8", purpose="held-out passive validation", camera_role=role,
            board_name=spec.name, width=self.width, height=self.height))
        self._senses = senses_mod.load()
        self._head_zeros = _zero_raw(results["head_zero"])
        self._pan_seat = _head_pan_seat()
        self._head_tracker = HeadAngleTracker(self._head_zeros, self._pan_seat)
        self._arm: str | None = None
        self._joint_names = list(model_map.HEAD_JOINTS)
        self._tracker = None
        if role != "head":
            self._arm = arm_of(role)
            stage4 = results["zeros_zero"].get("ranges") or {}
            measured = ranges.RangeSet.from_dict(stage4.get(self._arm))
            self._joint_names = list(model_map.ARM_JOINTS_NO_GRIPPER[self._arm])
            final_zeros = _zero_raw(results["zeros_zero"])
            zero = {name: final_zeros[name] for name in self._joint_names}
            self._tracker = ranges.RangeAngleTracker(zero, measured)
        self._tracking_active = False
        self._track_halt = threading.Event()
        self._track_thread = threading.Thread(
            target=self._track_loop, daemon=True)
        self._track_thread.start()

    def set_tracking_active(self, active: bool) -> None:
        if active and not self._tracking_active and self._tracker is not None:
            try:
                raw = self._read_raw()
                with self._lock:
                    self._tracker.reseed(raw)
                    self.error = None
            except Exception as exc:
                self.error = f"Cannot locate joints from the Stage 4 ranges: {exc}"
        self._tracking_active = active

    def stop_tracking(self) -> None:
        self._track_halt.set()
        self._track_thread.join(timeout=1.0)

    def _track_loop(self) -> None:
        while not self._track_halt.is_set():
            if not self._tracking_active or self._tracker is None:
                time.sleep(0.05)
                continue
            try:
                raw = self._read_raw()
                with self._lock:
                    self._tracker.update(raw)
            except Exception as exc:
                self.error = f"Continuous joint tracking failed: {exc}"
            time.sleep(0.03)

    def _read_raw(self) -> dict[str, int]:
        names = self._joint_names
        with self.bus_lock:
            raw = {n: self.robot.read_raw(n) for n in names}
        if any(v is None for v in raw.values()):
            raise RuntimeError("encoder read failed")
        return {n: int(v) for n, v in raw.items()}

    def _prediction(self, raw: dict[str, int]) -> tuple[dict, np.ndarray]:
        senses = {n: self._senses.sign(n) for n in raw}
        zeros = self.results["zeros_zero"]
        if self.role == "head":
            pan, tilt = self._head_tracker.update(raw)
            return {"angles": {"head_motor_1": pan, "head_motor_2": tilt}}, validation.head_camera_pose(
                self.results["head_zero"], pan, tilt, (senses["head_motor_1"], senses["head_motor_2"]))
        assert self._arm is not None and self._tracker is not None
        with self._lock:
            self._tracker.update(raw)
            angles = self._tracker.angles(senses)
        return {"angles": angles}, validation.wrist_camera_pose(
            self.sim, self._arm, angles, self.results["head_zero"],
            self.results["touch_zero"])

    def _measured_ranges(self, arm: str):
        data = self.results["zeros_zero"].get("ranges") or self.results["zeros_zero"].get("stage4_ranges")
        if not data:
            raise RuntimeError("zeros_zero does not carry Stage 4 ranges")
        return ranges.RangeSet.from_dict(data.get(arm, data))

    def detect(self, frame: np.ndarray) -> dict | None:
        """Run only the lightweight image-side work needed by live preview."""
        detection = self.detector.detect(frame)
        self.last = {
            "corners": int(detection["n"]) if detection else 0,
            "board_visible": detection is not None,
            "reproj_px": None,
            "predicted_pixel_rms_px": None,
        }
        return detection

    def evaluate(self, detection: dict | None,
                 sample_time: float | None = None) -> dict | None:
        """Latch PnP and FK against one frame-time encoder snapshot."""
        if detection is None:
            return None
        solved = self.detector.solve_pose(detection, self.K, self.dist)
        if solved is None:
            return None
        reproj = self.detector.reprojection_error(detection, self.K, self.dist,
                                                  solved["rvec"], solved["tvec"])
        raw = self._read_raw()
        joint_time = time.monotonic()
        info, predicted = self._prediction(raw)
        observed = validation.observed_camera_pose(solved["T_cam_board"])
        frame_time = joint_time if sample_time is None else float(sample_time)
        sample = {"servos_raw": raw, "angles": info["angles"],
                  "frame_time": frame_time, "joint_time": joint_time,
                  "sync_offset_ms": float((joint_time - frame_time) * 1000.0),
                  "T_W_cam_observed": observed.tolist(),
                  "T_W_cam_predicted": predicted.tolist(),
                  "error": validation.pose_residual(observed, predicted),
                  "pnp_reprojection_px": float(reproj),
                  "n_corners": int(detection["n"]),
                  "corners": np.asarray(detection["corners"], float),
                  "ids": np.asarray(detection["ids"], int),
                  "predicted_pixel_rms_px": validation.predicted_pixel_rms(
                      predicted, self.detector, detection, self.K, self.dist)}
        self.last.update({"reproj_px": float(reproj), "predicted_pixel_rms_px": sample["predicted_pixel_rms_px"],
                          "raw": raw, "predicted": predicted.tolist(), "good": reproj <= MAX_REPROJ_PX})
        return sample

    def _duplicate_view(self, T_W_cam: np.ndarray) -> bool:
        """Match Stage 5 Fusion: both position and orientation must be close."""
        T_new = np.asarray(T_W_cam, float).reshape(4, 4)
        for old in self.samples:
            T_old = np.asarray(old["T_W_cam_observed"], float).reshape(4, 4)
            trans_diff = float(np.linalg.norm(T_new[:3, 3] - T_old[:3, 3]))
            R_rel = T_new[:3, :3].T @ T_old[:3, :3]
            angle = float(np.arccos(np.clip(
                (np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))
            if trans_diff < 0.03 and angle < np.deg2rad(5.0):
                return True
        return False

    def capture(self, frame: np.ndarray, sample: dict | None = None) -> dict:
        try:
            if sample is None:
                detection = self.detect(frame)
                sample = self.evaluate(detection, time.monotonic())
            if sample is None:
                return {"ok": False, "error": "Too few ChArUco corners detected, or PnP failed"}
            age = time.monotonic() - float(sample.get("joint_time", 0.0))
            if age > MAX_CAPTURE_AGE_S:
                return {"ok": False, "error": "The paired image and joint snapshot are stale; hold still and try again"}
            if sample["pnp_reprojection_px"] > MAX_REPROJ_PX:
                return {"ok": False, "error": "PnP reprojection error exceeds 1.5 px"}
            with self._lock:
                if self._duplicate_view(sample["T_W_cam_observed"]):
                    return {
                        "ok": False,
                        "error": "Both camera position and orientation are too similar to an existing frame",
                    }
                self.samples.append(sample)
                self.session.add(
                    frame, sample["servos_raw"],
                    {"corners": sample["corners"], "ids": sample["ids"],
                     "n": sample["n_corners"]},
                    extra={k: v for k, v in sample.items()
                           if k not in ("corners", "ids")})
                return {"ok": True, "count": len(self.samples)}
        except Exception as exc:
            self.error = str(exc)
            return {"ok": False, "error": str(exc)}

    def undo(self) -> dict:
        with self._lock:
            if not self.samples:
                return {"ok": False, "error": "There is no frame to undo"}
            self.samples.pop()
            self.session.drop_last()
            return {"ok": True, "count": len(self.samples)}

    def coverage(self) -> dict:
        with self._lock:
            samples = list(self.samples)
        output = {}
        for name in self._joint_names:
            values = [float(sample["angles"][name]) for sample in samples
                      if name in sample.get("angles", {})]
            output[name] = {
                "min_deg": None if not values else float(np.rad2deg(min(values))),
                "max_deg": None if not values else float(np.rad2deg(max(values))),
                "span_deg": 0.0 if not values else float(
                    np.rad2deg(max(values) - min(values))),
                "latest_deg": None if not values else float(np.rad2deg(values[-1])),
            }
        return output

    def status(self) -> dict:
        return {
            "count": len(self.samples),
            "last": self.last,
            "error": self.error,
            "coverage": self.coverage(),
            "joint_names": list(self._joint_names),
        }

    def result(self) -> dict:
        with self._lock:
            samples = list(self.samples)
        summary = validation.summarise_samples(samples)
        return {"camera": self.role, "samples": samples, "summary": summary,
                "gates": _gate_summary(summary, "head" if self.role == "head" else "wrist"),
                "passed": all(_gate_summary(summary, "head" if self.role == "head" else "wrist").values())}


class CameraFeed:
    """Keep live preview current; expensive validation runs only on capture."""

    def __init__(self, sessions: dict[str, ValidationSession], spec):
        self.sessions, self.spec = sessions, spec
        self.active = "head"
        self.requested = "head"
        self.jpeg: bytes | None = None
        self.error: str | None = None
        self.fps = 0.0
        self.frame_time = 0.0
        self._frame: np.ndarray | None = None
        self._sample: dict | None = None
        self._frame_role: str | None = None
        self._lock = threading.RLock()
        self._halt = threading.Event()
        self._switch = threading.Event()
        self.cap = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.sessions[self.active].set_tracking_active(True)
        self.thread.start()

    def stop(self):
        self._halt.set()
        self.thread.join(timeout=3)
        for session in self.sessions.values():
            session.set_tracking_active(False)
            session.stop_tracking()
        if self.cap is not None:
            self.cap.release()
        common.release_camera(self.active)

    def switch(self, role: str) -> bool:
        if role not in self.sessions:
            return False
        with self._lock:
            self.requested = role
            if role != self.active:
                self._frame = None
                self._sample = None
                self._frame_role = None
                self.jpeg = None
                self._switch.set()
        return True

    def snapshot(self):
        with self._lock:
            return self.jpeg

    def status(self) -> dict:
        with self._lock:
            age_ms = ((time.monotonic() - self.frame_time) * 1000.0
                      if self.frame_time else None)
            return {
                "active": self.active,
                "requested": self.requested,
                "switching": self.active != self.requested,
                "fps": round(self.fps, 1),
                "frame_age_ms": None if age_ms is None else round(age_ms),
                "error": self.error,
            }

    def capture(self):
        with self._lock:
            role = self._frame_role
            frame = None if self._frame is None else self._frame.copy()
            sample = self._sample
        if frame is None or role is None or sample is None:
            return {"ok": False, "error": "No paired image and joint snapshot yet; hold still and wait for corners"}
        if role != self.active or role != self.requested:
            return {"ok": False, "error": "The camera is switching; wait for a new frame"}
        return self.sessions[role].capture(frame, sample)

    def _open(self):
        role = self.requested
        self.sessions[role].set_tracking_active(True)
        self.cap, _ = common.open_camera(
            role, self.sessions[role].width, self.sessions[role].height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        with self._lock:
            self.active = role
            self.error = None
            self.fps = 0.0
            self.frame_time = 0.0

    def _close_active(self):
        old_role = self.active
        self.sessions[old_role].set_tracking_active(False)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        common.release_camera(old_role)

    def _loop(self):
        frames = 0
        fps_since = time.monotonic()
        while not self._halt.is_set():
            try:
                if self.cap is None or self._switch.is_set():
                    self._close_active()
                    self._switch.clear()
                    self._open()
                    frames = 0
                    fps_since = time.monotonic()

                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                role = self.active
                frame_time = time.monotonic()
                detection = self.sessions[role].detect(frame)
                sample = self.sessions[role].evaluate(detection, frame_time)
                annotated = frame.copy()
                if detection is not None:
                    for px, py in detection["corners"]:
                        cv2.circle(annotated, (int(px), int(py)), 3,
                                   (0, 240, 0), -1)
                corners = int(detection["n"]) if detection is not None else 0

                frames += 1
                now = time.monotonic()
                elapsed = now - fps_since
                if elapsed >= 1.0:
                    self.fps = frames / elapsed
                    frames = 0
                    fps_since = now
                colour = (0, 240, 0) if corners >= MIN_CORNERS else (0, 170, 240)
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28),
                              (0, 0, 0), -1)
                cv2.putText(
                    annotated,
                    f"{role}  corners {corners}  {self.fps:.1f} fps",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1)
                encoded, buf = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
                with self._lock:
                    self._frame = frame.copy()
                    self._sample = sample
                    self._frame_role = role
                    self.frame_time = now
                    if encoded:
                        self.jpeg = buf.tobytes()
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
                time.sleep(0.1)


def _html() -> str:
    import frames as frames_mod

    mounting = frames_mod.declared_mounting()
    # Left first, by the operator's left. Back-to-front the wrist cameras ride
    # turned flanges, so the camera stored as right_wrist is the one on their
    # left; labelling and ordering by the stored name would send them to the
    # wrong arm.
    wrists = [frames_mod.named_camera(side, mounting)
              for side in frames_mod.SIDES]
    roles = json.dumps(["head"] + wrists)
    labels = json.dumps({
        "head": "Head camera",
        frames_mod.named_camera("left", mounting): "Left wrist camera",
        frames_mod.named_camera("right", mounting): "Right wrist camera",
    })
    return ("""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 8: three-camera verification</title>
<style>
body{font:16px system-ui;max-width:1050px;margin:20px auto;padding:0 14px;background:#f5f7fa;color:#17202a}
.tabs, .actions{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}button{padding:10px 16px;border:1px solid #aab4c0;border-radius:7px;background:#fff;cursor:pointer}button.active{color:#fff;background:#1769aa;border-color:#1769aa}button:disabled{opacity:.55;cursor:wait}
.feed{position:relative;background:#111;min-height:360px;display:flex;align-items:center;justify-content:center;border-radius:8px;overflow:hidden}.feed img{display:block;width:min(800px,100%);min-height:300px;object-fit:contain}.badge{position:absolute;left:10px;bottom:10px;padding:5px 9px;border-radius:5px;background:#000b;color:#fff}
#status{margin-bottom:10px}.coverage{background:#fff;padding:12px;border-radius:8px;margin:10px 0}.coverage-row{margin:10px 0}.coverage-label{display:flex;justify-content:space-between;gap:12px;font-size:14px}.coverage-track{height:9px;background:#e4e9ef;border-radius:5px;overflow:hidden;margin-top:4px}.coverage-fill{height:100%;background:#1769aa;transition:width .25s}.coverage-empty{color:#7b8794}
#message{min-height:24px;font-weight:600}.bad{color:#b42318}.good{color:#067647}.hint{color:#586574}.report{margin-top:16px}.report-verdict{padding:14px 16px;border-radius:7px;font-size:20px;font-weight:700;background:#fff;border:2px solid currentColor}.report-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin:12px 0}.camera-report{background:#fff;padding:14px;border-radius:7px;border:1px solid #d5dce5}.camera-report h3{font-size:17px;margin:0 0 8px}.metric{display:flex;justify-content:space-between;gap:12px;padding:5px 0;border-bottom:1px solid #edf0f4}.metric:last-child{border:0}.report-section{background:#fff;padding:14px;margin:10px 0;border-radius:7px}.report-section h3{font-size:17px;margin:0 0 8px}.report-section ul{margin:6px 0;padding-left:22px}.raw-report{margin-top:10px}.raw-report pre{max-height:420px;overflow:auto;white-space:pre-wrap;background:#fff;padding:12px;border-radius:7px}
</style>
<h1>Stage 8: independent three-camera verification</h1>
<p class="hint">Pick a camera, wait for the button to highlight and a new frame to appear; pose the robot, let it settle, then press space to capture.</p>
<div class="tabs" id="tabs"></div>
<div class="feed"><img id="feed" src="/mjpeg"><span class="badge" id="live">Connecting to video&hellip;</span></div>
<p id="message"></p><p id="status" class="hint"></p><div id="coverage" class="coverage"></div>
<p id="reportProgress" class="hint">Report progress: reading sample counts for all three cameras&hellip;</p>
<div class="actions"><button id="capture" onclick="capture()">Capture (Space)</button><button onclick="undo()">Undo</button><button id="finish" disabled onclick="finishReport()">Generate report</button></div>
<div id="report" class="report"></div>
<script>
const roles=__ROLES__;const minSamples=10;let requested='head',busy=false,reportReady=false;
const labels=__LABELS__;
const jointLabels={head_motor_1:'Head Pan',head_motor_2:'Head Tilt',left_arm_shoulder_pan:'Shoulder Pan',left_arm_shoulder_lift:'Shoulder Lift',left_arm_elbow_flex:'Elbow Flex',left_arm_wrist_flex:'Wrist Flex',left_arm_wrist_roll:'Wrist Roll',right_arm_shoulder_pan:'Shoulder Pan',right_arm_shoulder_lift:'Shoulder Lift',right_arm_elbow_flex:'Elbow Flex',right_arm_wrist_flex:'Wrist Flex',right_arm_wrist_roll:'Wrist Roll'};
const targets={head_motor_1:50,head_motor_2:35,left_arm_shoulder_pan:50,left_arm_shoulder_lift:60,left_arm_elbow_flex:80,left_arm_wrist_flex:80,left_arm_wrist_roll:90,right_arm_shoulder_pan:50,right_arm_shoulder_lift:60,right_arm_elbow_flex:80,right_arm_wrist_flex:80,right_arm_wrist_roll:90};
function renderTabs(active,switching,cameras={}){document.getElementById('tabs').innerHTML=roles.map(x=>`<button id="tab-${x}" class="${x===active&&!switching?'active':''}" ${busy?'disabled':''} onclick="switchTo('${x}')">${labels[x]} (${(cameras[x]&&cameras[x].count)||0})</button>`).join('')}
function renderCoverage(session){const cov=session.coverage||{};const names=session.joint_names||[];coverage.innerHTML='<strong>Sampled joint coverage</strong>'+(names.length?names.map(name=>{const c=cov[name]||{};const span=c.span_deg||0,target=targets[name]||60,pct=Math.min(100,span/target*100);const detail=c.latest_deg==null?'not sampled yet':`latest ${c.latest_deg.toFixed(1)}° · range ${c.min_deg.toFixed(1)}° … ${c.max_deg.toFixed(1)}° · span ${span.toFixed(1)}°`;return `<div class="coverage-row"><div class="coverage-label"><span>${jointLabels[name]||name}</span><span class="${c.latest_deg==null?'coverage-empty':''}">${detail}</span></div><div class="coverage-track"><div class="coverage-fill" style="width:${pct}%"></div></div></div>`}).join(''):'<p class="coverage-empty">Joint ranges appear once the first frame is captured.</p>')}
function metricLine(label,metric,digits=2){const value=metric.value==null?'not measured':metric.value.toFixed(digits)+' '+metric.unit;return `<div class="metric"><span>${label}</span><strong class="${metric.passed?'good':'bad'}">${value} / ≤ ${metric.limit.toFixed(digits)} ${metric.unit}</strong></div>`}
function renderReport(payload){const h=payload.human_report;const cards=roles.map(role=>{const c=h.cameras[role];return `<section class="camera-report"><h3>${c.label} · <span class="${c.passed?'good':'bad'}">${c.verdict}</span></h3><div class="metric"><span>Valid samples</span><strong>${c.count} frames</strong></div>${metricLine('Position RMS',c.position_rms)}${metricLine('Rotation RMS',c.rotation_rms)}${metricLine('PnP reprojection RMS',c.pnp_reprojection_rms,3)}<div class="metric"><span>Model corner RMS</span><strong>${c.model_pixel_rms_px.toFixed(1)} px</strong></div></section>`}).join('');const drift=h.shared_drift;report.innerHTML=`<div class="report-verdict ${h.passed?'good':'bad'}">${h.verdict}</div><div class="report-grid">${cards}</div><section class="report-section"><h3>Shared drift diagnosis</h3><div class="metric"><span>Overall translation</span><strong>${drift.translation_norm_mm.toFixed(2)} mm</strong></div><div class="metric"><span>Overall rotation</span><strong>${drift.rotation_deg.toFixed(2)}°</strong></div><div class="metric"><span>Mean position improvement after drift correction</span><strong>${drift.position_improvement_percent.toFixed(1)}%</strong></div></section><section class="report-section"><h3>Findings</h3><ul>${h.findings.map(x=>`<li>${x}</li>`).join('')}</ul></section><section class="report-section"><h3>Recommendations</h3><ul>${h.recommendations.map(x=>`<li>${x}</li>`).join('')}</ul></section><details class="raw-report"><summary>Show machine-readable technical detail</summary><pre>${JSON.stringify(payload,null,2)}</pre></details>`}
async function jsonPost(path){const response=await fetch(path,{method:'POST',cache:'no-store'});const text=await response.text();let data;try{data=JSON.parse(text)}catch(e){throw new Error('The server returned an invalid report: '+text.slice(0,160))}if(!response.ok)throw new Error(data.error||('Request failed: HTTP '+response.status));return data}
async function switchTo(role){if(busy)return;busy=true;requested=role;renderTabs('',true,{});message.className='';message.textContent='Releasing the current camera and switching to '+labels[role]+'\u2026';try{const r=await jsonPost('/switch/'+role);if(!r.ok)throw new Error(r.error||'Switch failed')}catch(e){message.className='bad';message.textContent=e.message}finally{busy=false}}
async function capture(){if(busy)return;try{const r=await jsonPost('/capture');message.className=r.ok?'good':'bad';message.textContent=r.ok?'Captured frame '+r.count:r.error}catch(e){message.className='bad';message.textContent=e.message}}
async function undo(){try{const r=await jsonPost('/undo');message.className=r.ok?'good':'bad';message.textContent=r.ok?'Undone, '+r.count+' frames remain':r.error}catch(e){message.className='bad';message.textContent=e.message}}
async function finishReport(){if(busy||!reportReady)return;busy=true;finish.disabled=true;finish.textContent='Generating\u2026';reportProgress.textContent='Computing three-camera verification errors and saving the report\u2026';try{const r=await jsonPost('/finish');renderReport(r);reportProgress.className=r.passed?'good':'bad';reportProgress.textContent=r.passed?'Report generated: all three cameras passed.':'Stage complete and results saved, but the error is too large \u2014 see the diagnosis below.'}catch(e){reportProgress.className='bad';reportProgress.textContent='Report generation failed: '+e.message}finally{busy=false;finish.textContent='Generate report';finish.disabled=!reportReady}}
async function refresh(){try{const r=await (await fetch('/status',{cache:'no-store'})).json();const f=r.feed;renderTabs(f.active,f.switching,r.cameras);live.textContent=f.switching?'Switching to '+labels[f.requested]:(labels[f.active]+' · '+f.fps+' fps · latency '+(f.frame_age_ms??'--')+' ms');capture.disabled=f.switching||f.frame_age_ms===null;const s=r.cameras[f.active];status.textContent='Captured '+s.count+' / '+minSamples+' frames · ChArUco corners '+((s.last&&s.last.corners)||0)+(f.error?' · camera error: '+f.error:'');renderCoverage(s);const counts=roles.map(role=>(r.cameras[role]&&r.cameras[role].count)||0);reportReady=counts.every(count=>count>=minSamples);finish.disabled=busy||!reportReady;reportProgress.className=reportReady?'good':'hint';reportProgress.textContent='Report progress: '+roles.map((role,i)=>labels[role]+' '+counts[i]+' / '+minSamples).join(' · ')+(reportReady?' · ready to generate':' · available once all three reach the target')}catch(e){status.textContent='Status connection failed: '+e.message;reportReady=false;finish.disabled=true}}
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!e.repeat){e.preventDefault();capture()}});renderTabs('head',false,{});setInterval(refresh,400);refresh();
</script>""".replace("__ROLES__", roles).replace("__LABELS__", labels))


def main() -> int:
    try:
        results = common.require_results("senses", "zeros_zero", "head_zero", "touch_zero",
                                        "intrinsics_head", "intrinsics_left_wrist", "intrinsics_right_wrist")
        _paired(results)
        spec = common.load_board()
        problems = senses_mod.load()
        if problems is None: raise RuntimeError("Stage 2 senses result is missing")
        robot = __import__("core.servos", fromlist=["RawRobot"]).RawRobot()
        safety = robot.verify()
        if safety:
            raise RuntimeError("; ".join(safety))
    except Exception as exc:
        print(f"Stage 8 cannot start: {exc}"); return 1
    sim = model_map.SimModel(); lock = threading.Lock()
    sessions = {role: ValidationSession(role, spec, results, sim, robot, lock) for role in CAMERAS}
    feed = CameraFeed(sessions, spec)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def send(self, body, code=200, content="application/json"):
            b = body if isinstance(body, bytes) else body.encode(); self.send_response(code); self.send_header('Content-Type', content); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path.startswith('/mjpeg'):
                self.send_response(200); self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame'); self.end_headers()
                while not feed._halt.is_set():
                    img=feed.snapshot()
                    if img: self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+img+b'\r\n')
                    time.sleep(.08)
                return
            if self.path == '/status':
                camera_status = {
                    role: sessions[role].status()
                    for role in CAMERAS
                }
                self.send(json.dumps(
                    {'feed': feed.status(), 'cameras': camera_status},
                    default=storage.json_default))
                return
            self.send(_html(), content='text/html; charset=utf-8')
        def do_POST(self):
            if self.path.startswith('/switch/'):
                role = self.path.rsplit('/', 1)[-1]
                if feed.switch(role):
                    self.send(json.dumps({"ok": True, "requested": role}))
                else:
                    self.send(json.dumps({"ok": False,
                                          "error": "unknown camera role"}), 400)
                return
            if self.path == '/capture': self.send(json.dumps(feed.capture(), default=storage.json_default)); return
            if self.path == '/undo': self.send(json.dumps(sessions[feed.active].undo())); return
            if self.path == '/finish':
                missing = _missing_samples(sessions)
                if missing:
                    detail = "，".join(
                        f"{role} still needs {count} frames"
                        for role, count in missing.items())
                    self.send(json.dumps({
                        "ok": False,
                        "error": f"Each of the three cameras needs at least {MIN_SAMPLES} frames: {detail}",
                        "missing": missing,
                    }), 409)
                    return
                self.send(json.dumps(
                    finish_results(results, sessions),
                    default=storage.json_default,
                    allow_nan=False))
                return
            self.send('{"ok":false}', 404)
    feed.start(); server = ThreadingHTTPServer((HOST, PORT), Handler); print(f"Stage 8 Web UI: http://localhost:{PORT}")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: feed.stop(); robot.close()
    return 0


def _metric(value, limit: float, unit: str) -> dict:
    return {
        "value": value,
        "limit": limit,
        "unit": unit,
        "passed": bool(value is not None and value <= limit),
    }


def _human_report(payload: dict) -> dict:
    import frames as frames_mod

    # Named for the arm the operator can point at, matching the tabs they just
    # used; back-to-front the stored right_wrist rides the arm on their left.
    mounting = frames_mod.declared_mounting()
    labels = {
        "head": "Head camera",
        frames_mod.named_camera("left", mounting): "Left wrist camera",
        frames_mod.named_camera("right", mounting): "Right wrist camera",
    }
    cameras = {}
    for role, result in payload["cameras"].items():
        summary = result["summary"]
        position_limit = 6.0 if role == "head" else 8.0
        position = _metric(
            summary["translation_rms_mm"], position_limit, "mm")
        rotation = _metric(summary["rotation_rms_deg"], 3.0, "deg")
        pnp = _metric(summary["pnp_reprojection_rms_px"], MAX_REPROJ_PX, "px")
        failed = []
        if not position["passed"]:
            failed.append("position error")
        if not rotation["passed"]:
            failed.append("rotation error")
        if not pnp["passed"]:
            failed.append("corner detection quality")
        cameras[role] = {
            "label": labels[role],
            "passed": result["passed"],
            "verdict": "PASS" if result["passed"] else "FAIL: " + ", ".join(failed),
            "count": summary["count"],
            "position_rms": position,
            "position_p95_mm": summary["translation_p95_mm"],
            "position_max_mm": summary["translation_max_mm"],
            "rotation_rms": rotation,
            "rotation_p95_deg": summary["rotation_p95_deg"],
            "rotation_max_deg": summary["rotation_max_deg"],
            "pnp_reprojection_rms": pnp,
            "model_pixel_rms_px": summary["predicted_pixel_rms_px"],
        }

    raw_position = np.mean([
        result["summary"]["translation_rms_mm"]
        for result in payload["cameras"].values()
    ])
    corrected_position = np.mean([
        summary["translation_rms_mm"]
        for summary in payload["drift_corrected_summaries"].values()
    ])
    improvement = float(
        (raw_position - corrected_position) / raw_position * 100.0
        if raw_position > 0 else 0.0)
    pnp_good = all(
        camera["pnp_reprojection_rms"]["passed"]
        for camera in cameras.values())
    drift_explains = improvement >= 20.0

    findings = []
    if pnp_good:
        findings.append(
            "ChArUco PnP reprojection error is within limits on all three cameras, so image detection and single-frame pose solving are trustworthy.")
    else:
        findings.append(
            "At least one camera fails the ChArUco PnP reprojection limit; check image sharpness, corner count and intrinsics first.")
    if not payload["passed"] and pnp_good:
        findings.append(
            "The dominant error is in the model prediction chain rather than corner detection; check camera extrinsics, joint zeros/senses and frame conventions.")
    if drift_explains:
        findings.append(
            f"Correcting a shared world drift improves mean position RMS by {improvement:.1f}%, so the board or the robot base most likely moved as a whole.")
    else:
        findings.append(
            f"Correcting a shared world drift improves mean position RMS by only {improvement:.1f}%, so a single rigid displacement does not explain the errors on all three cameras.")

    recommendations = []
    if payload["passed"]:
        recommendations.append("All three independent checks passed; the deployment configuration generated by this run can be used.")
    else:
        recommendations.extend([
            "This stage is complete and its results are saved; only the accuracy gates failed, so there is no need to repeat the capture itself.",
            "Do not deploy this robot.yaml; it is saved for diagnosis and traceability only.",
            "Check whether the worst camera degrades sharply at opposite joint directions and large angles, which points at a zero or sense problem.",
            "After fixing the upstream calibration, capture a fresh Stage 8 set that took no part in the fit and verify again.",
        ])

    drift = payload["shared_drift"]
    return {
        "title": "Stage 8 independent three-camera verification report",
        "passed": payload["passed"],
        # The stage is finished either way: the measurement was taken, the
        # report is saved and the run counts as complete. Only the accuracy
        # gates failed, so say that rather than "verification failed", which
        # reads as though the stage itself has to be done again.
        "verdict": "Verification passed; ready to deploy" if payload["passed"]
                   else "Stage complete, but the error is too large to deploy",
        "cameras": cameras,
        "shared_drift": {
            "translation_norm_mm": drift["translation_norm_mm"],
            "translation_mm": drift["translation_mm"],
            "rotation_deg": drift["rotation_deg"],
            "rotation_vector_deg": drift["rotation_vector_deg"],
            "position_improvement_percent": improvement,
            "explains_errors": drift_explains,
        },
        "findings": findings,
        "recommendations": recommendations,
    }


def _human_report_text(report: dict) -> str:
    lines = [report["title"], "=" * len(report["title"]), "", report["verdict"], ""]
    for camera in report["cameras"].values():
        position = camera["position_rms"]
        rotation = camera["rotation_rms"]
        pnp = camera["pnp_reprojection_rms"]
        lines.extend([
            f"{camera['label']}: {camera['verdict']} ({camera['count']} frames)",
            f"  Position RMS: {position['value']:.2f} mm, limit <= {position['limit']:.2f} mm",
            f"  Rotation RMS: {rotation['value']:.2f} deg, limit <= {rotation['limit']:.2f} deg",
            f"  PnP reprojection RMS: {pnp['value']:.3f} px, limit <= {pnp['limit']:.2f} px",
            f"  Model corner RMS: {camera['model_pixel_rms_px']:.2f} px",
            "",
        ])
    drift = report["shared_drift"]
    lines.extend([
        "Shared drift diagnosis",
        f"  Translation: {drift['translation_norm_mm']:.2f} mm",
        f"  Rotation: {drift['rotation_deg']:.2f} deg",
        f"  Mean position RMS improvement after correction: {drift['position_improvement_percent']:.1f}%",
        "",
        "Findings",
        *[f"  - {item}" for item in report["findings"]],
        "",
        "Recommendations",
        *[f"  - {item}" for item in report["recommendations"]],
        "",
    ])
    return "\n".join(lines)


def _missing_samples(sessions) -> dict[str, int]:
    return {
        role: MIN_SAMPLES - len(sessions[role].samples)
        for role in CAMERAS
        if len(sessions[role].samples) < MIN_SAMPLES
    }


def finish_results(results, sessions) -> dict:
    missing = _missing_samples(sessions)
    if missing:
        detail = ", ".join(
            f"{role}: {MIN_SAMPLES - count}/{MIN_SAMPLES}"
            for role, count in missing.items())
        raise ValueError(f"insufficient Stage 8 samples ({detail})")
    per = {role: sessions[role].result() for role in CAMERAS}
    all_samples = {role: item["samples"] for role, item in per.items()}
    drift = validation.shared_world_drift(all_samples)
    drift_matrix = np.asarray(drift["T_observedWorld_predictedWorld"], float)
    corrected = {
        role: validation.summarise_samples(
            validation.drift_corrected_samples(item["samples"], drift_matrix))
        for role, item in per.items()
    }
    payload = {"stage": "8", "sources": _source_payload(results), "cameras": per,
               "shared_drift": drift, "drift_corrected_summaries": corrected,
               "passed": all(x["passed"] for x in per.values()),
               "session_paths": {r: str(sessions[r].session.path) for r in CAMERAS}}
    payload["human_report"] = _human_report(payload)
    payload.pop("saved_at", None); payload.pop("git_revision", None)
    storage.save_result("validation", payload)
    (storage.RESULTS_DIR / "validation_report.txt").write_text(
        _human_report_text(payload["human_report"]), encoding="utf-8")
    deploy = {"calibration_sources": payload["sources"], "body_frame_id": payload["sources"]["body_frame_id"],
              "head": results["head_zero"], "arms": results["touch_zero"], "zeros": results["zeros_zero"],
              "validation": {"passed": payload["passed"], "summary": {r: x["summary"] for r,x in per.items()}}}
    storage.save_result("robot_yaml", deploy)
    plain = json.loads(json.dumps(deploy, default=storage.json_default))
    (storage.RESULTS_DIR / "robot.yaml").write_text(
        yaml.safe_dump(plain, sort_keys=False, allow_unicode=True))
    for s in sessions.values(): s.session.finish(passed=payload["passed"])
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
