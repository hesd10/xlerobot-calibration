"""Stage 5: arm contact calibration, web interface.

One arm at a time. The page shows:
  - HEAD camera feed with ChArUco detection overlay
  - Suggested corners highlighted on the board
  - Joint spread bars showing which joints need more variety
  - Current touch count and target
  - Capture button to record a touch after posing the arm
  - Solve button once enough touches are collected

The operator picks one fixed point on the static jaw (a jaw corner works), touches
it to corners on the board in many postures, then solves. The variety of postures
is what determines the zeros, not the touch count.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import model_map  # noqa: E402
import stage5_touch as stage5_core  # noqa: E402
from core import arm_model, arm_solve, ranges, senses as senses_mod  # noqa: E402
from core import servos, storage, zeros as zeros_mod  # noqa: E402

ARMS = ("left_arm", "right_arm")
TARGET_TOUCHES = 24
WANT_SPREAD_DEG = 70.0


def corner_world(spec, index: int) -> np.ndarray:
    return np.asarray(spec.corner_positions()[index], dtype=float)


def corner_label(spec, index: int) -> str:
    per_row = spec.squares_x - 1
    return f"corner {index} (row {index // per_row + 1}, col {index % per_row + 1})"


def suggest_corners(spec, n: int = 12) -> list[int]:
    """Spread corners over the board's full extent."""
    per_row = spec.squares_x - 1
    rows = spec.squares_y - 1

    n_rows = max(2, int(round(np.sqrt(n * rows / max(1, per_row)))))
    n_cols = max(2, int(np.ceil(n / n_rows)))

    def pick(count: int, extent: int) -> list[int]:
        if count == 1:
            return [extent // 2]
        return [int(round(i * (extent - 1) / (count - 1))) for i in range(count)]

    row_idx = pick(n_rows, rows)
    col_idx = pick(n_cols, per_row)

    out = []
    for i, r in enumerate(row_idx):
        cols = col_idx[::-1] if i % 2 else col_idx
        for c in cols:
            out.append(r * per_row + c)
    return out[:n]


def spreads(taken: list[dict], arm: str) -> dict[str, float]:
    """Degrees of travel each solved joint has covered across the touches so far."""
    out = {}
    for name in arm_model.joint_names(arm):
        vals = [np.rad2deg(t["angles"][name]) for t in taken
                if name in t.get("angles", {})]
        out[name] = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
    return out


def next_advice(taken: list[dict], arm: str) -> str:
    if len(taken) < 2:
        return "vary the whole arm posture between touches"
    sp = spreads(taken, arm)
    worst = min(sp, key=sp.get)
    if sp[worst] >= WANT_SPREAD_DEG:
        return "spread is good on every joint"
    short = worst.replace(f"{arm}_", "").replace("_", " ")
    return f"{short} has only {sp[worst]:.0f}° — reach corners with it varied"


class Session:
    """One arm's contact capture session."""

    def __init__(self, arm: str, spec, sim, robot, zero_raw: dict, signs: dict,
                 measured_ranges: ranges.RangeSet, T_W_B: np.ndarray):
        self.arm = arm
        self.spec = spec
        self.sim = sim
        self.robot = robot
        self.zero_raw = zero_raw
        self.signs = signs
        self.measured_ranges = measured_ranges
        self.T_W_B = T_W_B
        self.suggestions = suggest_corners(spec, TARGET_TOUCHES)
        self.taken: list[dict] = []
        self.solving = False
        self.last_solve: dict | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()
        
        # Camera streaming: snapshot is the latest annotated frame
        self._snapshot: bytes | None = None
        self._snapshot_lock = threading.Lock()

    def set_snapshot(self, jpeg: bytes) -> None:
        """Called by the camera thread to update the latest frame."""
        with self._snapshot_lock:
            self._snapshot = jpeg

    def get_snapshot(self) -> bytes | None:
        """Called by the HTTP handler to retrieve the latest frame."""
        with self._snapshot_lock:
            return self._snapshot

    def status(self) -> dict:
        with self._lock:
            sp = spreads(self.taken, self.arm) if self.taken else {}
            return {
                "arm": self.arm,
                "count": len(self.taken),
                "target": TARGET_TOUCHES,
                "min_touches": arm_solve.MIN_TOUCHES,
                "suggestions": self.suggestions,
                "spreads": sp,
                "advice": next_advice(self.taken, self.arm),
                "solving": self.solving,
                "last_solve": self.last_solve,
                "last_error": self.last_error,
            }

    def capture(self, corner: int) -> dict:
        """Record a touch at the given corner index."""
        # Don't use _lock here to avoid blocking the HTTP thread
        if not 0 <= corner < self.spec.n_corners:
            return {"ok": False, "error": f"corner {corner} out of range"}

        solved = arm_model.joint_names(self.arm)
        all_names = list(self.zero_raw)
        
        # Quick check without long wait
        try:
            raw = {n: self.robot.read_raw(n) for n in all_names}
            missing = [n for n, v in raw.items() if v is None]
            if missing:
                return {"ok": False, "error": f"no reading from {', '.join(missing)}"}

            angles = ranges.angles_from_ranges(
                {n: int(v) for n, v in raw.items()}, self.zero_raw,
                self.signs, self.measured_ranges)
            target_world = corner_world(self.spec, corner)
            target_base = (np.linalg.inv(self.T_W_B) @ np.append(target_world, 1.0))[:3]

            touch = {
                "corner": corner,
                "raw": raw,
                "angles": angles,
                "target_base": target_base.tolist(),
                "target_world": target_world.tolist(),
            }
            
            with self._lock:
                self.taken.append(touch)
                count = len(self.taken)
            
            return {"ok": True, "count": count}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def undo(self) -> dict:
        with self._lock:
            if self.taken:
                dropped = self.taken.pop()
                return {"ok": True, "corner": dropped["corner"], "count": len(self.taken)}
            return {"ok": False, "error": "no touches to undo"}

    def solve(self) -> None:
        """Solve in the background so the UI stays responsive."""
        def _run():
            try:
                with self._lock:
                    if len(self.taken) < arm_solve.MIN_TOUCHES:
                        self.last_error = f"need at least {arm_solve.MIN_TOUCHES} touches"
                        self.solving = False
                        return
                    taken = list(self.taken)

                result = arm_solve.fit(
                    self.sim, self.arm,
                    [t["angles"] for t in taken],
                    [np.asarray(t["target_base"], float) for t in taken],
                    zeros_guess=None)

                with self._lock:
                    if result is None:
                        self.last_error = "fit returned None"
                    else:
                        self.last_solve = result
                        self.last_error = None
            except Exception as exc:
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
        threading.Thread(target=_run, daemon=True).start()


class CameraFeed:
    """Background thread that reads the head camera and annotates with ChArUco."""

    def __init__(self, spec):
        self.spec = spec
        self.running = False
        self._snapshot: bytes | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> tuple[bool, str]:
        """Start the camera thread."""
        if self.running:
            return True, "already running"
        
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        
        # Wait up to 3 seconds for first frame
        for _ in range(30):
            if self._snapshot is not None:
                return True, "camera started"
            time.sleep(0.1)
        
        return False, "camera did not produce a frame in time"

    def _loop(self):
        """Continuously read camera and annotate with ChArUco detection."""
        import cv2
        from core import charuco as charuco_mod
        
        try:
            cap, device = common.open_camera("head", width=640, height=480)
        except Exception as exc:
            print(f"  Camera error: {exc}")
            return
        
        print(f"  Head camera live on {device}")
        self.running = True
        detector = charuco_mod.BoardDetector(self.spec)
        
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                # A bad frame or a transient detector error must not kill the
                # feed: the browser holds one long-lived connection, so an
                # uncaught exception here blanks the video until the stage is
                # restarted.
                try:
                    detected = detector.detect(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    if detected is not None:
                        for i, (px, py) in enumerate(detected["corners"]):
                            corner_id = int(detected["ids"][i])
                            cv2.circle(frame, (int(px), int(py)), 4,
                                       (0, 255, 0), -1)
                            cv2.putText(frame, str(corner_id),
                                        (int(px) + 8, int(py) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 255), 1)
                except Exception as exc:  # noqa: BLE001
                    print(f"  Frame annotation skipped: "
                          f"{type(exc).__name__}: {exc}")

                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with self._lock:
                    self._snapshot = jpg.tobytes()
        finally:
            cap.release()
            common.release_camera("head")
            self.running = False

    def get_snapshot(self) -> bytes | None:
        with self._lock:
            return self._snapshot

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def app(sessions: dict[str, Session], spec, camera_feed, stage4, got_senses,
        recorded, rough_zeros):
    """HTTP server for the web UI."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import json

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress request logging

        def _json(self, data):
            from core import storage as storage_mod
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=storage_mod.json_default).encode())

        def _html(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE_HTML.encode("utf-8"))

        def do_GET(self):
            # Strip any query string (the page cache-busts /feed with ?t=...),
            # so routing matches on the path alone rather than 404-ing.
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._html()
            elif path.startswith("/status/"):
                arm = path.split("/")[-1]
                if arm in sessions:
                    self._json(sessions[arm].status())
                else:
                    self.send_error(404)
            elif path == "/feed":
                # MJPEG stream from the camera thread
                self.send_response(200)
                self.send_header("Content-Type",
                                "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while camera_feed.running:
                        jpeg = camera_feed.get_snapshot()
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                        b"Content-Length: " + str(len(jpeg)).encode()
                                        + b"\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path.startswith("/capture/"):
                arm = self.path.split("/")[-1]
                print(f"  Capture request for {arm}", flush=True)
                if arm not in sessions:
                    print(f"    Error: {arm} not in sessions", flush=True)
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    print(f"    Reading body, length={length}", flush=True)
                    body = json.loads(self.rfile.read(length)) if length else {}
                    corner = body.get("corner")
                    print(f"    Corner: {corner}", flush=True)
                    if corner is None:
                        self._json({"ok": False, "error": "corner not provided"})
                        return
                    print(f"    Calling capture...", flush=True)
                    result = sessions[arm].capture(int(corner))
                    print(f"    Result: {result}", flush=True)
                    self._json(result)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    self._json({"ok": False, "error": str(exc)})
            elif self.path.startswith("/undo/"):
                arm = self.path.split("/")[-1]
                if arm in sessions:
                    self._json(sessions[arm].undo())
                else:
                    self.send_error(404)
            elif self.path.startswith("/solve/"):
                arm = self.path.split("/")[-1]
                if arm in sessions:
                    sessions[arm].solve()
                    self._json({"ok": True})
                else:
                    self.send_error(404)
            elif self.path.startswith("/save/"):
                arm = self.path.split("/")[-1]
                if arm not in sessions:
                    self.send_error(404)
                    return
                
                sess = sessions[arm]
                if not sess.last_solve:
                    self._json({"ok": False, "error": "No solution to save. Click Solve first."})
                    return
                
                # Check gates
                result = sess.last_solve
                gates = arm_solve.grade(result)
                failed = [g.name for g in gates if not g.passed]
                
                # Parse body for force flag
                content_len = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(content_len)) if content_len else {}
                force = body.get("force", False)
                
                if failed and not force:
                    self._json({"ok": False, "gates_failed": failed, 
                               "error": f"Gates failed: {', '.join(failed)}"})
                    return
                
                # Save this arm's result
                try:
                    # Update zeros for this arm from Stage 4's rough baseline.
                    stage5_core.reset_arm_to_rough(recorded, rough_zeros, arm)
                    for joint_name, zero_deg in result["zeros_deg"].items():
                        rough_raw = rough_zeros.joints[joint_name].raw
                        sense = got_senses.sign(joint_name)
                        correction_counts = int(round(
                            zero_deg / 360.0 * servos.COUNTS_PER_TURN * sense))
                        new_raw = (rough_raw + correction_counts) % servos.COUNTS_PER_TURN
                        recorded.add(joint_name, new_raw, source="contact",
                                   note=f"stage 5: {zero_deg:+.2f} deg from rough pose")
                    
                    # Save updated zeros.json, preserving other keys
                    zeros_payload = {
                        "zeros": recorded.to_dict(),
                        "ranges": stage4.get("ranges", {}),
                        "gauge": stage4.get("gauge", {}),
                    }
                    if "solved_joints" in stage4:
                        zeros_payload["solved_joints"] = stage4["solved_joints"]
                    
                    storage.save_result("zeros", zeros_payload)
                    
                    # Load existing touch.json if any
                    existing = storage.load_result("touch") or {"arms": {}, "captures": {}}
                    
                    # Update with this arm's data
                    existing["arms"][arm] = result
                    existing["captures"][arm] = [{k: v for k, v in t.items() if k != "angles"}
                                                 for t in sess.taken]
                    existing["zeros_used"] = rough_zeros.to_dict()
                    existing["senses_used"] = {n: got_senses.sign(n)
                                              for a in ARMS
                                              for n in model_map.ARM_JOINTS_NO_GRIPPER[a]}
                    
                    storage.save_result("touch", existing)
                    msg = f"Saved {arm}"
                    if failed:
                        msg += f" (forced, gates failed: {', '.join(failed)})"
                    print(f"\n  {msg} from web UI.", flush=True)
                    self._json({"ok": True, "message": msg})
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    self._json({"ok": False, "error": str(exc)})
            else:
                self.send_error(404)

    return ThreadingHTTPServer(("127.0.0.1", 8090), Handler)


# Use ThreadingHTTPServer for concurrent requests
from http.server import ThreadingHTTPServer as HTTPServer


PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 5: Arm Contact Calibration</title>
<style>
body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }
h1 { margin: 0 0 20px 0; }
.tabs { display: flex; gap: 10px; margin-bottom: 20px; }
.tab { padding: 10px 20px; background: #ddd; border: none; cursor: pointer; font-size: 16px; }
.tab.active { background: #4CAF50; color: white; }
.container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.camera { border: 2px solid #ddd; border-radius: 4px; width: 100%; height: auto; }
.section { margin-bottom: 20px; }
.section h3 { margin: 0 0 10px 0; font-size: 18px; }
.progress-bar { width: 100%; height: 24px; background: #eee; border-radius: 4px; position: relative; margin-bottom: 8px; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #ff5252 0%, #ffc107 50%, #4CAF50 100%); border-radius: 4px; transition: width 0.3s; }
.progress-label { position: absolute; left: 8px; top: 3px; font-size: 14px; font-weight: bold; }
.button { padding: 12px 24px; font-size: 16px; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; }
.button-primary { background: #4CAF50; color: white; }
.button-secondary { background: #2196F3; color: white; }
.button-warning { background: #ff9800; color: white; }
.button:disabled { background: #ccc; cursor: not-allowed; }
.advice { background: #e3f2fd; padding: 12px; border-radius: 4px; margin-bottom: 12px; font-size: 14px; }
.error { background: #ffebee; color: #c62828; padding: 12px; border-radius: 4px; margin-bottom: 12px; }
.result { background: #f1f8e9; padding: 16px; border-radius: 4px; margin-top: 12px; }
.result h4 { margin: 0 0 8px 0; }
.result-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 14px; }
.suggestions { font-size: 14px; line-height: 1.6; }
.input-group { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; }
.input-group input { padding: 8px; font-size: 16px; width: 100px; border: 1px solid #ddd; border-radius: 4px; }
</style>
</head>
<body>
<h1>Stage 5: Arm Contact Calibration</h1>
<div class="tabs">
  <button class="tab active" onclick="switchArm('left_arm')">Left Arm</button>
  <button class="tab" onclick="switchArm('right_arm')">Right Arm</button>
</div>
<div class="container">
  <div class="grid">
    <div>
      <div class="section">
        <h3>Head Camera</h3>
        <div style="position: relative; background: #f0f0f0; min-height: 360px;">
          <img id="camera" class="camera" src="" alt="camera feed" style="display: block; width: 100%; height: auto;">
          <div id="camera-loading" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 16px; color: #666;">
            Connecting to camera...
          </div>
        </div>
      </div>
    </div>
    <div>
      <div class="section">
        <h3 id="arm-title">Left Arm</h3>
        <div id="advice" class="advice"></div>
        <div id="error" class="error" style="display:none;"></div>
        <p><strong>Touches:</strong> <span id="count">0</span> / <span id="target">24</span></p>
        <div class="input-group">
          <label>Corner index:</label>
          <input type="number" id="corner-input" min="0" placeholder="0">
          <button class="button button-primary" onclick="capture()">Capture</button>
          <button class="button button-secondary" onclick="undo()">Undo</button>
        </div>
        <button class="button button-warning" onclick="solve()" id="solve-btn" disabled>Solve</button>
      </div>
      <div class="section" id="result-section" style="display: none;">
        <h3>Calibration Result</h3>
        <div id="result-details"></div>
        <div id="result-gates"></div>
        <div id="result-recommendation" style="margin-top: 15px; padding: 10px; border-radius: 5px;"></div>
        <button class="button button-primary" onclick="saveResult()" id="save-btn" style="margin-top: 15px;">Save This Result</button>
      </div>
      <div class="section">
        <h3>Joint Spread</h3>
        <div id="spreads"></div>
      </div>
      <div class="section">
        <h3>Suggested Corners</h3>
        <div id="suggestions" class="suggestions"></div>
      </div>
    </div>
  </div>
</div>
<script>
let currentArm = 'left_arm';
let frameInterval = null;
let statusInterval = null;
let solveSeen = false;

function switchArm(arm) {
  currentArm = arm;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('arm-title').textContent = arm.replace('_', ' ').replace(/\\b\\w/g, c => c.toUpperCase());
  solveSeen = false;
  updateStatus();
}

function startCamera() {
  const img = document.getElementById('camera');
  const loading = document.getElementById('camera-loading');
  
  // Cache-bust so a reconnect opens a fresh stream rather than a dead cached one.
  img.src = '/feed?t=' + Date.now();
  
  // Show loading indicator
  if (loading) loading.style.display = 'block';
  
  // MJPEG streams never fire onload because they're multipart/x-mixed-replace
  // and never "finish". Poll naturalWidth instead to detect first frame.
  let checkCount = 0;
  const checkLoaded = setInterval(() => {
    checkCount++;
    if (img.naturalWidth > 0) {
      // First frame arrived
      clearInterval(checkLoaded);
      if (loading) loading.style.display = 'none';
    } else if (checkCount > 50) {
      // 5 seconds timeout
      clearInterval(checkLoaded);
      if (loading) {
        loading.textContent = 'Camera connection timed out; check the server';
        loading.style.color = '#c00';
      }
    }
  }, 100);
  
  // The MJPEG feed is a single long-lived connection. If it drops the <img>
  // fires 'error' and never reconnects on its own, which reads as "the video
  // disappeared". Reopen it after a short delay whenever that happens.
  // Only attach this once to avoid duplicate handlers.
  if (!img.dataset.reconnectAttached) {
    img.addEventListener('error', () => {
      // Only reconnect if src was actually set (not the initial empty state)
      if (img.src && img.src.includes('/feed')) {
        console.log('Camera feed error, reconnecting in 1s...');
        if (loading) {
          loading.textContent = 'Camera connection lost; reconnecting in 1s...';
          loading.style.display = 'block';
        }
        setTimeout(startCamera, 1000);
      }
    });
    img.dataset.reconnectAttached = 'true';
  }
}

function updateStatus() {
  fetch(`/status/${currentArm}`)
    .then(r => r.json())
    .then(data => {
      document.getElementById('count').textContent = data.count;
      document.getElementById('target').textContent = data.target;
      document.getElementById('advice').textContent = data.advice;
      
      const solveBtn = document.getElementById('solve-btn');
      solveBtn.disabled = data.count < data.min_touches || data.solving;
      solveBtn.textContent = data.solving ? 'Solving...' : 'Solve';

      let html = '';
      const want = 70;
      for (const [name, deg] of Object.entries(data.spreads || {})) {
        const pct = Math.min(100, (deg / want) * 100);
        const shortName = name.replace(currentArm + '_', '').replace(/_/g, ' ');
        html += `<div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div>`;
        html += `<div class="progress-label">${shortName}: ${deg.toFixed(0)}°</div></div>`;
      }
      document.getElementById('spreads').innerHTML = html;

      if (data.suggestions && data.suggestions.length) {
        document.getElementById('suggestions').textContent = 
          'Suggested: ' + data.suggestions.slice(0, 12).join(', ');
      }

      if (data.last_error) {
        document.getElementById('error').textContent = data.last_error;
        document.getElementById('error').style.display = 'block';
      } else {
        document.getElementById('error').style.display = 'none';
      }

      if (data.last_solve && !data.solving && !solveSeen) {
        showResult(data.last_solve);
        solveSeen = true;
      } else if (data.solving) {
        document.getElementById('result-section').style.display = 'none';
      }
    });
}

function showResult(res) {
  // Show result details
  let detailsHtml = `<p><strong>Touches:</strong> ${res.n_touches_total} (${res.n_touches_fit} fit, ${res.n_touches_holdout} holdout)</p>`;
  detailsHtml += `<p><strong>Holdout RMS:</strong> ${res.holdout_rms_mm.toFixed(2)} mm (threshold: 5.0 mm)</p>`;
  detailsHtml += `<p><strong>Holdout max:</strong> ${res.holdout_max_mm.toFixed(2)} mm</p>`;
  detailsHtml += `<p><strong>Fit RMS:</strong> ${res.fit_rms_mm.toFixed(2)} mm</p>`;
  detailsHtml += `<p><strong>Condition:</strong> ${res.condition_number.toExponential(1)}</p>`;
  detailsHtml += `<p><strong>Zero corrections:</strong></p><ul>`;
  for (const [name, deg] of Object.entries(res.zeros_deg || {})) {
    const short = name.replace(currentArm + '_', '').replace(/_/g, ' ');
    detailsHtml += `<li>${short}: ${deg >= 0 ? '+' : ''}${deg.toFixed(2)}°</li>`;
  }
  detailsHtml += `</ul>`;
  document.getElementById('result-details').innerHTML = detailsHtml;

  // Show gates
  let gatesHtml = '<p><strong>Acceptance Gates:</strong></p><ul>';
  let allPassed = true;
  
  // Check holdout error
  const holdoutPass = res.holdout_rms_mm <= 5.0;
  allPassed = allPassed && holdoutPass;
  gatesHtml += `<li>${holdoutPass ? '✓ PASS' : '✗ FAIL'}: Holdout error ${res.holdout_rms_mm.toFixed(2)} mm (max 5.0 mm)</li>`;
  
  // Check worst holdout view
  const worstPass = res.holdout_max_mm <= 12.5;
  allPassed = allPassed && worstPass;
  gatesHtml += `<li>${worstPass ? '✓ PASS' : '✗ FAIL'}: Worst holdout view ${res.holdout_max_mm.toFixed(2)} mm (max 12.5 mm)</li>`;
  
  // Check posture spread
  if (res.min_spread_deg !== undefined) {
    const spreadPass = res.min_spread_deg >= 40.0;
    allPassed = allPassed && spreadPass;
    gatesHtml += `<li>${spreadPass ? '✓ PASS' : '✗ FAIL'}: Min joint spread ${res.min_spread_deg.toFixed(1)}° (min 40.0°)</li>`;
  }
  
  // Check condition number
  const condPass = res.condition_number <= 1e6;
  allPassed = allPassed && condPass;
  gatesHtml += `<li>${condPass ? '✓ PASS' : '✗ FAIL'}: Condition number ${res.condition_number.toExponential(1)} (max 1.0e+6)</li>`;
  
  gatesHtml += '</ul>';
  document.getElementById('result-gates').innerHTML = gatesHtml;

  // Show recommendation
  const recDiv = document.getElementById('result-recommendation');
  if (allPassed) {
    recDiv.innerHTML = '<strong>✓ Recommendation:</strong> All gates passed. This calibration is ready to save.';
    recDiv.style.background = '#d4edda';
    recDiv.style.color = '#155724';
  } else {
    recDiv.innerHTML = '<strong>✗ Recommendation:</strong> Some gates failed. Consider re-capturing with more varied postures. You can still save if you choose.';
    recDiv.style.background = '#f8d7da';
    recDiv.style.color = '#721c24';
  }

  document.getElementById('result-section').style.display = 'block';
}

function capture() {
  const corner = parseInt(document.getElementById('corner-input').value);
  if (isNaN(corner)) {
    alert('Enter a corner index');
    return;
  }
  console.log('Capturing corner', corner, 'for arm', currentArm);
  fetch(`/capture/${currentArm}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({corner})
  }).then(r => {
    console.log('Response status:', r.status);
    return r.json();
  }).then(data => {
    console.log('Response data:', data);
    if (!data.ok) alert(data.error);
    updateStatus();
  }).catch(err => {
    console.error('Capture error:', err);
    alert('Capture failed: ' + err);
  });
}

function undo() {
  fetch(`/undo/${currentArm}`, {method: 'POST'})
    .then(r => r.json())
    .then(data => {
      if (!data.ok) alert(data.error);
      updateStatus();
    });
}

function solve() {
  fetch(`/solve/${currentArm}`, {method: 'POST'});
  solveSeen = false;
  setTimeout(updateStatus, 500);
}

function saveResult() {
  if (!confirm(`Save the calibration result for ${currentArm.replace('_', ' ')}?`)) {
    return;
  }
  fetch(`/save/${currentArm}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({force: false})
  })
    .then(r => r.json())
    .then(data => {
      if (data.ok) {
        alert(`Saved ${currentArm.replace('_', ' ')} calibration successfully!`);
        document.getElementById('result-section').style.display = 'none';
      } else if (data.gates_failed) {
        const msg = `Gates failed:\\n${data.gates_failed.join('\\n')}\\n\\nSave anyway? This may cause issues in later stages.`;
        if (confirm(msg)) {
          return fetch(`/save/${currentArm}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({force: true})
          }).then(r => r.json());
        } else {
          throw new Error('Save cancelled');
        }
      } else {
        alert('Error: ' + (data.error || 'Unknown error'));
      }
    })
    .then(data => {
      if (data && data.ok) {
        alert(`Saved ${currentArm.replace('_', ' ')} calibration (forced)!`);
        document.getElementById('result-section').style.display = 'none';
      }
    })
    .catch(err => {
      if (err.message !== 'Save cancelled') {
        alert('Error: ' + err);
      }
    });
}

startCamera();
updateStatus();
statusInterval = setInterval(updateStatus, 1000);
</script>
</body>
</html>
"""


def main() -> int:
    common.heading("Stage 5: arm contact calibration (web interface)")
    print("  Touch a fixed point on each gripper's static jaw to known board")
    print("  corners from many different arm postures. The variety of postures")
    print("  is what determines the joint zeros, not the touch count.")
    print("\n  Open http://127.0.0.1:8090 in your browser for the interactive UI.")
    print("  Click 'Save & Exit' in the browser when done, or press Ctrl+C here.")

    try:
        results = common.require_results("senses", "zeros", "head")
        if not common.confirm_overwrite("touch"):
            return 1
    except common.Aborted:
        return 1

    spec = common.load_board()
    common.warn_board_drift(spec)

    head = results["head"]
    T_W_B = np.asarray(head["T_W_B"], dtype=float)
    stage4 = results["zeros"]
    rough_zeros = stage5_core.stage4_rough_zero_set(stage4)
    recorded = zeros_mod.ZeroSet.from_dict(stage4.get("zeros"))
    measured_ranges = {
        arm: ranges.RangeSet.from_dict(data)
        for arm, data in (stage4.get("ranges") or {}).items()
    }
    got_senses = senses_mod.load()
    if got_senses is None:
        print("\n  No joint senses recorded. Run stage 2 first.")
        return 1

    print("\n  The board must not have moved since stage 3. Its position relative")
    print("  to the robot is what makes each corner a known point.")
    if not common.confirm("board and robot base both untouched since stage 3", False):
        print("\n  Re-run stage 3 first to re-measure where the board sits.")
        return 1

    sim = model_map.SimModel()
    try:
        robot = servos.RawRobot()
    except Exception as exc:
        print(f"\n  Cannot reach the servos: {exc}")
        return 1

    sessions: dict[str, Session] = {}
    with robot:
        for arm in ARMS:
            names = [m for m in model_map.ARM_JOINTS_NO_GRIPPER[arm]]
            zero_raw = {n: rough_zeros.joints[n].raw for n in names
                        if n in rough_zeros.joints}
            if len(zero_raw) != len(names):
                print(f"\n  Stage 4 has no rough zero for "
                      f"{', '.join(n for n in names if n not in zero_raw)}.")
                print(f"  Skipping {arm}.")
                continue
            signs = {n: got_senses.sign(n) for n in names}
            arm_ranges = measured_ranges.get(arm)
            if arm_ranges is None or any(n not in arm_ranges.travels for n in names):
                print(f"\n  Stage 4 has incomplete ranges for {arm}; skipping.")
                continue
            sessions[arm] = Session(
                arm, spec, sim, robot, zero_raw, signs, arm_ranges, T_W_B)

        if not sessions:
            print("\n  No arms ready. Check stage 4 output.")
            return 1

        # Start camera feed
        camera_feed = CameraFeed(spec)
        ok, msg = camera_feed.start()
        if not ok:
            print(f"\n  {msg}")
            return 1
        print(f"  {msg}")

        print(f"\n  Starting server on http://127.0.0.1:8090")
        print("  Click 'Save' in the browser after solving each arm.")
        print("  Close the browser tab when finished. Press Ctrl+C here to stop the server.")
        server = app(sessions, spec, camera_feed, stage4, got_senses,
                     recorded, rough_zeros)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n\n  Server stopped.")
        finally:
            camera_feed.stop()

    # Report what was saved
    saved = storage.load_result("touch")
    if saved and saved.get("arms"):
        common.heading("Saved Arms")
        for arm in sorted(saved["arms"].keys()):
            print(f"  {arm.replace('_', ' ').title()}")
        print("\n  Next: python calibration/run.py --stage 6")
        return 0
    else:
        print("\n  No arms were saved.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
