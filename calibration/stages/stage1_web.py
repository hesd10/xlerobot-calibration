"""Stage 1 as a single web service: pick a camera, capture, solve, in one page.

    python calibration/stages/stage1_intrinsics.py
    then open http://127.0.0.1:8422

Everything happens in the browser. One camera streams at a time, which is all the
calibration needs and also keeps three 500mA modules from fighting over one USB
bus. Nothing else has to be running.

OpenCV here is a headless build, so frames are served over HTTP rather than shown
in a window. Binds to localhost only.

Why the guidance matters: a stack of centred, frontal views gives a very low
reprojection error and a distortion model that is pure extrapolation at the frame
edges, which is exactly where distortion is largest. The tool therefore tracks
coverage and tilt and says what is still missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

STAGES = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGES))

import common  # noqa: E402
from core import charuco, gates, storage  # noqa: E402

# The capture rules, fitting and gate checks are shared with the command-line
# path; only the interface differs.
from stage1_intrinsics import (Capture, ROLES, TARGET_VIEWS,  # noqa: E402
                              camera_label, report_camera, solve_camera)

HOST, PORT = "127.0.0.1", 8422


def _mounting() -> str:
    """How the robot is standing, declared for this workspace."""
    try:
        import frames
        return frames.declared_mounting()
    except Exception:  # noqa: BLE001 - a label must never stop the stage
        return "normal"


def ordered_roles(mounting: str) -> list[str]:
    """Configured roles, head first then the wrists physically left to right."""
    try:
        import frames
        preferred = ["head"] + [frames.named_camera(side, mounting)
                                for side in frames.SIDES]
    except Exception:  # noqa: BLE001
        preferred = list(ROLES)
    ordered = [role for role in preferred if role in ROLES]
    return ordered + [role for role in ROLES if role not in ordered]


class Session:
    """The one camera being calibrated right now, if any.

    Only one camera streams at a time: that is all stage 1 needs, and it keeps
    three 500mA modules off the same USB bus at once.
    """

    def __init__(self, spec: charuco.BoardSpec, target: int):
        self.spec = spec
        self.target = target
        self.lock = threading.Lock()
        self.capture: Capture | None = None
        self.role: str | None = None
        self.solving = False
        self.last_solve: dict | None = None
        self.last_error: str | None = None

    def start(self, role: str) -> tuple[bool, str]:
        with self.lock:
            if self.solving:
                return False, "still solving the previous camera"
            self._stop_locked()

            capture = Capture(role, self.spec, self.target)
            capture.start()
            # Opening a camera can fail slowly, so wait for it to report either
            # a device or an error rather than assuming success.
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
                self.last_error = f"{role} did not start in time"
                return False, self.last_error

            self.capture = capture
            self.role = role
            self.last_error = None
            self.last_solve = None
            return True, f"{role} live on {capture.device}"

    def _stop_locked(self) -> None:
        if self.capture is not None:
            self.capture.stop()
            self.capture.join(timeout=3.0)
            self.capture = None
            self.role = None

    def stop(self) -> None:
        with self.lock:
            self._stop_locked()

    def solve(self) -> tuple[bool, str]:
        """Fit intrinsics for the running camera, then release it."""
        with self.lock:
            capture = self.capture
            if capture is None:
                return False, "no camera is running"
            if self.solving:
                return False, "already solving"
            if len(capture.stored) < gates.INTRINSICS_MIN_VIEWS:
                return False, (f"only {len(capture.stored)} views, need "
                               f"{gates.INTRINSICS_MIN_VIEWS}")
            self.solving = True
            role = capture.role

        def work():
            try:
                result, checks = solve_camera(capture, save=True)
                passed = bool(result) and all(c.passed for c in checks)
                summary = {
                    "role": role,
                    "passed": passed,
                    "gates": [{"name": c.name, "passed": c.passed,
                               "value": c.value, "threshold": c.threshold,
                               "line": c.line()} for c in checks],
                }
                if result:
                    summary.update({
                        "rms_px": result["fit_rms_px"],
                        "holdout_rms_px": result["holdout_rms_px"],
                        "n_views": result["n_views_fit"],
                        "n_views_holdout": result["n_views_holdout"],
                        "coverage": result["coverage"],
                        "fov": result["fov"],
                    })
                    report_camera(result)
                if passed and capture.session is not None:
                    capture.session.finish(
                        solved=True, n_views=len(capture.stored),
                        holdout_rms_px=(result or {}).get("holdout_rms_px"))
                with self.lock:
                    self.last_solve = summary
            except Exception as exc:
                # A failed fit must not kill the server, but swallowing the
                # reason turns a bug in this code into "the calibration did not
                # pass", which sends the operator off recapturing for nothing.
                detail = traceback.format_exc()
                print(f"\n  Solving {role} raised {type(exc).__name__}: {exc}")
                print("  This is a fault in the tool, not in the captured data.")
                print(detail)
                with self.lock:
                    self.last_solve = {
                        "role": role, "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "internal": True,
                    }
            finally:
                with self.lock:
                    self.solving = False
                    self._stop_locked()

        threading.Thread(target=work, daemon=True).start()
        return True, "solving"

    def status(self) -> dict:
        with self.lock:
            capture = self.capture
            out = {
                "board": {"name": self.spec.name,
                          "squares": f"{self.spec.squares_x}x{self.spec.squares_y}",
                          "square_mm": self.spec.square_mm,
                          "dictionary": self.spec.dictionary,
                          "legacy": self.spec.legacy,
                          "measured": self.spec.measured},
                "active": self.role,
                "solving": self.solving,
                "last_solve": self.last_solve,
                "last_error": self.last_error,
                "target": self.target,
                "min_views": gates.INTRINSICS_MIN_VIEWS,
                "cameras": camera_states(self.role),
            }
        out["capture"] = capture.stats() if capture is not None else None
        return out


def camera_states(active: str | None) -> list[dict]:
    """Which cameras exist, which are solved, which is running."""
    from config.cameras import resolve

    try:
        devices = resolve(strict=False)
    except Exception:
        devices = {}

    mounting = _mounting()
    out = []
    for role in ordered_roles(mounting):
        result = storage.load_result(f"intrinsics_{role}")
        out.append({
            "role": role,
            "label": camera_label(role, mounting),
            "present": role in devices,
            "device": devices.get(role),
            "active": role == active,
            "solved": bool(result),
            "holdout_rms_px": result.get("holdout_rms_px") if result else None,
            "fovx_deg": (result.get("fov", {}) or {}).get("fovx_deg")
            if result else None,
        })
    return out


PAGE_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Stage 1: camera intrinsics</title>
<style>
  * { box-sizing: border-box; }
  body { background:#14161a; color:#e6e6e6; font-family:system-ui,sans-serif;
         margin:0; padding:20px; }
  h1 { font-size:19px; margin:0 0 3px; }
  .sub { color:#9aa0a6; font-size:13px; margin:0 0 16px; }
  .layout { display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; }
  .panel { background:#1d2025; border:1px solid #2c3038; border-radius:9px;
           padding:14px; }
  .cams { width:250px; }
  .view { flex:1; min-width:660px; }
  .cam { display:flex; align-items:center; gap:9px; width:100%;
         background:#24272d; border:1px solid #33373f; color:#e6e6e6;
         border-radius:6px; padding:9px 11px; margin-bottom:8px;
         cursor:pointer; font-size:13px; text-align:left; font-family:inherit; }
  .cam:hover:not(:disabled) { background:#2c3038; }
  .cam:disabled { opacity:0.45; cursor:not-allowed; }
  .cam.on { border-color:#3f7d46; background:#1f2c22; }
  .dot { width:8px; height:8px; border-radius:50%; background:#5a6070;
         flex:none; }
  .dot.ok { background:#4caf50; } .dot.live { background:#ffb300; }
  .dot.gone { background:#8b3d3d; }
  .cam .name { flex:1; }
  .cam .note { color:#9aa0a6; font-size:11px; }
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
           font-size:13px; margin-top:12px; font-variant-numeric:tabular-nums; }
  .stats dt { color:#9aa0a6; }
  .advice { margin-top:12px; padding:10px 12px; background:#20262e;
            border-left:3px solid #2f6fb5; border-radius:4px; font-size:13px;
            line-height:1.5; }
  .msg { margin-top:8px; font-size:12px; color:#9aa0a6; min-height:16px; }
  .msg.bad { color:#e0a0a0; } .msg.good { color:#9fd9a3; }
  .bar { height:7px; background:#24272d; border-radius:4px; overflow:hidden;
         margin-top:5px; }
  .bar i { display:block; height:100%; background:#2f6fb5; width:0; }
  .idle { color:#9aa0a6; font-size:14px; padding:44px 20px; text-align:center;
          line-height:1.7; }
  table.g { border-collapse:collapse; margin-top:10px; font-size:12px;
            width:100%; }
  table.g td { padding:4px 8px; border-top:1px solid #2c3038; }
  .pass { color:#9fd9a3; } .fail { color:#e0a0a0; }
  code { background:#24272d; padding:1px 5px; border-radius:3px; font-size:12px; }
  kbd { background:#2c3038; border:1px solid #3d434d; border-radius:3px;
        padding:1px 5px; font-size:11px; font-family:inherit; }
  .warn { background:#3a2f20; border-left:3px solid #a5651f; padding:9px 12px;
          border-radius:4px; font-size:12px; margin-bottom:14px; }
</style></head><body>
<h1>Stage 1: camera intrinsics</h1>
<p class="sub" id="sub">loading...</p>
<div id="pollwarn"></div>
<div id="boardwarn"></div>
<div class="layout">
  <div class="panel cams">
    <div style="font-size:12px;color:#9aa0a6;margin-bottom:9px">
      Cameras &mdash; one at a time
    </div>
    <div id="camlist"></div>
  </div>
  <div class="panel view" id="view"></div>
</div>
"""

PAGE_SCRIPT = """
<script>
let active = null, built = false, busy = false, lastState = null;

function post(path) {
  return fetch(path, {method:'POST'}).then(r => r.json()).catch(() => null);
}

async function pick(role) {
  if (busy) return;
  busy = true;
  setMsg('starting ' + camName(lastState || {}, role) + '...');
  const r = await post('/start/' + role);
  busy = false;
  if (r && !r.ok) setMsg(r.message, 'bad');
  refresh();
}

async function grab() {
  if (!active) return;
  await post('/grab');
  refresh();
}

async function undo() {
  if (!active) return;
  await post('/undo');
  refresh();
}

async function solve() {
  if (!active || busy) return;
  busy = true;
  const r = await post('/solve');
  busy = false;
  if (r && !r.ok) setMsg(r.message, 'bad');
  refresh();
}

async function release() {
  await post('/stop');
  refresh();
}

function setMsg(text, cls) {
  const el = document.getElementById('msg');
  if (el) { el.textContent = text || ''; el.className = 'msg ' + (cls || ''); }
}

// The stored role stays the key for feeds and saved files; the operator only
// ever reads the label, which names the side they can point at.
function camName(s, role) {
  const c = (s.cameras || []).find(x => x.role === role);
  return (c && c.label) ? c.label : role;
}

function camList(s) {
  return s.cameras.map(c => {
    let dot = 'gone', note = 'not detected';
    if (c.active) { dot = 'live'; note = 'streaming'; }
    else if (!c.present) { dot = 'gone'; note = 'not detected'; }
    else if (c.solved) {
      dot = 'ok';
      note = 'done, ' + c.holdout_rms_px.toFixed(3) + 'px';
    } else { dot = ''; note = 'ready'; }
    const dis = (!c.present || s.solving) ? 'disabled' : '';
    return `<button class="cam ${c.active ? 'on' : ''}" ${dis}
             onclick="pick('${c.role}')">
      <span class="dot ${dot}"></span>
      <span class="name">${c.label || c.role}</span>
      <span class="note">${note}</span>
    </button>`;
  }).join('');
}

function gateTable(g) {
  if (!g || !g.length) return '';
  return '<table class="g">' + g.map(x =>
    `<tr><td class="${x.passed ? 'pass' : 'fail'}">${x.passed ? 'PASS' : 'FAIL'}</td>
     <td>${x.line}</td></tr>`).join('') + '</table>';
}

function idleView(s) {
  const done = s.cameras.filter(c => c.solved).length;
  const missing = s.cameras.filter(c => !c.present).map(c => c.label || c.role);
  let extra = '';
  if (missing.length) {
    extra = `<p style="color:#e0a0a0">Not detected: ${missing.join(', ')}.
      Check the cable, then re-run
      <code>tools/cameras/identify.py</code> if it moved socket.</p>`;
  }
  if (s.last_solve) {
    const r = s.last_solve;
    return `<h2 style="font-size:15px;margin:0 0 4px">
      ${camName(s, r.role)}: ${r.passed ? '<span class="pass">passed</span>'
                            : '<span class="fail">did not pass</span>'}</h2>
      ${r.error ? `<p class="fail">${r.error}</p>${r.internal ? `
        <p class="sub">This is a fault in the tool, not in what you captured.
        The views are still on disk, so there is nothing to recapture. The
        terminal has the traceback.</p>` : ''}` : `
      <dl class="stats">
        <dt>fit RMS</dt><dd>${r.rms_px.toFixed(4)} px (${r.n_views} views)</dd>
        <dt>holdout RMS</dt><dd>${r.holdout_rms_px.toFixed(4)} px
            (${r.n_views_holdout} views)</dd>
        <dt>coverage</dt><dd>${(r.coverage*100).toFixed(0)}% of the frame</dd>
        <dt>field of view</dt><dd>${r.fov.fovx_deg.toFixed(1)}&deg; horizontal</dd>
      </dl>${gateTable(r.gates)}`}
      <p class="sub" style="margin-top:14px">Pick the next camera on the left.</p>`;
  }
  return `<div class="idle">
    <p>${done} of ${s.cameras.length} cameras calibrated.</p>
    <p>Pick a camera on the left to start.<br>
    Only that one will stream.</p>${extra}</div>`;
}

function liveView(s) {
  const c = s.capture;
  const pct = Math.min(100, 100 * c.stored / Math.max(1, c.target));
  const enough = c.stored >= s.min_views;
  return `<div style="display:flex;gap:16px;flex-wrap:wrap">
    <div>
      <img id="feed" src="/feed?t=${Date.now()}">
      <div class="row">
        <button class="act" onclick="grab()">Capture <kbd>space</kbd></button>
        <button class="act sec" onclick="undo()"
          ${c.stored ? '' : 'disabled'}>Undo last</button>
        <button class="act go" onclick="solve()" ${enough ? '' : 'disabled'}>
          Solve${enough ? '' : ' (minimum ' + s.min_views + ')'}</button>
        <button class="act sec" onclick="release()">Release camera</button>
      </div>
      <div class="msg" id="msg">${c.message || ''}</div>
    </div>
    <div style="min-width:250px;flex:1">
      <dl class="stats">
        <dt>camera</dt><dd>${camName(s, c.role)}</dd>
        <dt>device</dt><dd><code>${c.device}</code> ${c.width}x${c.height}</dd>
        <dt>rate</dt><dd>${c.fps} fps</dd>
        <dt>corners now</dt><dd>${c.corners}</dd>
        <dt>sharpness</dt><dd>${c.sharpness}</dd>
        <dt>views stored</dt><dd><b>${c.stored}</b> / ${c.target}</dd>
        <dt>coverage</dt><dd>${c.coverage}%</dd>
        <dt>tilt seen</dt>
        <dd>${c.tilt[0].toFixed(0)}&ndash;${c.tilt[1].toFixed(0)}&deg;</dd>
      </dl>
      <div class="bar"><i style="width:${pct}%"></i></div>
      <div class="advice">${c.advice}</div>
    </div>
  </div>`;
}

let pollFails = 0;

async function refresh() {
  let s;
  // A silent catch here is what turned a broken /status into an apparent hang:
  // the page kept showing its last state forever with no hint anything was wrong.
  try {
    const r = await fetch('/status');
    if (!r.ok) throw new Error('status ' + r.status);
    s = await r.json();
    lastState = s;
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
    `${s.board.square_mm}mm, ${s.board.dictionary}` +
    (s.board.legacy ? ', legacy' : '');

  if (!s.board.measured) {
    document.getElementById('boardwarn').innerHTML =
      `<div class="warn">The board square size was not measured with calipers,
       so every distance this calibration reports carries the printer's scale
       error.</div>`;
  }

  // Only rewrite when it actually changed. Replacing this every poll destroys
  // the buttons mid-click, so a click can land on an element that no longer
  // exists and do nothing.
  const cams = document.getElementById('camlist'), html = camList(s);
  if (cams.dataset.html !== html) { cams.innerHTML = html; cams.dataset.html = html; }

  const view = document.getElementById('view');
  if (s.solving) {
    view.innerHTML = `<div class="idle"><p>Solving ${s.active}...</p>
      <p class="sub">Fitting the model and scoring it on held-out views.</p></div>`;
    active = null;
  } else if (s.capture) {
    // Rebuild only when switching camera, so the stream is not restarted and
    // the buttons do not flicker on every poll.
    if (active !== s.capture.role) {
      view.innerHTML = liveView(s);
      active = s.capture.role;
    } else {
      updateLive(s);
    }
  } else {
    const html = idleView(s);
    if (view.dataset.html !== html) { view.innerHTML = html; view.dataset.html = html; }
    active = null;
  }
  if (s.capture || s.solving) view.dataset.html = '';
}

function updateLive(s) {
  const c = s.capture;
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.textContent = v;
  };
  const dds = document.querySelectorAll('#view .stats dd');
  if (dds.length >= 8) {
    dds[2].textContent = c.fps + ' fps';
    dds[3].textContent = c.corners;
    dds[4].textContent = c.sharpness;
    dds[5].innerHTML = '<b>' + c.stored + '</b> / ' + c.target;
    dds[6].textContent = c.coverage + '%';
    dds[7].innerHTML = c.tilt[0].toFixed(0) + '&ndash;' +
                       c.tilt[1].toFixed(0) + '&deg;';
  }
  const adv = document.querySelector('#view .advice');
  if (adv) adv.textContent = c.advice;
  const bar = document.querySelector('#view .bar i');
  if (bar) bar.style.width =
    Math.min(100, 100 * c.stored / Math.max(1, c.target)) + '%';
  const msg = document.getElementById('msg');
  if (msg) {
    msg.textContent = c.message || '';
    msg.className = 'msg ' + (/reject|no board|fail/i.test(c.message || '')
      ? 'bad' : /stored|removed/i.test(c.message || '') ? 'good' : '');
  }
  const btns = document.querySelectorAll('#view button.act');
  if (btns.length >= 3) {
    btns[1].disabled = !c.stored;
    btns[2].disabled = c.stored < s.min_views;
    btns[2].textContent = c.stored >= s.min_views
      ? 'Solve' : 'Solve (minimum ' + s.min_views + ')';
  }
}

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && active) { e.preventDefault(); grab(); }
});
setInterval(refresh, 500);
refresh();
</script></body></html>
"""


def page() -> str:
    return PAGE_HEAD + PAGE_SCRIPT


class Server(ThreadingHTTPServer):
    # Without this a restart fails for a minute while the old socket sits in
    # TIME_WAIT, which reads as a bug rather than a wait.
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
        # K and dist come back from OpenCV as ndarrays. Without this the dumps
        # raises inside the handler, /status never answers, and the page sits on
        # whatever it last saw -- which looks like a solve that never finishes.
        body = json.dumps(payload, default=storage.json_default)
        self._send(code, body, "application/json")

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
        parts = path.strip("/").split("/")

        if parts[0] == "start" and len(parts) == 2:
            if parts[1] not in ROLES:
                self._json({"ok": False, "message": "unknown camera"}, 404)
                return
            ok, message = self.session.start(parts[1])
            self._json({"ok": ok, "message": message})
        elif path == "/grab":
            capture = self.session.capture
            if capture is None:
                self._json({"ok": False, "message": "no camera running"}, 409)
                return
            capture.request_grab()
            self._json({"ok": True})
        elif path == "/undo":
            capture = self.session.capture
            if capture is None:
                self._json({"ok": False, "message": "no camera running"}, 409)
                return
            self._json({"ok": capture.drop_last()})
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
        description="Stage 1: camera intrinsics, in the browser")
    parser.add_argument("--target", type=int, default=TARGET_VIEWS,
                        help="views to aim for per camera")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    try:
        common.require_results("board")
        spec = common.load_board()
    except common.Aborted:
        return 1

    common.heading("Stage 1: camera intrinsics")
    print(f"  Board: {spec.name}, {spec.squares_x}x{spec.squares_y} squares, "
          f"{spec.square_mm:g} mm, {spec.dictionary}"
          f"{', legacy' if spec.legacy else ''}")
    if not spec.measured:
        print("\n  WARNING: this board was not measured with calipers. Every")
        print("  distance the calibration reports will carry the printer's")
        print("  scale error, and nothing downstream can detect it.")

    for role in ROLES:
        existing = storage.load_result(f"intrinsics_{role}")
        state = (f"done, holdout {existing['holdout_rms_px']:.4f} px"
                 if existing else "to do")
        print(f"  {role:<13}{state}")

    print("\n  Intrinsics are tied to both the lens focus and the resolution.")
    print("  Do not refocus or change resolution after this stage.")

    session = Session(spec, args.target)
    Handler.session = session
    try:
        server = Server((HOST, args.port), Handler)
    except OSError as exc:
        print(f"\n  Cannot listen on {HOST}:{args.port}: {exc}")
        print(f"  Another copy may already be running. Open "
              f"http://{HOST}:{args.port}")
        return 1

    print(f"\n  Open http://{HOST}:{args.port}")
    print("  Pick a camera there; only that one will stream.")
    print("  Ctrl-C here when you are done.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping...")
    finally:
        session.stop()
        server.shutdown()
        server.server_close()

    solved = [r for r in ROLES if storage.load_result(f"intrinsics_{r}")]
    common.heading("Stage 1 summary")
    for role in ROLES:
        result = storage.load_result(f"intrinsics_{role}")
        if result:
            print(f"  {role:<13}OK    holdout {result['holdout_rms_px']:.4f} px, "
                  f"fov {result['fov']['fovx_deg']:.1f} deg")
        else:
            print(f"  {role:<13}not calibrated")

    if len(solved) == len(ROLES):
        print("\n  All three cameras done. Next: python calibration/run.py --stage 2")
        return 0
    print(f"\n  {len(solved)} of {len(ROLES)} done. Rerun when ready.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except common.Aborted:
        print("\n  Stopped. Nothing further was saved.")
        raise SystemExit(130)
