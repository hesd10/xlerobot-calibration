"""Guided, one-camera-at-a-time Stage 1 preparation page."""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# This file runs inside an isolated copy of calibration/. Reuse the stable camera
# ownership and ChArUco code without changing the legacy source tree.
_RUNTIME_ROOT = Path(__file__).resolve().parent.parent
_CALIBRATION = _RUNTIME_ROOT / "calibration"
for _path in (_CALIBRATION, _CALIBRATION / "stages", _RUNTIME_ROOT / "tools", _RUNTIME_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_CAMERA_ORDER = ("head", "left_wrist", "right_wrist")
try:
    from config.cameras import ROLES as _CONFIGURED_ROLES
except Exception:
    _CONFIGURED_ROLES = _CAMERA_ORDER
ROLES = tuple(role for role in _CAMERA_ORDER if role in _CONFIGURED_ROLES)
ROLES += tuple(role for role in _CONFIGURED_ROLES if role not in ROLES)

def _mounting() -> str:
    """How the robot is standing, as declared for this workspace.

    Read at call time rather than at import so the page follows the switch at
    the top of the dashboard without the process being restarted.
    """
    try:
        import frames
        return frames.declared_mounting()
    except Exception:  # noqa: BLE001 - a label must never stop the stage
        return "normal"


def labels(mounting: str | None = None) -> dict[str, str]:
    """Name each camera by the side the operator can point at.

    The wrist cameras are bolted to the arms, so they turn with them: used
    back-to-front, the camera stored as `right_wrist` is the one on the
    operator's left. Labelling by the stored name sends them to the wrong arm
    while they are holding the board in front of it, and nothing downstream can
    detect the mix-up because the images are filed under the stored role.

    Keys stay model-named, because that is what the feed URLs and the saved
    intrinsics use; only the words change.
    """
    if mounting is None:
        mounting = _mounting()
    try:
        import frames
        return {"head": "Head camera",
                frames.named_camera("left", mounting): "Left wrist camera",
                frames.named_camera("right", mounting): "Right wrist camera"}
    except Exception:  # noqa: BLE001
        return {"head": "Head camera", "left_wrist": "Left wrist camera",
                "right_wrist": "Right wrist camera"}


def ordered_roles(mounting: str | None = None) -> tuple[str, ...]:
    """The configured roles, head first then the wrists physically left to right."""
    if mounting is None:
        mounting = _mounting()
    try:
        import frames
        preferred = ("head",) + tuple(frames.named_camera(side, mounting)
                                      for side in frames.SIDES)
    except Exception:  # noqa: BLE001
        preferred = _CAMERA_ORDER
    ordered = tuple(role for role in preferred if role in ROLES)
    return ordered + tuple(role for role in ROLES if role not in ordered)


# Normal-mounting fallback for callers and tests that index it directly.
LABELS = {"head": "Head camera", "left_wrist": "Left wrist camera",
          "right_wrist": "Right wrist camera"}


def _preview_corner_points(corners):
    """Return OpenCV-5-compatible integer points for preview drawing."""
    try:
        import numpy as np
        points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(points) < 2 or not np.isfinite(points).all():
        return None
    return np.rint(points).astype(np.int32).reshape(-1, 1, 2)


class Feed:
    """Own exactly one camera and expose its latest annotated JPEG."""

    def __init__(self, role: str, spec):
        self.role = role
        self.spec = spec
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.first_frame = threading.Event()
        self.corners = 0
        self.error: str | None = None
        self.device: str | None = None
        self.running = True
        self._cap = None
        self._camera_common = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            import cv2
            import common
            from core import charuco, gates

            self._camera_common = common
            cap, device = common.open_camera(self.role)
            detector = charuco.BoardDetector(spec=self.spec,
                                              min_corners=gates.PNP_MIN_CORNERS)
            with self.lock:
                self._cap = cap
                self.device = device
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    with self.lock:
                        self.error = f"{device} produced no frame"
                    time.sleep(0.05)
                    continue
                detection = detector.detect(frame)
                corners = 0 if detection is None else int(detection["n"])
                if detection is not None:
                    points = _preview_corner_points(detection["corners"])
                    if points is not None:
                        for point in points:
                            x, y = point[0]
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 220, 0), -1)
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    with self.lock:
                        self.jpeg = encoded.tobytes()
                        self.corners = corners
                        self.error = None
                    self.first_frame.set()
                    self.ready.set()
        except Exception as exc:
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
        finally:
            cap = self._cap
            if cap is not None:
                cap.release()
            if self._camera_common is not None:
                try:
                    self._camera_common.release_camera(self.role)
                except Exception:
                    pass

    def snapshot(self) -> tuple[bytes | None, int, str | None, str | None]:
        with self.lock:
            return self.jpeg, self.corners, self.error, self.device

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=3)
        with self.lock:
            cap = self._cap
        if cap is not None:
            cap.release()


class App:
    def __init__(self, workspace: Path, port: int):
        self.workspace = workspace
        self.port = port
        self.mounting = _mounting()
        self.roles = ordered_roles(self.mounting)
        self.labels = labels(self.mounting)
        self.lock = threading.Lock()
        self.spec = None
        self.feed: Feed | None = None
        self.selected_role: str | None = None
        self.mapping_error: str | None = None
        self.visited_roles: set[str] = set()

    def _stop_feed(self) -> None:
        feed = self.feed
        self.feed = None
        if feed is not None:
            feed.stop()

    def configure(self, body: dict) -> dict:
        try:
            from core.charuco import BoardSpec
            sx, sy = int(body["squares_x"]), int(body["squares_y"])
            square, marker = float(body["square_mm"]), float(body["marker_mm"])
            if not (math.isfinite(square) and math.isfinite(marker)):
                raise ValueError("Square size and marker size must be finite numbers")
            if square <= 0 or marker <= 0:
                raise ValueError("Square size and marker size must be greater than 0")
            if square > 1000 or marker > 1000:
                raise ValueError("The board dimensions must be smaller than 1000 mm")
            dictionary = str(body["dictionary"])
            spec = BoardSpec(
                squares_x=sx, squares_y=sy, square_mm=square, marker_mm=marker,
                dictionary=dictionary, legacy=bool(body.get("legacy")),
                name="main", measured=True)
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

        with self.lock:
            self._stop_feed()
            self.spec = spec
            self.selected_role = None
            self.visited_roles.clear()
            self.mapping_error = None
        return {"ok": True, "message": "Board parameters confirmed. Pick a camera to start checking."}

    def switch(self, role: str) -> dict:
        if role not in ROLES:
            return {"ok": False, "error": f"Unknown camera role: {role}"}
        with self.lock:
            if self.spec is None:
                return {"ok": False, "error": "Confirm the board parameters first."}
            self._stop_feed()
            self.selected_role = role
            feed = Feed(role, self.spec)
            self.feed = feed
        feed.ready.wait(timeout=8.0)
        feed.first_frame.wait(timeout=8.0)
        _, _, error, device = feed.snapshot()
        if error or not device or not feed.first_frame.is_set():
            with self.lock:
                if self.feed is feed:
                    self._stop_feed()
            return {"ok": False,
                    "error": error or f"{self.labels[role]} timed out while starting"}
        with self.lock:
            self.visited_roles.add(role)
        return {"ok": True, "role": role}

    def status(self) -> dict:
        with self.lock:
            feed = self.feed
            selected = self.selected_role
            configured = self.spec is not None
        visited = sorted(self.visited_roles)
        if feed is None:
            return {"configured": configured, "selected_role": selected,
                    "camera": None, "visited_roles": visited,
                    "can_complete": configured and set(visited) == set(ROLES)}
        _, corners, error, device = feed.snapshot()
        return {"configured": configured, "selected_role": selected,
                "camera": {"role": feed.role, "device": device,
                           "corners": corners, "error": error},
                "visited_roles": visited,
                "can_complete": configured and set(visited) == set(ROLES)}

    def complete(self) -> dict:
        with self.lock:
            spec = self.spec
            visited = set(self.visited_roles)
        if spec is None:
            return {"ok": False, "error": "Confirm the board parameters first."}
        missing = [role for role in self.roles if role not in visited]
        if missing:
            return {"ok": False, "error": "Check these one by one first: "
                                          + ", ".join(self.labels[r] for r in missing)}
        payload = {"boards": {"main": vars(spec)}, "world_board": "main",
                   "source": "unified_stage1_web"}
        calibration = self.workspace / "calibration"
        result = calibration / "results" / "board.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        (calibration / "board.json").write_text(
            json.dumps(payload["boards"], ensure_ascii=False, indent=2) + "\n")
        return {"ok": True}


def page(mounting: str | None = None) -> bytes:
    role_labels = labels(mounting)
    tabs = "".join(
        f'<button class="tab" data-role="{role}" onclick="switchCamera(\'{role}\')">'
        f'{role_labels[role]}</button>' for role in ordered_roles(mounting)
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage 1</title>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--text:#17202a;--muted:#5b6672;--line:#d9dee5;--accent:#1769aa;--ok:#087443;--bad:#b42318}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1000px;margin:auto;padding:20px}}h1{{margin:0 0 4px}}h2{{margin:0 0 8px}}.muted{{color:var(--muted)}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:18px;margin-top:18px}}.form{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}label{{display:grid;gap:4px;font-weight:600}}input,select{{padding:9px;border:1px solid var(--line);border-radius:5px;background:var(--panel);color:inherit}}.wide{{grid-column:1/-1}}button{{padding:10px 15px;border:1px solid var(--line);border-radius:5px;background:var(--panel);color:inherit;font-weight:650;cursor:pointer}}button.primary{{background:var(--accent);border-color:var(--accent);color:white}}button:disabled{{opacity:.5;cursor:not-allowed}}.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}}.tab.active{{background:var(--accent);border-color:var(--accent);color:white}}.camera{{display:none}}.camera.active{{display:block}}.camera img{{display:block;width:100%;max-height:65vh;aspect-ratio:4/3;object-fit:contain;background:#111}}.status{{margin:8px 0;color:var(--muted)}}.error{{color:var(--bad);font-weight:600}}.success{{color:var(--ok);font-weight:600}}#cameraPanel{{display:none}}#cameraPanel.visible{{display:block}}@media(max-width:700px){{.form{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Stage 1 &middot; Setup and calibration board</h1>
<p class="muted">Enter the board parameters measured from your print and confirm them; the cameras start only after that. One camera is opened at a time &mdash; use the tabs to switch.</p>
<section id="boardPanel" class="panel"><h2>Step 1: confirm the board parameters</h2><p class="muted">The square counts are squares, not corners. Measure the printed sizes with calipers. The defaults are for this project's board.</p>
<form id="board" class="form"><label>Squares across<input name="squares_x" type="number" min="3" value="10" required></label>
<label>Squares down<input name="squares_y" type="number" min="3" value="14" required></label>
<label>Square size (mm)<input name="square_mm" type="number" min="1" step="0.01" value="20" required></label>
<label>ArUco marker size (mm)<input name="marker_mm" type="number" min="0.1" step="0.01" value="15" required></label>
<label>Dictionary<select name="dictionary"><option>DICT_4X4_1000</option><option>DICT_5X5_1000</option><option>DICT_6X6_1000</option></select></label>
<label>Legacy OpenCV pattern<input name="legacy" type="checkbox" checked></label>
<div class="wide"><button class="primary" type="submit">Confirm parameters and check cameras</button></div></form><div id="message"></div></section>
<section id="cameraPanel" class="panel"><h2>Step 2: check each camera</h2><p class="muted">Pick a tab. Hold the board in the current camera's view, check that the image and the green detection dots look right, then move to the next camera.</p>
<div class="tabs">{tabs}</div><div id="cameraView"></div><p id="cameraStatus" class="status">Pick a camera.</p>
<button id="complete" class="primary" onclick="completeStage()" disabled>All three cameras checked &mdash; finish Stage 1</button></section></main>
<script>
let configured=false,selected=null;
const labels={json.dumps(role_labels, ensure_ascii=False)};
function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
async function api(path,opts={{}}){{const r=await fetch(path,{{cache:'no-store',...opts}});const text=await r.text();let d;try{{d=JSON.parse(text)}}catch{{throw new Error(text||`HTTP ${{r.status}}`)}}if(!r.ok)throw new Error(d.error||`HTTP ${{r.status}}`);return d}}
async function switchCamera(role){{try{{const d=await api('/switch',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{role}})}});selected=d.role;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.role===role));cameraView.innerHTML=`<img alt="${{esc(labels[role])}}" src="/feed/${{role}}?t=${{Date.now()}}">`;cameraStatus.textContent='Starting '+labels[role]+'\u2026'}}catch(e){{cameraStatus.innerHTML=`<span class="error">${{esc(e.message)}}</span>`}}}}
async function refresh(){{if(!configured)return;try{{const d=await api('/status');if(d.camera){{cameraStatus.textContent=(d.camera.device||'device starting')+' \u00b7 ChArUco corners '+(d.camera.corners||0)+(d.camera.error?' · '+d.camera.error:'')}}document.getElementById('complete').disabled=!d.can_complete}}catch(e){{cameraStatus.textContent=e.message}}}}
setInterval(refresh,1000);
document.getElementById('board').onsubmit=async e=>{{e.preventDefault();message.textContent='Checking parameters\u2026';const body=Object.fromEntries(new FormData(e.target));body.legacy=e.target.legacy.checked;try{{await api('/configure',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});configured=true;boardPanel.style.display='none';cameraPanel.classList.add('visible');message.textContent='';cameraStatus.textContent='Parameters confirmed. Pick a camera tab.'}}catch(err){{message.innerHTML=`<p class="error">${{esc(err.message)}}</p>`}}}};
async function completeStage(){{const button=document.getElementById('complete');button.disabled=true;try{{await api('/complete',{{method:'POST'}});cameraStatus.innerHTML='<span class="success">Stage 1 is complete; submitting and moving on to Stage 2.</span>';window.parent.postMessage({{type:'stage-completed',stage:'prepare'}},window.location.origin)}}catch(e){{button.disabled=false;cameraStatus.innerHTML=`<span class="error">${{esc(e.message)}}</span>`}}}}
</script></body></html>"""
    return html.encode()


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_body(self, body: bytes, content: str = "application/json", status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_body(page(app.mounting), "text/html; charset=utf-8")
            elif path == "/status":
                self.send_body(json.dumps(app.status(), ensure_ascii=False).encode())
            elif path.startswith("/feed/"):
                role = path.rsplit("/", 1)[-1]
                with app.lock:
                    feed = app.feed
                    selected = app.selected_role
                if not feed or selected != role:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while feed.running and app.selected_role == role:
                    jpeg, _, _, _ = feed.snapshot()
                    if jpeg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.05)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if path == "/configure":
                    result = app.configure(body)
                elif path == "/switch":
                    result = app.switch(str(body.get("role", "")))
                elif path == "/complete":
                    result = app.complete()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_body(json.dumps(result, ensure_ascii=False).encode(),
                               status=200 if result.get("ok") else 400)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self.send_body(json.dumps({"ok": False, "error": str(exc)},
                                          ensure_ascii=False).encode(), status=400)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    app = App(args.workspace, args.port)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        app._stop_feed()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
