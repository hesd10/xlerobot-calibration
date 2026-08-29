"""Interactive camera identification.

Lists every capture device on this machine, streams them side by side in the
browser, and lets you assign a role to each one by looking at the picture. The
result is saved to tools/config/camera_mapping.json.

This is the only reliable way to tell the three robot cameras apart: they share a
VID:PID and serial string, and /dev/videoN numbering changes between boots.

Usage:
    python tools/cameras/identify.py
    then open http://127.0.0.1:8421

Binds to localhost only.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.cameras import (  # noqa: E402
    FPS,
    HEIGHT,
    MAPPING_FILE,
    PHYSICAL_ROLES,
    WIDTH,
    list_capture_devices,
    load_mapping,
    save_mapping,
)

# The page records the side the operator SEES a camera on, not the model role.
# resolve() folds physical onto model later, once, for the declared mounting.
#
# The nominal side is the invariant one: it is fixed by the camera's name and
# the port it is plugged into. The physical side is what moves. Back-to-front,
# the body turns 180 degrees while the working side stays put, so each flange
# turns 180 degrees to keep facing forward and the arm that was on the left is
# now on the right. All this page can do is show a picture and ask which side it
# is seen on, which observes the moving quantity, so that is what gets stored.
ROLES = PHYSICAL_ROLES
ROLE_LABELS = {
    "left_wrist_physical": "left wrist (as you face the robot's working side)",
    "right_wrist_physical": "right wrist (as you face the robot's working side)",
    "head": "head",
}

HOST, PORT = "127.0.0.1", 8421

# Used to find earlier runs of this exact tool at startup, so a leftover
# instance that is still holding the cameras can be cleared before we open them.
SCRIPT_PATH = Path(__file__).resolve()


class Streamer(threading.Thread):
    """Grabs frames from one camera so HTTP clients always get the latest."""

    daemon = True

    def __init__(self, info: dict):
        super().__init__(name=info["device"])
        self.info = info
        self.device = info["device"]
        self.key = Path(info["device"]).name  # e.g. "video2", safe in URLs
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self.sharpness = 0.0
        self.brightness = 0.0
        self.fps = 0.0
        self.error: str | None = None
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def run(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.error = "could not open"
            return

        # MJPEG keeps several streams inside USB 2.0 bandwidth.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        last, frames = time.time(), 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                self.brightness = float(frame.mean())

                frames += 1
                now = time.time()
                if now - last >= 1.0:
                    self.fps = frames / (now - last)
                    last, frames = now, 0

                self._overlay(frame)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
        finally:
            cap.release()

    def _overlay(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        label = f"{self.device}  port {self.info.get('port', '?')}"
        cv2.putText(frame, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if self.info["is_robot_camera"] else (0, 200, 255), 1)


STREAMERS: dict[str, Streamer] = {}

PAGE_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>XLeRobot camera identification</title>
<style>
  body { background:#14161a; color:#e6e6e6; font-family:system-ui,sans-serif;
         margin:0; padding:22px; }
  h1 { font-size:19px; margin:0 0 6px; }
  p.hint { color:#9aa0a6; font-size:13px; margin:0 0 6px; max-width:900px; }
  .grid { display:flex; flex-wrap:wrap; gap:16px; margin-top:18px; }
  .cam { background:#1d2025; border:1px solid #2c3038; border-radius:8px;
         padding:11px; width:440px; }
  .cam.assigned { border-color:#3d8b40; }
  .cam.notrobot { opacity:.62; }
  .cam img { display:block; width:418px; height:314px; background:#000;
             border-radius:4px; }
  .cam h2 { font-size:14px; margin:0 0 8px; font-weight:600; }
  .cam .meta { color:#9aa0a6; font-size:12px; margin:8px 0;
               font-variant-numeric:tabular-nums; }
  select { background:#2c3038; color:#e6e6e6; border:1px solid #3a3f47;
           border-radius:5px; padding:7px 9px; font-size:13px; width:100%; }
  .bar { position:sticky; top:0; background:#14161a; padding:12px 0;
         border-bottom:1px solid #2c3038; z-index:5; }
  button { background:#2c3038; color:#e6e6e6; border:1px solid #3a3f47;
           border-radius:5px; padding:8px 15px; font-size:13px; cursor:pointer;
           margin-right:8px; }
  button.primary { background:#2f6f33; border-color:#3d8b40; }
  button:hover { filter:brightness(1.18); }
  #status { margin-left:6px; font-size:13px; color:#9aa0a6; }
  code { background:#24272d; padding:1px 5px; border-radius:3px; font-size:12px; }
  .tag { font-size:11px; padding:2px 6px; border-radius:3px; margin-left:6px; }
  .tag.robot { background:#26402a; color:#9fd9a3; }
  .tag.other { background:#3d3524; color:#e0c68a; }
</style></head><body>
<h1>Camera identification</h1>
<p class="hint">Wave a hand in front of one camera at a time and watch which
picture moves. Set that picture's dropdown to the matching role. Each role can
only be used once.</p>
<p class="hint">The three robot cameras are marked <span class="tag robot">robot</span>;
they are electrically identical, so the picture is the only way to tell them apart.
Devices marked <span class="tag other">other</span> are this laptop's built-in
webcam.</p>
<div class="bar">
  <button class="primary" onclick="save()">Save mapping</button>
  <button onclick="location.reload()">Reload</button>
  <span id="status"></span>
</div>
<div class="grid">
"""

PAGE_TAIL = """</div>
<script>
const ROLES = __ROLES__;

function selects() { return Array.from(document.querySelectorAll('select')); }

function refreshHighlight() {
  const used = new Map();
  for (const s of selects()) {
    const card = s.closest('.cam');
    card.classList.toggle('assigned', !!s.value);
    if (s.value) used.set(s.value, (used.get(s.value) || 0) + 1);
  }
  const dupes = [...used].filter(([, n]) => n > 1).map(([r]) => r);
  const st = document.getElementById('status');
  if (dupes.length) {
    st.textContent = 'Each role must be unique. Duplicated: ' + dupes.join(', ');
    st.style.color = '#ff6b6b';
    return false;
  }
  const missing = ROLES.filter(r => !used.has(r));
  if (missing.length) {
    st.textContent = 'Still unassigned: ' + missing.join(', ');
    st.style.color = '#9aa0a6';
    return false;
  }
  st.textContent = 'All roles assigned. Ready to save.';
  st.style.color = '#9fd9a3';
  return true;
}

async function save() {
  if (!refreshHighlight()) return;
  const roles = {};
  for (const s of selects()) if (s.value) roles[s.value] = s.dataset.key;
  const st = document.getElementById('status');
  try {
    const r = await fetch('/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ roles })
    });
    const t = await r.text();
    st.textContent = t;
    st.style.color = r.ok ? '#9fd9a3' : '#ff6b6b';
  } catch (e) {
    st.textContent = 'save failed: ' + e;
    st.style.color = '#ff6b6b';
  }
}

async function poll() {
  try {
    const s = await (await fetch('/stats')).json();
    for (const [k, v] of Object.entries(s)) {
      const el = document.getElementById('meta-' + k);
      if (el) el.textContent = v.error
        ? 'ERROR: ' + v.error
        : `bright ${v.brightness.toFixed(0)}  sharp ${v.sharpness.toFixed(0)}  ${v.fps.toFixed(1)} fps`;
    }
  } catch (e) {}
  setTimeout(poll, 600);
}

document.addEventListener('change', e => {
  if (e.target.tagName === 'SELECT') refreshHighlight();
});
refreshHighlight();
poll();
</script>
</body></html>
"""


def build_page() -> str:
    """Render one card per camera, preselecting any previously saved role."""
    previous = load_mapping().get("roles", {})
    # port -> role, so a saved camera keeps its role even if renumbered.
    port_to_role = {e.get("port"): r for r, e in previous.items() if e.get("port")}

    cards = []
    for s in STREAMERS.values():
        info = s.info
        preselect = port_to_role.get(info.get("port"), "")
        options = ['<option value="">-- not assigned --</option>']
        for role in ROLES:
            sel = " selected" if role == preselect else ""
            options.append(
                f'<option value="{role}"{sel}>{ROLE_LABELS.get(role, role)}</option>')

        tag = ('<span class="tag robot">robot</span>' if info["is_robot_camera"]
               else '<span class="tag other">other</span>')
        klass = "cam" + ("" if info["is_robot_camera"] else " notrobot")

        cards.append(
            f'<div class="{klass}">'
            f'<h2>{info["device"]}{tag}</h2>'
            f'<img src="/stream/{s.key}" alt="{info["device"]}">'
            f'<div class="meta" id="meta-{s.key}">connecting...</div>'
            f'<div class="meta">{info["name"]}<br>USB port <code>{info.get("port", "?")}</code></div>'
            f'<select data-key="{s.key}">{"".join(options)}</select>'
            f'</div>'
        )

    roles_json = json.dumps(list(ROLES))
    return PAGE_HEAD + "\n".join(cards) + PAGE_TAIL.replace("__ROLES__", roles_json)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self._send(build_page(), "text/html; charset=utf-8")
        elif self.path.startswith("/stream/"):
            self._stream(self.path.rsplit("/", 1)[-1])
        elif self.path == "/stats":
            stats = {
                k: {"sharpness": s.sharpness, "brightness": s.brightness,
                    "fps": s.fps, "error": s.error}
                for k, s in STREAMERS.items()
            }
            self._send(json.dumps(stats), "application/json")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            roles = payload["roles"]
        except Exception as exc:
            self._send(f"bad request: {exc}", "text/plain", code=400)
            return

        missing = [r for r in ROLES if r not in roles]
        if missing:
            self._send(f"not saved: missing roles {missing}", "text/plain", code=400)
            return
        if len(set(roles.values())) != len(roles):
            self._send("not saved: a camera was assigned to more than one role",
                       "text/plain", code=400)
            return

        assignments = {}
        for role, key in roles.items():
            streamer = STREAMERS.get(key)
            if streamer is None:
                self._send(f"not saved: unknown camera {key}", "text/plain", code=400)
                return
            info = streamer.info
            assignments[role] = {
                "device": info["device"],
                "port": info.get("port"),
                "name": info["name"],
                "vid": info.get("vid"),
                "pid": info.get("pid"),
            }

        save_mapping(assignments, note="assigned via tools/cameras/identify.py")
        summary = ", ".join(f"{r}={assignments[r]['device']}" for r in ROLES)
        print(f"\nsaved mapping: {summary}")
        print(f"  -> {MAPPING_FILE}")
        self._send(f"Saved to {MAPPING_FILE.name}: {summary}", "text/plain")

    def _stream(self, key):
        streamer = STREAMERS.get(key)
        if streamer is None:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                jpeg = streamer.snapshot()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send(self, body: str, content_type: str, code: int = 200):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _earlier_runs_of_this_script() -> list[int]:
    """PIDs of other Python processes actually running this script.

    Read straight from /proc rather than through pgrep, and match on this
    script being one of the interpreter's own arguments. A shell whose command
    line merely mentions the path -- an editor, a grep, this very launch -- does
    not qualify, so nothing but a genuine leftting run can be selected for a
    kill.
    """
    me = os.getpid()
    target = str(SCRIPT_PATH)
    found: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found  # Not Linux; skip rather than guess.

    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue  # Process gone or not ours to read.
        args = [a.decode("utf-8", "replace") for a in argv if a]
        if not args or "python" not in Path(args[0]).name:
            continue
        if any(_is_this_script(arg, target) for arg in args[1:]):
            found.append(pid)
    return found


def _is_this_script(arg: str, target: str) -> bool:
    if arg == target:
        return True
    try:
        return Path(arg).resolve() == SCRIPT_PATH
    except (OSError, ValueError):
        return False  # Not a usable path, so not this script.


def _free_the_cameras() -> None:
    """End any earlier run of this tool that is still holding the cameras.

    A previous window that was closed by the browser rather than by Ctrl-C can
    leave this process alive, and while it lives it keeps every /dev/videoN
    open. The next run then sees only black frames, because a V4L2 device can
    be streamed by one process at a time. This is the single most common reason
    the page shows no pictures, so we clear it before opening anything.

    Scoped to this exact script so nothing else on the machine is touched.
    """
    stale = _earlier_runs_of_this_script()
    if not stale:
        return

    print(f"closing {len(stale)} earlier camera-identification process(es) "
          "still holding the cameras")
    for pid in stale:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Give the kernel a moment to release the /dev/videoN handles before we
    # try to open them ourselves.
    for _ in range(20):
        if not any(_alive(pid) for pid in stale):
            break
        time.sleep(0.1)
    for pid in stale:
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(0.3)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--all", action="store_true",
                        help="also stream non-robot cameras (built-in webcam)")
    args = parser.parse_args()

    _free_the_cameras()

    devices = list_capture_devices(robot_only=not args.all)
    if not devices:
        print("no capture devices found")
        return 1

    print(f"found {len(devices)} capture device(s):")
    for d in devices:
        kind = "robot camera" if d["is_robot_camera"] else "other"
        print(f"  {d['device']:<14} {d['name'][:30]:<32} port {d.get('port', '?'):<10} {kind}")

    for d in devices:
        s = Streamer(d)
        s.start()
        STREAMERS[s.key] = s

    time.sleep(1.5)
    failed = [k for k, s in STREAMERS.items() if s.error]
    for k in failed:
        print(f"  WARNING: {STREAMERS[k].device}: {STREAMERS[k].error}")

    existing = load_mapping()
    if existing:
        print(f"\nexisting mapping from {existing.get('confirmed_at', '?')} "
              "will be preselected in the page")

    server = ThreadingHTTPServer((HOST, args.port), Handler)
    server.daemon_threads = True
    print(f"\nopen http://{HOST}:{args.port}  (localhost only)")
    print("assign each role, then click Save mapping. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        for s in STREAMERS.values():
            s.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
