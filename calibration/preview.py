"""Live view of all three robot cameras, for use throughout calibration.

A V4L2 device can only be opened by one process, so a calibration stage that
needs a camera and a preview that holds it cannot coexist. This server therefore
owns the cameras and lends frames out:

  - browser preview of all three at once, with board detection overlaid
  - an HTTP frame API that stages use instead of opening a device themselves

Run it once and leave it running for the whole session:

    python calibration/preview.py
    then open http://127.0.0.1:8420

Stages detect it automatically and take frames through it. If it is not running
they fall back to opening the device directly, which is fine when nothing else
needs the camera.

Binds to localhost only.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

CALIB = Path(__file__).resolve().parent
TOOLS = CALIB.parent / "tools"
for _p in (str(CALIB), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.cameras import FPS, HEIGHT, ROLES, WIDTH, resolve  # noqa: E402
from core import charuco  # noqa: E402

import frames  # noqa: E402

HOST, PORT = "127.0.0.1", 8420
BASE_URL = f"http://{HOST}:{PORT}"


def camera_label(role: str, mounting: str | None = None) -> str:
    """Name a camera by the side the operator can point at.

    This page is a picture of three cameras, and the only thing the operator
    can do with it is compare a label against what they see. Printing the
    stored role instead sends them to the opposite arm whenever the robot is
    used back-to-front, because the wrist roles are named after the arm they
    are bolted to and turn with the flanges.

    Never allowed to raise: a label is decoration, and a preview that dies
    because it could not read workflow.json is worse than one with the stored
    name on it.
    """
    try:
        return frames.spoken_camera(role, mounting or _mounting())
    except Exception:  # noqa: BLE001
        return role


def _mounting() -> str:
    """The declared mounting, or normal if the workspace cannot be read."""
    try:
        return frames.declared_mounting()
    except Exception:  # noqa: BLE001
        return frames.NORMAL

# How long a request to suppress the board overlay stands without being renewed.
# A live stage renews it every few seconds, so this only expires once the stage
# is gone, including when it was killed and could not clean up after itself.
DETECT_HOLD_S = 10.0


class CameraFeed:
    """Owns one camera and keeps the newest frame available.

    Holds a thread rather than subclassing Thread: the obvious attribute names
    for this job (_stop, _started) are ones Thread uses internally.
    """

    def __init__(self, role: str, device: str, width: int, height: int,
                 detector: charuco.BoardDetector | None = None):
        self.role = role
        self.label = camera_label(role)
        self.device = device
        self.width, self.height = width, height
        self.detector = detector

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"feed-{role}",
                                        daemon=True)

        self.error: str | None = None
        # Turned off while a stage detects on this camera, so the same expensive
        # work is not done twice per frame.
        self.detect_enabled = True
        self._detect_off_since: float | None = None
        self.fps = 0.0
        self.sharpness = 0.0
        self.brightness = 0.0
        self.n_corners = 0
        self.frame_index = 0

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout: float | None = None):
        self._thread.join(timeout)

    def set_detect(self, enabled: bool) -> None:
        self.detect_enabled = enabled
        self._detect_off_since = None if enabled else time.time()

    def detect_watchdog(self) -> None:
        """Turn the overlay back on if whoever disabled it went away.

        A stage killed with SIGKILL cannot restore anything, and a preview with a
        permanently blank overlay reads as broken software. Stages that are still
        alive keep refreshing the request, so this only fires once they are gone.
        """
        if self.detect_enabled or self._detect_off_since is None:
            return
        if time.time() - self._detect_off_since > DETECT_HOLD_S:
            print(f"  {self.role}: re-enabling board overlay "
                  f"(no stage refreshed it for {DETECT_HOLD_S:.0f}s)")
            self.set_detect(True)

    def pause(self):
        """Release the device so a stage can open it exclusively."""
        self._paused.set()

    def resume(self):
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def latest(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._frame is None:
                return None, self.frame_index
            return self._frame.copy(), self.frame_index

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def _relocate(self) -> bool:
        """Follow this camera if the kernel gave it a new /dev/video number.

        Re-plugging or a USB reset changes the device node while the physical port
        stays put, so the port-based mapping still knows where the camera is.
        Returns False when the camera is no longer on the bus at all, which is a
        cable or hub problem no amount of retrying will fix.
        """
        try:
            devices = resolve(strict=False)
        except Exception:
            return True  # cannot tell; let the open attempt decide

        current = devices.get(self.role)
        if current is None:
            self.error = (f"{self.role} is no longer on the USB bus "
                          f"(was {self.device}). Check its cable and hub.")
            return False
        if current != self.device:
            print(f"  {self.role}: moved from {self.device} to {current}")
            self.device = current
        return True

    def _loop(self):
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.1)
                continue

            if not self._relocate():
                self.fps = 0.0
                time.sleep(2.0)
                continue

            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not cap.isOpened():
                self.error = f"could not open {self.device}"
                self.fps = 0.0
                time.sleep(1.0)
                continue

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, FPS)
            self.error = None
            # Stale from before a pause, and a low reading looks like a fault.
            self.fps = 0.0

            last, frames, failures = time.time(), 0, 0
            try:
                while not self._stop.is_set() and not self._paused.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        # A camera that re-enumerates (say /dev/video2 becomes
                        # video3) leaves a handle that fails forever. Retrying it
                        # silently while still reporting the last known fps makes
                        # the feed look healthy but frozen, so give up and reopen.
                        failures += 1
                        if failures > 60:
                            self.error = (f"{self.device} stopped delivering "
                                          f"frames; reopening")
                            self.fps = 0.0
                            break
                        time.sleep(0.02)
                        continue

                    failures = 0
                    frames += 1
                    now = time.time()
                    if now - last >= 1.0:
                        self.fps = frames / (now - last)
                        last, frames = now, 0

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    self.brightness = float(frame.mean())

                    detection = None
                    if self.detector is not None and self.detect_enabled:
                        detection = self.detector.detect(gray)
                    self.n_corners = detection["n"] if detection else 0

                    with self._lock:
                        self._frame = frame.copy()
                        self.frame_index += 1

                    annotated = frame.copy()
                    self._overlay(annotated, detection)
                    ok, buf = cv2.imencode(".jpg", annotated,
                                           [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        with self._lock:
                            self._jpeg = buf.tobytes()
            finally:
                cap.release()

    def _overlay(self, frame, detection) -> None:
        h, w = frame.shape[:2]
        if detection is not None:
            for (px, py) in detection["corners"]:
                cv2.circle(frame, (int(px), int(py)), 3, (0, 240, 0), -1)

        cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
        if self.detector is not None and self.detect_enabled:
            colour = (0, 240, 0) if self.n_corners >= 6 else (0, 170, 240)
            text = f"{self.label}  corners {self.n_corners}"
        elif self.detector is not None:
            colour = (200, 200, 200)
            text = f"{self.label}  (a stage is detecting on this camera)"
        else:
            colour = (200, 200, 200)
            text = self.label
        text += f"  sharp {self.sharpness:.0f}  {self.fps:.0f}fps"
        cv2.putText(frame, text, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1)

    def stats(self) -> dict:
        return {
            "role": self.role,
            # What the operator reads. The role stays the key for every URL and
            # saved file; only the words follow the robot.
            "label": self.label,
            "device": self.device,
            "error": self.error,
            "paused": self.paused,
            "fps": round(self.fps, 1),
            "sharpness": round(self.sharpness, 1),
            "brightness": round(self.brightness, 1),
            "corners": self.n_corners,
            "width": self.width,
            "height": self.height,
            # Exposed so a frozen feed is detectable: fps alone can look healthy
            # while the stored frame never changes.
            "frame_index": self.frame_index,
        }


FEEDS: dict[str, CameraFeed] = {}

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>XLeRobot calibration cameras</title>
<style>
  body { background:#14161a; color:#e6e6e6; font-family:system-ui,sans-serif;
         margin:0; padding:18px; }
  h1 { font-size:18px; margin:0 0 4px; }
  p.hint { color:#9aa0a6; font-size:13px; margin:0 0 4px; max-width:900px; }
  .grid { display:flex; flex-wrap:wrap; gap:14px; margin-top:16px; }
  .cam { background:#1d2025; border:1px solid #2c3038; border-radius:8px;
         padding:10px; width:430px; }
  .cam.paused { border-color:#a5651f; }
  .cam.error { border-color:#8b3d3d; }
  .cam img { display:block; width:410px; height:308px; background:#000;
             border-radius:4px; }
  .cam h2 { font-size:14px; margin:0 0 7px; font-weight:600; }
  .meta { color:#9aa0a6; font-size:12px; margin-top:7px; line-height:1.7;
          font-variant-numeric:tabular-nums; }
  .meta b { color:#e6e6e6; font-weight:500; }
  .tag { font-size:11px; padding:2px 7px; border-radius:3px; margin-left:7px; }
  .tag.ok { background:#26402a; color:#9fd9a3; }
  .tag.paused { background:#3d3524; color:#e0c68a; }
  .tag.err { background:#40262a; color:#d99f9f; }
  code { background:#24272d; padding:1px 5px; border-radius:3px; font-size:12px; }
  .err { color:#d99f9f; font-size:12px; margin-top:5px; }
</style></head><body>
<h1>Calibration cameras</h1>
<p class="hint">Leave this running for the whole calibration. Stages take frames
through this server, so the cameras never have to be handed back and forth.</p>
<p class="hint">Green dots are detected board corners. A camera shows
<span class="tag paused">paused</span> while a stage is using it exclusively.</p>
<div class="grid" id="grid"></div>
<script>
let built = false;
const lastIndex = {};
async function poll() {
  let s;
  try { s = await (await fetch('/stats')).json(); } catch (e) { return; }
  const grid = document.getElementById('grid');
  if (!built) {
    grid.innerHTML = s.cameras.map(c => `
      <div class="cam" id="cam-${c.role}">
        <h2>${c.label || c.role}<span class="tag ok" id="tag-${c.role}"></span></h2>
        <img src="/stream/${c.role}">
        <div class="meta">
          corners <b id="corners-${c.role}">-</b> &middot;
          sharpness <b id="sharp-${c.role}">-</b> &middot;
          <b id="fps-${c.role}">-</b> fps<br>
          <code id="dev-${c.role}">-</code>
          <span id="res-${c.role}"></span>
          <div class="err" id="err-${c.role}"></div>
        </div>
      </div>`).join('');
    built = true;
  }
  for (const c of s.cameras) {
    const box = document.getElementById('cam-' + c.role);
    const tag = document.getElementById('tag-' + c.role);
    // A stalled feed still shows its last frame, which reads as live. Compare
    // frame indices between polls so a frozen camera is called out.
    const prev = lastIndex[c.role];
    const stalled = !c.paused && prev !== undefined && prev === c.frame_index;
    lastIndex[c.role] = c.frame_index;
    const bad = c.error || stalled;
    box.className = 'cam' + (bad ? ' error' : c.paused ? ' paused' : '');
    tag.className = 'tag ' + (bad ? 'err' : c.paused ? 'paused' : 'ok');
    tag.textContent = c.error ? 'error'
      : stalled ? 'no new frames'
      : c.paused ? 'in use by a stage' : 'live';
    const err = document.getElementById('err-' + c.role);
    if (err) err.textContent = c.error || '';
    document.getElementById('corners-' + c.role).textContent = c.corners;
    document.getElementById('sharp-' + c.role).textContent = c.sharpness;
    document.getElementById('fps-' + c.role).textContent = c.fps;
    document.getElementById('dev-' + c.role).textContent = c.device;
    document.getElementById('res-' + c.role).textContent =
      c.width + 'x' + c.height;
  }
}
setInterval(poll, 700); poll();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
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

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE)
        elif path == "/stats":
            self._send(200, json.dumps(
                {"cameras": [f.stats() for f in FEEDS.values()]}),
                "application/json")
        elif path == "/health":
            self._send(200, json.dumps({"service": "xlerobot-calibration-preview",
                                        "roles": list(FEEDS)}),
                       "application/json")
        elif path.startswith("/stream/"):
            self._stream(path.rsplit("/", 1)[1])
        elif path.startswith("/frame/"):
            self._frame(path.rsplit("/", 1)[1])
        else:
            self._send(404, "not found")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path.startswith("/pause/"):
            role = path.rsplit("/", 1)[1]
            if role in FEEDS:
                FEEDS[role].pause()
                # Give the capture loop time to release the device.
                time.sleep(0.4)
                self._send(200, json.dumps({"paused": role}), "application/json")
            else:
                self._send(404, "unknown role")
        elif path.startswith("/resume/"):
            role = path.rsplit("/", 1)[1]
            if role in FEEDS:
                FEEDS[role].resume()
                self._send(200, json.dumps({"resumed": role}), "application/json")
            else:
                self._send(404, "unknown role")
        elif path.startswith("/detect/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[1] in FEEDS:
                FEEDS[parts[1]].set_detect(parts[2] == "on")
                self._send(200, json.dumps({parts[1]: parts[2]}),
                           "application/json")
            else:
                self._send(404, "unknown role")
        else:
            self._send(404, "not found")

    def _frame(self, role: str):
        """One raw frame as PNG, so a stage gets undegraded pixels."""
        feed = FEEDS.get(role)
        if feed is None:
            self._send(404, "unknown role")
            return
        frame, index = feed.latest()
        if frame is None:
            self._send(503, "no frame yet")
            return
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            self._send(500, "encode failed")
            return
        payload = buf.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("X-Frame-Index", str(index))
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream(self, role: str):
        feed = FEEDS.get(role)
        if feed is None:
            self._send(404, "unknown role")
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpeg = feed.latest_jpeg()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpeg)).encode()
                                 + b"\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.06)
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(ThreadingHTTPServer):
    # Without this a restart fails for a minute while the old socket is in
    # TIME_WAIT, which looks like a bug rather than a wait.
    allow_reuse_address = True
    daemon_threads = True


def is_running(port: int = PORT) -> bool:
    """Is another instance of this server already listening?"""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/health",
                                    timeout=1.5) as r:
            return json.loads(r.read()).get(
                "service") == "xlerobot-calibration-preview"
    except (urllib.error.URLError, OSError, ValueError):
        return False


@contextmanager
def live_session(spec: charuco.BoardSpec | None = None, port: int = PORT + 1,
                 roles: list[str] | None = None):
    """A short-lived preview, for a stage that wants a live view of its own.

    Used when the board spec is not yet saved, so the long-running preview cannot
    know about it. Borrows the cameras from the long-running preview if it is up,
    and hands them back on exit.

    Yields the URL to open.
    """
    from core import preview_client

    devices = resolve(strict=False)
    wanted = [r for r in (roles or ROLES) if r in devices]
    if not wanted:
        raise RuntimeError("no cameras found; run tools/cameras/identify.py")

    detector = charuco.BoardDetector(spec, min_corners=4) if spec else None
    borrowed: list[str] = []
    if preview_client.is_running():
        for role in wanted:
            if role in preview_client.roles():
                preview_client.pause(role)
                borrowed.append(role)

    # A hard kill skips the finally block, which would leave the long-running
    # preview with every camera paused and nothing to un-pause them.
    def hand_back():
        for role in borrowed:
            try:
                preview_client.resume(role)
            except Exception:
                pass

    atexit.register(hand_back)

    feeds = {}
    server = None
    try:
        for role in wanted:
            feed = CameraFeed(role, devices[role], WIDTH, HEIGHT, detector)
            feeds[role] = feed
            FEEDS[role] = feed
            feed.start()

        for _ in range(60):
            if all(f.latest()[0] is not None or f.error for f in feeds.values()):
                break
            time.sleep(0.1)

        server = Server((HOST, port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"http://{HOST}:{port}", feeds
    finally:
        # Each step is isolated: a failure while tearing down our own feeds must
        # not stop the cameras being handed back, or the long-running preview is
        # left with everything paused and no way to notice.
        for feed in feeds.values():
            try:
                feed.stop()
            except Exception:
                pass
        for feed in feeds.values():
            try:
                feed.join(timeout=2.0)
            except Exception:
                pass
        for role in feeds:
            FEEDS.pop(role, None)
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        hand_back()
        atexit.unregister(hand_back)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live preview server for calibration")
    parser.add_argument("--no-detect", action="store_true",
                        help="skip board detection overlay (lower CPU)")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    devices = resolve(strict=False)
    if not devices:
        print("No cameras found. Run tools/cameras/identify.py first.")
        return 1

    # Bind before touching the cameras: a port clash otherwise takes the cameras
    # and then dies, leaving them held by a dead process's file descriptors.
    try:
        server = Server((HOST, args.port), Handler)
    except OSError as exc:
        print(f"Cannot listen on {HOST}:{args.port}: {exc}")
        if is_running(args.port):
            print("\nA preview server is already running. Just open "
                  f"http://{HOST}:{args.port}")
        else:
            print(f"\nSomething else holds the port. Find it with:\n"
                  f"  ss -ltnp 'sport = :{args.port}'\n"
                  f"Or pick another port with --port.")
        return 1

    detector = None
    if not args.no_detect:
        boards = charuco.load_boards()
        if boards:
            spec = next(iter(boards.values()))
            detector = charuco.BoardDetector(spec, min_corners=4)
            print(f"Detecting board '{spec.name}' "
                  f"({spec.squares_x}x{spec.squares_y}, {spec.dictionary})")
        else:
            print("No board recorded yet, so no detection overlay.")
            print("  Run: python calibration/run.py --stage prep")

    for role in frames.camera_order(_mounting()):
        if role not in devices:
            print(f"  {camera_label(role)}: not found")
            continue
        feed = CameraFeed(role, devices[role], WIDTH, HEIGHT, detector)
        FEEDS[role] = feed
        feed.start()

    if not FEEDS:
        print("No usable cameras.")
        server.server_close()
        return 1

    for _ in range(50):
        if all(f.latest()[0] is not None or f.error for f in FEEDS.values()):
            break
        time.sleep(0.1)

    print(f"\nServing {len(FEEDS)} cameras on http://{HOST}:{args.port}")
    for role, feed in FEEDS.items():
        state = feed.error or f"{feed.width}x{feed.height}"
        print(f"  {feed.label:<20} {feed.device:<14} {state}")
    print("\nLeave this running for the whole calibration. Ctrl-C to stop.")

    def watchdog():
        while True:
            time.sleep(2.0)
            for feed in list(FEEDS.values()):
                feed.detect_watchdog()

    threading.Thread(target=watchdog, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for feed in FEEDS.values():
            feed.stop()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
