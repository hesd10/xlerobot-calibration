"""Live preview of the three XLeRobot cameras in the browser.

Two jobs:
  1. Confirm the /dev/videoN mapping in xlerobot_cameras.py matches reality --
     each stream is labelled with its role, so a wrong label is visible at once.
  2. Manual focus: each stream shows a live sharpness score, so you can turn the
     lens ring until it peaks.

Usage:
    python focus_cameras.py
    then open http://127.0.0.1:8420

Binds to localhost only: the streams are not exposed to the network.
"""

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.cameras import FPS, HEIGHT, WIDTH, resolve, v4l_name  # noqa: E402

HOST, PORT = "127.0.0.1", 8420


def sharpness(gray) -> float:
    """Variance of Laplacian: higher means better focus."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class CameraWorker(threading.Thread):
    """Continuously grabs frames so HTTP clients always get the newest one."""

    daemon = True

    def __init__(self, role: str, device: str):
        super().__init__(name=role)
        self.role = role
        self.device = device
        self.v4l_name = v4l_name(device) or "unknown"
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self.score = 0.0
        self.peak = 0.0
        self.fps = 0.0
        self.error: str | None = None
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def reset_peak(self):
        self.peak = 0.0

    def run(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.error = f"could not open {self.device}"
            return

        # MJPEG keeps three 640x480 streams inside USB 2.0 bandwidth.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        last_t, frames = time.time(), 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.score = sharpness(gray)
                self.peak = max(self.peak, self.score)

                frames += 1
                now = time.time()
                if now - last_t >= 1.0:
                    self.fps = frames / (now - last_t)
                    last_t, frames = now, 0

                self._draw_overlay(frame, gray)

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
        finally:
            cap.release()

    def _draw_overlay(self, frame, gray):
        """Center focus box, sharpness readout, and a peak-relative bar."""
        h, w = frame.shape[:2]
        cx, cy, half = w // 2, h // 2, 80
        cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 255), 1)

        pct = (self.score / self.peak * 100.0) if self.peak > 1 else 0.0
        # Green once within 5% of the best sharpness seen so far.
        color = (0, 255, 0) if pct >= 95 else (0, 165, 255) if pct >= 75 else (0, 0, 255)

        cv2.rectangle(frame, (0, 0), (w, 58), (0, 0, 0), -1)
        cv2.putText(frame, f"{self.role}   {self.device}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"sharp {self.score:7.1f}  peak {self.peak:7.1f}  {pct:5.1f}%  {self.fps:4.1f}fps",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        bar_w = int((w - 16) * min(pct, 100.0) / 100.0)
        cv2.rectangle(frame, (8, h - 18), (8 + bar_w, h - 8), color, -1)
        cv2.rectangle(frame, (8, h - 18), (w - 8, h - 8), (200, 200, 200), 1)


WORKERS: dict[str, CameraWorker] = {}
SNAP_DIR = Path(__file__).resolve().parent.parent / "captures"

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>XLeRobot camera focus</title>
<style>
  body { background:#14161a; color:#e6e6e6; font-family:system-ui,sans-serif; margin:0; padding:20px; }
  h1 { font-size:18px; font-weight:600; margin:0 0 4px; }
  p.hint { color:#9aa0a6; font-size:13px; margin:0 0 18px; }
  .grid { display:flex; flex-wrap:wrap; gap:16px; }
  .cam { background:#1d2025; border:1px solid #2c3038; border-radius:8px; padding:10px; }
  .cam img { display:block; width:520px; height:390px; background:#000; border-radius:4px; }
  .cam h2 { font-size:14px; margin:0 0 8px; font-weight:600; }
  .cam .meta { color:#9aa0a6; font-size:12px; margin-top:8px; font-variant-numeric:tabular-nums; }
  button { background:#2c3038; color:#e6e6e6; border:1px solid #3a3f47; border-radius:5px;
           padding:7px 13px; font-size:13px; cursor:pointer; margin-right:8px; }
  button:hover { background:#363b44; }
  .err { color:#ff6b6b; font-size:13px; }
</style></head><body>
<h1>XLeRobot cameras</h1>
<p class="hint"><b>Check the labels first.</b> Wave a hand in front of each camera and confirm the
motion shows up in the stream carrying the matching role label. If a label is wrong, fix the mapping
in <code>xlerobot_cameras.py</code>.</p>
<p class="hint"><b>Then focus.</b> Aim each camera at something with fine detail (printed text works
well) at its normal working distance, then turn the lens ring slowly. The bar is relative to the
sharpest frame seen since the last reset, so reset after each adjustment. Green means you are at the
best focus found so far.</p>
<div>
  <button onclick="fetch('/reset').then(()=>0)">Reset peaks</button>
  <button onclick="fetch('/snapshot').then(r=>r.text()).then(t=>alert(t))">Save snapshots</button>
</div>
<div class="grid">__CAMS__</div>
<script>
async function poll() {
  try {
    const r = await fetch('/stats');
    const s = await r.json();
    for (const [k, v] of Object.entries(s)) {
      const el = document.getElementById('meta-' + k);
      if (el) el.textContent = v.error
        ? 'ERROR: ' + v.error
        : `sharpness ${v.score.toFixed(1)}  |  peak ${v.peak.toFixed(1)}  |  ${v.pct.toFixed(1)}%  |  ${v.fps.toFixed(1)} fps`;
    }
  } catch (e) {}
  setTimeout(poll, 500);
}
poll();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self._page()
        elif self.path.startswith("/stream/"):
            self._stream(self.path.rsplit("/", 1)[-1])
        elif self.path == "/stats":
            self._stats()
        elif self.path == "/reset":
            for w in WORKERS.values():
                w.reset_peak()
            self._text("ok")
        elif self.path == "/snapshot":
            self._snapshot()
        else:
            self.send_error(404)

    def _page(self):
        cards = []
        for role, w in WORKERS.items():
            cards.append(
                f'<div class="cam"><h2>{role} &mdash; <code>{w.device}</code></h2>'
                f'<img src="/stream/{role}" alt="{role}">'
                f'<div class="meta" id="meta-{role}">connecting...</div></div>'
            )
        self._html(PAGE.replace("__CAMS__", "\n".join(cards)))

    def _stream(self, name):
        worker = WORKERS.get(name)
        if worker is None:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                jpeg = worker.snapshot()
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

    def _stats(self):
        import json
        out = {}
        for name, w in WORKERS.items():
            out[name] = {
                "score": w.score,
                "peak": w.peak,
                "pct": (w.score / w.peak * 100.0) if w.peak > 1 else 0.0,
                "fps": w.fps,
                "error": w.error,
            }
        self._json(json.dumps(out))

    def _snapshot(self):
        SNAP_DIR.mkdir(exist_ok=True)
        saved = []
        for role, w in WORKERS.items():
            jpeg = w.snapshot()
            if jpeg:
                dest = SNAP_DIR / f"focus_{role}.jpg"
                dest.write_bytes(jpeg)
                saved.append(f"{dest.name} (sharpness {w.score:.1f})")
        self._text("saved:\n" + "\n".join(saved) if saved else "nothing to save")

    def _html(self, body: str):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, body: str):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _text(self, body: str):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--skip-verify", action="store_true",
                        help="stream even if the device check fails")
    args = parser.parse_args()

    try:
        cameras = resolve(strict=not args.skip_verify)
    except Exception as exc:
        print(f"WARNING: {exc}")
        if not args.skip_verify:
            print("\nRefusing to start. Pass --skip-verify to stream anyway.")
            return 1
        cameras = resolve(strict=False)

    if not cameras:
        print("no cameras resolved -- run: python tools/cameras/identify.py")
        return 1

    for role, device in cameras.items():
        if not Path(device).exists():
            print(f"skipping {role}: {device} not present")
            continue
        w = CameraWorker(role, device)
        w.start()
        WORKERS[role] = w

    if not WORKERS:
        print("no cameras available")
        return 1

    time.sleep(1.5)
    for role, w in WORKERS.items():
        state = f"ERROR: {w.error}" if w.error else "streaming"
        print(f"  {role:<12} {w.device:<14} {state}")

    server = ThreadingHTTPServer((HOST, args.port), Handler)
    server.daemon_threads = True
    print(f"\nopen http://{HOST}:{args.port}  (localhost only)")
    print("Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        for w in WORKERS.values():
            w.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
