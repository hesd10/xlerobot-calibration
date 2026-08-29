"""Client for the preview server, so stages and the preview can coexist.

A V4L2 device admits one opener, so a stage that opens a camera directly will
fail while the preview holds it. Rather than making the operator stop and start
the preview around every stage, a stage asks the server to hand the camera over
and gives it back afterwards.

Two modes:

  borrow(role)   the server pauses its own capture and the stage opens the device
                 exclusively. Best when the stage needs precise control of
                 resolution or timing.
  frames(role)   the server keeps capturing and the stage pulls raw frames over
                 HTTP. Best when several things want to watch at once.

If the server is not running, both fall back to opening the device directly.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from contextlib import contextmanager

import cv2
import numpy as np

HOST, PORT = "127.0.0.1", 8420
BASE_URL = f"http://{HOST}:{PORT}"
TIMEOUT = 2.0


def is_running() -> bool:
    """Is the preview server up?"""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=TIMEOUT) as r:
            return json.loads(r.read()).get(
                "service") == "xlerobot-calibration-preview"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def roles() -> list[str]:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=TIMEOUT) as r:
            return list(json.loads(r.read()).get("roles", []))
    except (urllib.error.URLError, OSError, ValueError):
        return []


def stats() -> list[dict]:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/stats", timeout=TIMEOUT) as r:
            return json.loads(r.read()).get("cameras", [])
    except (urllib.error.URLError, OSError, ValueError):
        return []


def _post(path: str) -> bool:
    try:
        req = urllib.request.Request(f"{BASE_URL}{path}", method="POST")
        with urllib.request.urlopen(req, timeout=TIMEOUT):
            return True
    except (urllib.error.URLError, OSError):
        return False


def pause(role: str) -> bool:
    """Ask the server to release one camera."""
    return _post(f"/pause/{role}")


def resume(role: str) -> bool:
    return _post(f"/resume/{role}")


def set_detect(role: str, enabled: bool) -> bool:
    """Turn the server's board-detection overlay on or off for one camera.

    Detection costs ~30ms a frame at this resolution. When a stage is already
    detecting on the same camera, having the server do it too roughly halves the
    frame rate both of them see for no benefit.
    """
    return _post(f"/detect/{role}/{'on' if enabled else 'off'}")


def get_frame(role: str) -> tuple[np.ndarray | None, int]:
    """Pull one raw frame from the server. PNG, so pixels are unaltered."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/frame/{role}",
                                    timeout=TIMEOUT) as r:
            index = int(r.headers.get("X-Frame-Index", "0"))
            data = np.frombuffer(r.read(), dtype=np.uint8)
    except (urllib.error.URLError, OSError, ValueError):
        return None, 0
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return frame, index


@contextmanager
def borrowed(role: str):
    """Hold a camera exclusively, returning it to the preview afterwards.

    Yields True when the preview was paused for us, False when it was not running
    and the caller simply has the device to itself.
    """
    running = is_running() and role in roles()
    if running:
        pause(role)
    try:
        yield running
    finally:
        if running:
            resume(role)
