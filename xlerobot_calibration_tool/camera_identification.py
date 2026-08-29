"""Run the camera identification tool from inside the dashboard.

The three robot cameras share a VID:PID and serial string, and /dev/videoN
numbering changes between boots, so telling them apart means looking at the
pictures. tools/cameras/identify.py already does exactly that in a browser;
this module starts it on a free port so the dashboard can embed it, rather
than asking the operator to open a second terminal.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = SOURCE_ROOT / "tools" / "cameras" / "identify.py"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# The tool was written to be opened directly, so its page asks for /stream/...,
# /stats and /save at the server root. Inside the dashboard's /cameras iframe
# those resolve against the dashboard instead, where nothing answers, and every
# panel stays blank. A <base> tag cannot fix root-absolute paths, so rewrite
# them as the page passes through the proxy.
PREFIX = "/cameras"
REWRITES = (('src="/stream/', f'src="{PREFIX}/stream/'),
            ("fetch('/stats'", f"fetch('{PREFIX}/stats'"),
            ("fetch('/save'", f"fetch('{PREFIX}/save'"))


# The identification tool now records the side the operator SEES a camera on
# (left_wrist_physical / right_wrist_physical), not the model role. That side does
# not move when the mounting declaration changes, so the saved assignment is
# mounting-invariant and the page needs no per-mounting relabelling: the fold onto
# a model role happens once, later, in config.cameras.resolve().
#
# All that is left to do here is point the tool's root-absolute URLs at the proxy
# prefix; the page was written to be opened directly at the server root.
def rewrite_page(html: str) -> str:
    """Point the page's root-absolute URLs at the proxy prefix."""
    for old, new in REWRITES:
        html = html.replace(old, new)
    return html


class CameraIdentification:
    """A single instance of the identification tool, owned by the dashboard."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.port: int | None = None

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def proxy_base(self) -> str | None:
        if not self.running() or self.port is None:
            return None
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> int:
        """Start the tool, or return the port of the instance already up."""
        if self.running() and self.port is not None:
            return self.port
        if not SCRIPT.is_file():
            raise FileNotFoundError(str(SCRIPT))
        port = _free_port()
        self.process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--port", str(port)],
            cwd=str(SOURCE_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        self.port = port
        # Opening several cameras takes a moment; wait for the socket so the
        # first page load does not race the server and show a proxy error.
        deadline = time.time() + 20
        while time.time() < deadline:
            if self.process.poll() is not None:
                self.port = None
                raise RuntimeError("identify")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                    return port
            except OSError:
                time.sleep(0.15)
        self.stop()
        raise TimeoutError("identify")

    def stop(self) -> None:
        process, self.process, self.port = self.process, None, None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
