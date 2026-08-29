"""Single-port guided calibration dashboard."""
from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from . import mounting
from .camera_identification import CameraIdentification, rewrite_page
from .diagnostics import payload as diagnostics_payload
from .diagnostics import render_checks, run_startup_checks
from .diagnostics import startup_payload as run_startup_diagnostics
from .engine import WorkflowEngine
from .i18n import LocalizedRuntimeError, catalog, message_of, text
from .legacy_runtime import LegacyRuntime
from .registry import BY_KEY, STAGES
from .result_summary import summarize_stage
from .workspace import Workspace


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__T_app_title__</title><style>
:root{color-scheme:light dark;--bg:#f4f6f8;--panel:#fff;--text:#17202a;--muted:#5b6672;--line:#d9dee5;--accent:#1769aa;--ok:#087443;--warn:#a15c00;--bad:#b42318}
@media(prefers-color-scheme:dark){:root{--bg:#111417;--panel:#191d21;--text:#edf1f5;--muted:#aeb7c2;--line:#363d45;--accent:#60a5dc;--ok:#55c993;--warn:#ffb65c;--bad:#ff8178}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif;letter-spacing:0}.shell{max-width:1600px;margin:auto;padding:20px}.top{display:flex;justify-content:space-between;gap:20px;align-items:start}.eyebrow{color:var(--accent);font-weight:700}.top h1{font-size:26px;margin:4px 0}.topright{display:flex;flex-direction:column;align-items:flex-end;gap:8px}.muted{color:var(--muted)}.layout{display:grid;grid-template-columns:330px 1fr;gap:14px;margin-top:18px}
/* A stage page has its own fixed-width layout (stage 6 needs ~1140px for its
   video box plus side panel). While one is open the sidebar is collapsed and
   the padding dropped, so the iframe gets the full window instead of what is
   left over after the dashboard chrome. */
.layout.stage-open{grid-template-columns:1fr}.layout.stage-open .sidebar{display:none}.layout.stage-open .content{padding:12px}.stage-frame{width:100%;height:78vh;min-height:760px;border:1px solid var(--line);border-radius:6px;background:white;display:block}.stage-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px}.stage-head .actions{margin:0}.panel{background:var(--panel);border:1px solid var(--line);border-radius:7px}.sidebar{padding:8px}.stage{width:100%;display:grid;grid-template-columns:32px 1fr auto;gap:8px;text-align:left;border:0;border-bottom:1px solid var(--line);padding:12px 9px;background:transparent;color:inherit;cursor:pointer}.stage:last-child{border:0}.stage.active{background:color-mix(in srgb,var(--accent) 10%,transparent)}.stage .num{color:var(--accent);font-weight:800}.badge{font-size:12px;padding:2px 7px;border-radius:4px;background:var(--bg);white-space:nowrap}.badge.completed{color:var(--ok)}.badge.completed_large_error{color:var(--warn)}.badge.blocked{color:var(--muted)}.badge.ready,.badge.authorized,.badge.running,.badge.review{color:var(--accent)}.content{padding:22px}.current{border-left:4px solid var(--accent);padding:14px 16px;background:var(--bg);margin-bottom:18px}.content h2{font-size:22px;margin:0 0 4px}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}.step{padding:12px;border-top:2px solid var(--line)}.step strong{display:block;margin-bottom:4px}.reason{padding:12px;background:var(--bg);margin:14px 0}.startup{border:1px solid var(--warn);padding:14px;margin-top:18px}.startup.ok{border-color:var(--ok);background:color-mix(in srgb,var(--ok) 8%,var(--panel))}.startup.failed{border-color:var(--bad);background:color-mix(in srgb,var(--bad) 7%,var(--panel))}.check{padding:6px 0;border-top:1px solid var(--line)}.actions{display:flex;gap:8px;flex-wrap:wrap}button.action{padding:10px 15px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:inherit;cursor:pointer}button.primary{background:var(--accent);border-color:var(--accent);color:white}button:disabled{opacity:.5;cursor:not-allowed}.check{white-space:pre-wrap}.error{color:var(--bad);font-weight:600}.success{color:var(--ok);font-weight:600}.confirm{border:1px solid var(--warn);padding:14px;margin-top:12px}.files{display:flex;gap:7px;flex-wrap:wrap;margin-top:8px}.file{font-family:monospace;font-size:12px;background:var(--bg);padding:4px 7px}.interaction{border-left:4px solid var(--accent);padding:18px;background:var(--bg)}.interaction h2{margin:0 0 8px}.caution{border:1px solid var(--warn);padding:10px 12px;margin:12px 0;color:var(--warn)}.reference{margin:14px 0;padding:12px 14px;background:var(--panel);border:1px solid var(--line);border-radius:6px}.reference h3{margin:0 0 8px;font-size:14px}.reftable{border-collapse:collapse;width:100%;font-size:13px}.reftable th{text-align:left;padding:4px 12px 4px 0;white-space:nowrap;font-weight:600;vertical-align:top;width:1%}.reftable td{padding:4px 0;vertical-align:top}.refnote{margin:8px 0 0;font-size:12px;color:var(--muted)}.measurements{margin-top:16px;padding:16px;background:var(--panel);border:1px solid var(--line);border-radius:7px;overflow:auto}.measurements h3{margin:0 0 6px}.measurements table,.result-summary table{border-collapse:collapse;width:100%;min-width:720px;font-size:13px}.measurements th,.measurements td,.result-summary th,.result-summary td{padding:8px;border-top:1px solid var(--line);text-align:left;white-space:nowrap}.measurements th,.result-summary th{font-weight:600;color:var(--muted)}.result-summary-wrap{margin-top:16px;padding:16px;background:var(--bg);border:1px solid var(--line);border-left:4px solid var(--ok)}.result-summary-wrap>h3{margin:0 0 14px}.result-summary{margin-top:14px;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px}.result-summary h3{margin:0 0 5px}.summary-scroll{overflow:auto}.summary-figure{overflow:auto;padding:6px 0}.summary-figure svg{max-width:100%;height:auto;display:block}details.log{margin-top:14px}.ok-mark{color:var(--ok);font-weight:600}.short-mark{color:var(--warn);font-weight:600}details.log pre{white-space:pre-wrap;max-height:220px;overflow:auto;background:var(--bg);padding:12px}
@media(max-width:800px){.layout{grid-template-columns:1fr}.steps{grid-template-columns:1fr}.top{display:block}}
.mounting{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-top:14px;padding:12px 14px;background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:6px}.mounting button.on{background:var(--accent);border-color:var(--accent);color:white}
</style></head><body><div class="shell">
<div class="top"><div><h1>__T_app_heading__</h1></div><div class="topright"><div id="workspace" class="muted"></div><div class="actions"><button class="action" onclick="location.href='/api/export'">__T_app_export__</button></div></div></div>
<div id="mounting" class="mounting"><div><strong>__T_mounting_title__</strong><div id="mountingNow" class="muted">__T_ui_loading__</div></div><div class="actions" id="mountingActions"></div></div>
<div id="startup" class="startup"><strong id="startupTitle">__T_startup_running__</strong><div id="startupChecks" class="muted">__T_startup_pending__</div><div class="actions" style="margin-top:10px"><button class="action primary" onclick="runStartupChecks(true)">__T_ui_startupRerun__</button><button class="action" onclick="identifyCameras()">__T_ui_identifyCameras__</button></div></div><div class="layout" id="layout"><aside class="panel sidebar" id="stages"></aside><main class="panel content"><div id="main">__T_ui_loading__</div></main></div></div>
<script src="/dashboard.js" defer></script></body></html>"""

SCRIPT_PATH = Path(__file__).resolve().parent / "dashboard.js"


def script_source() -> str:
    """The dashboard's JavaScript, before phrases are substituted."""
    return SCRIPT_PATH.read_text(encoding="utf-8")


def render_script() -> str:
    """The dashboard's JavaScript, with its phrase table substituted in.

    Lives in dashboard.js so it can be read and edited as JavaScript. The
    phrase table is substituted here rather than fetched by the script, so
    the first render already has its text.
    """
    return script_source().replace(
        "__STRINGS__",
        json.dumps(catalog(), ensure_ascii=False).replace("</", "<\\/"))


def render_page() -> str:
    """The dashboard page.

    The page carries its own static text; the script it loads carries the
    phrase table for everything rendered later.
    """
    page = PAGE
    # Static markup outside the script needs the phrases substituted directly.
    for key, value in catalog().items():
        placeholder = "__T_" + key.replace(".", "_") + "__"
        if placeholder in page:
            page = page.replace(placeholder, escape(value))
    return page


class CalibrationApp:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.engine = WorkflowEngine(workspace)
        self.runtime = LegacyRuntime(workspace)
        self.cameras = CameraIdentification()
        self.startup_checks: dict | None = None
        self.startup_results: list | None = None
        self.operation_lock = threading.RLock()

    def sync_runtime_phase(self) -> None:
        runtime_status = self.runtime.status()
        stage_key = runtime_status.progress.get("stage")
        if self.engine.state.get("active_stage") != stage_key:
            return
        if runtime_status.phase == "review":
            self.engine.set_phase(stage_key, "review")
        elif runtime_status.phase == "failed":
            self.engine.set_phase(stage_key, "failed")
            self.runtime.stop()

    def payload(self) -> dict:
        self.sync_runtime_phase()
        stages = self.engine.overview()
        guide_by_key = {stage.key: stage for stage in STAGES}
        for item in stages:
            item["purpose"] = guide_by_key[item["key"]].purpose
            item["notice"] = guide_by_key[item["key"]].notice
            if item.get("reason_key"):
                item["reason"] = text(item["reason_key"])
        recommended = self.engine.recommended()
        recommended_payload = (None if recommended is None
                               else dict(recommended.__dict__))
        return {"workspace": str(self.workspace.root), "stages": stages,
                "results": sorted(path.name for path in self.workspace.results.glob("*")
                                  if path.is_file()),
                "mounting": self.engine.mounting,
                "mountingLabel": mounting.label(self.engine.mounting),
                "recommended": recommended_payload}

    def complete_stage(self, key: str) -> None:
        with self.operation_lock:
            if self.engine.state.get("active_stage") != key:
                raise LocalizedRuntimeError("err.notActive")
            current = self.engine.state.get("runs", {}).get(key, {})
            handle = self.runtime.handle
            if handle is None or handle.run_id != current.get("run_id"):
                raise LocalizedRuntimeError("err.runExpired")
            runtime_status = self.runtime.status()
            if runtime_status.phase != "review":
                raise LocalizedRuntimeError("err.notReviewable")
            self.sync_runtime_phase()
            outputs, artifacts = self.runtime.collect_outputs()
            self.engine.complete(key, outputs, artifacts)
            self.runtime.stop()

    def handler(self):
        app = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def handle_one_request(self):
                # A browser dropping a camera stream mid-flight is routine for an
                # endless multipart feed, but the stdlib lets the resulting
                # socket error escape this thread as a traceback. Catch it here,
                # once, so every streaming endpoint stays quiet without each
                # having to guard its own writes.
                try:
                    super().handle_one_request()
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

            def send_body(self, body: bytes, content: str, status: int = 200):
                self.send_response(status)
                self.send_header("Content-Type", content)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def send_json(self, payload: dict, status: int = 200):
                self.send_body(json.dumps(payload, ensure_ascii=False).encode(),
                               "application/json; charset=utf-8", status)

            def proxy(self, base: str | None = None, prefix: str = "/stage",
                      rewrite=None):
                base = base if base is not None else app.runtime.proxy_base()
                if base is None:
                    self.send_json({"error": text("err.stageNotStarted")},
                                   HTTPStatus.CONFLICT)
                    return
                suffix = (self.path[len(prefix):] or "/"
                          if self.path.startswith(prefix) else self.path)
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else None
                headers = {key: value for key, value in self.headers.items()
                           if key.lower() not in {"host", "content-length", "connection"}}
                request = Request(base + suffix, data=body, headers=headers,
                                  method=self.command)
                try:
                    with urlopen(request, timeout=120) as response:
                        content_type = response.headers.get(
                            "Content-Type", "application/octet-stream")
                        streaming = ("multipart/" in content_type.lower()
                                     or response.headers.get("Content-Length") is None)
                        if not streaming:
                            payload = response.read()
                            if rewrite is not None and "text/html" in content_type:
                                payload = rewrite(
                                    payload.decode("utf-8")).encode("utf-8")
                            self.send_body(payload, content_type, response.status)
                        else:
                            self.send_response(response.status)
                            self.send_header("Content-Type", content_type)
                            self.send_header("Cache-Control", "no-store")
                            self.end_headers()
                            try:
                                while True:
                                    chunk = response.read(64 * 1024)
                                    if not chunk:
                                        break
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                # The browser closed the camera stream (switched
                                # tab, reloaded, navigated away). That is normal
                                # for an endless multipart feed, not an error, so
                                # end quietly rather than dumping a traceback.
                                return
                except HTTPError as exc:
                    self.send_body(exc.read(), exc.headers.get(
                        "Content-Type", "text/plain; charset=utf-8"), exc.code)
                except URLError as exc:
                    self.send_json({"error": text("err.stageUnavailable",
                                                  message=exc)},
                                   HTTPStatus.BAD_GATEWAY)

            def do_GET(self):
                path = urlparse(self.path).path
                if path.startswith("/stage"):
                    self.proxy()
                elif path == "/api/workflow":
                    self.send_json(app.payload())
                elif path == "/api/export":
                    bundle = app.workspace.export_bundle()
                    payload = bundle.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Disposition", f'attachment; filename="{bundle.name}"')
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif path == "/api/runtime":
                    runtime_status = app.runtime.status()
                    app.sync_runtime_phase()
                    self.send_json(runtime_status.__dict__)
                elif path == "/api/diagnostics":
                    self.send_json(diagnostics_payload())
                elif len(path.strip("/").split("/")) == 4 and path.strip("/").split("/")[:2] == ["api", "stages"] and path.endswith("/summary"):
                    key = path.strip("/").split("/")[2]
                    if key not in BY_KEY:
                        raise KeyError(key)
                    self.send_json(summarize_stage(app.workspace, key))
                elif path == "/api/startup-checks":
                    # The checks probe hardware and take seconds, so a plain
                    # reload re-words the last run. "?refresh=1" is how the
                    # operator asks for the hardware to be looked at again.
                    query = parse_qs(urlparse(self.path).query)
                    refresh = query.get("refresh", ["0"])[0] not in ("0", "")
                    with app.operation_lock:
                        if refresh or app.startup_results is None:
                            app.startup_results = run_startup_checks()
                        app.startup_checks = render_checks(app.startup_results)
                    self.send_json(app.startup_checks)
                elif path == "/":
                    self.send_body(render_page().encode(),
                                   "text/html; charset=utf-8")
                elif path == "/dashboard.js":
                    body = render_script().encode()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type",
                                     "application/javascript; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path.startswith("/results/"):
                    # Serve result artefacts by name only. Resolving and then
                    # confirming the parent keeps ".." and symlinks from
                    # reaching outside the results directory.
                    name = Path(path[len("/results/"):]).name
                    target = (app.workspace.results / name).resolve()
                    if (not name or target.parent != app.workspace.results.resolve()
                            or not target.is_file()):
                        self.send_json(
                            {"error": text("err.noPage")},
                            HTTPStatus.NOT_FOUND)
                        return
                    kind = ("text/html; charset=utf-8" if target.suffix == ".html"
                            else "application/octet-stream")
                    self.send_body(target.read_bytes(), kind)
                elif path.startswith("/cameras"):
                    base = app.cameras.proxy_base()
                    if base is None:
                        self.send_json(
                            {"error": text("err.cameraToolStopped")},
                            HTTPStatus.CONFLICT)
                        return
                    self.proxy(base, "/cameras", lambda html: rewrite_page(html))
                elif app.runtime.proxy_base() is not None:
                    self.proxy()
                else:
                    self.send_json({"error": text("err.noPage")},
                                   HTTPStatus.NOT_FOUND)

            def do_POST(self):
                path = urlparse(self.path).path
                if path.startswith("/stage"):
                    self.proxy()
                    return
                if path.startswith("/cameras"):
                    self.proxy(app.cameras.proxy_base(), "/cameras")
                    return
                if path == "/api/cameras/identify":
                    try:
                        with app.operation_lock:
                            app.cameras.start()
                        self.send_json({"running": True})
                    except FileNotFoundError:
                        self.send_json(
                            {"error": text("err.cameraToolMissing")},
                            HTTPStatus.NOT_FOUND)
                    except (RuntimeError, TimeoutError):
                        self.send_json(
                            {"error": text("err.cameraToolFailed")},
                            HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                if path == "/api/cameras/stop":
                    with app.operation_lock:
                        app.cameras.stop()
                    self.send_json({"running": False})
                    return
                if path == "/api/mounting":
                    # Two steps on purpose. The page asks what a switch would
                    # cost, shows that to the operator, and only then commits.
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    target = str(body.get("mounting", ""))
                    try:
                        with app.operation_lock:
                            if body.get("preview"):
                                self.send_json(app.engine.mounting_change(target))
                            else:
                                self.send_json(app.engine.set_mounting(target))
                    except ValueError as error:
                        self.send_json({"error": str(error)},
                                       HTTPStatus.BAD_REQUEST)
                    except LocalizedRuntimeError as error:
                        self.send_json({"error": message_of(error)},
                                       HTTPStatus.CONFLICT)
                    return
                parts = path.strip("/").split("/")
                try:
                    if len(parts) == 4 and parts[:2] == ["api", "stages"]:
                        key, action = parts[2], parts[3]
                        if key not in BY_KEY:
                            raise KeyError(key)
                        if action == "authorize":
                            with app.operation_lock:
                                stage = BY_KEY[key]
                                if stage.hardware and (not app.startup_checks or not app.startup_checks.get("passed")):
                                    raise LocalizedRuntimeError("err.startupNotPassed")
                                length = int(self.headers.get("Content-Length", "0"))
                                body = json.loads(self.rfile.read(length) or b"{}")
                                run = app.engine.authorize(key, bool(body.get("rerun")))
                                if (app.runtime.handle is not None
                                        and app.runtime.handle.run_id != run["run_id"]):
                                    raise LocalizedRuntimeError("err.otherStageRunning")
                                try:
                                    runtime = app.runtime.start(key, run["run_id"])
                                except Exception as exc:
                                    app.runtime.stop()
                                    app.engine.fail_authorization(key, str(exc))
                                    raise LocalizedRuntimeError(
                                        "err.startFailed", message=exc) from exc
                                app.engine.set_phase(key, "running")
                                run["runtime"] = runtime.__dict__
                                self.send_json(run)
                            return
                        active = app.engine.state.get("active_stage")
                        if action in {"input", "commit", "cancel"} and active != key:
                            raise LocalizedRuntimeError("err.notActiveTask")
                        if action in {"input", "commit", "cancel"}:
                            current = app.engine.state.get("runs", {}).get(key, {})
                            runtime_handle = app.runtime.handle
                            if runtime_handle is None or runtime_handle.run_id != current.get("run_id"):
                                raise LocalizedRuntimeError("err.pageExpired")
                        if action == "input":
                            length = int(self.headers.get("Content-Length", "0"))
                            body = json.loads(self.rfile.read(length) or b"{}")
                            self.send_json(app.runtime.send_input(
                                str(body.get("value", "")),
                                str(body.get("interaction_token", "")),
                            ).__dict__)
                            return
                        if action == "commit":
                            app.complete_stage(key)
                            self.send_json({"ok": True})
                            return
                        if action == "cancel":
                            app.runtime.stop()
                            app.engine.set_phase(key, "cancelled")
                            self.send_json({"ok": True})
                            return
                    if app.runtime.proxy_base() is not None:
                        self.proxy()
                        return
                    self.send_json({"error": text("err.noAction")},
                                   HTTPStatus.NOT_FOUND)
                except FileExistsError as exc:
                    self.send_json({"error": message_of(exc)},
                                   HTTPStatus.CONFLICT)
                except (KeyError, ValueError, RuntimeError, TimeoutError,
                        FileNotFoundError) as exc:
                    self.send_json({"error": message_of(exc)},
                                   HTTPStatus.BAD_REQUEST)
        return Handler


def main(argv: list[str] | None = None) -> int:
    # The terminal stays English whatever the browser language is set to: the
    # operator who starts the server is not always the one at the screen, and a
    # shell is a poor place to discover you cannot read the startup line.
    parser = argparse.ArgumentParser(
        description="XLeRobot guided calibration tool")
    parser.add_argument("--workspace", type=Path,
                        default=Path.home() / ".xlerobot" / "calibration")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8422)
    args = parser.parse_args(argv)
    workspace = Workspace.open(args.workspace)
    app = CalibrationApp(workspace)
    server = ThreadingHTTPServer((args.host, args.port), app.handler())
    print(f"XLeRobot calibration tool: http://{args.host}:{args.port}")
    print(f"Workspace: {workspace.root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.runtime.stop()
        # The camera tool is a separate child process holding /dev/videoN, which
        # is exclusive. Leaving it running strands the cameras: the next run
        # cannot open them and every panel shows blank. Stop it on the way out.
        app.cameras.stop()
        active = app.engine.state.get("active_stage")
        if active:
            try:
                app.engine.set_phase(active, "cancelled")
            except RuntimeError:
                pass
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
