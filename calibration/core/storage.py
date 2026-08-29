"""On-disk format for calibration captures and results.

One schema for every stage, so a later stage can always tell what it is reading
and refuse mismatched data instead of silently producing wrong numbers. Each
capture session is a directory:

    data/stage1_intrinsics_head/
        session.json          metadata: stage, camera, board, resolution, git rev
        frames/000000.png     raw images, lossless
        observations.jsonl    one record per frame: timestamp, servo raw counts,
                              detected corners, ids

Design notes
------------
  - Images are PNG, not JPEG. JPEG artefacts move detected corners by a fraction
    of a pixel, which is the same order as the accuracy we are chasing.
  - Servo readings are stored as RAW encoder counts, never as calibrated angles.
    Calibrated values depend on firmware offsets that this procedure is trying to
    determine, so storing them would bake in a circular dependency.
  - Every record carries a timestamp so a stage can detect that the robot was
    still moving when the frame was taken.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SCHEMA_VERSION = 1


def git_revision() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class SessionMeta:
    """Everything needed to interpret a capture directory."""

    stage: str
    purpose: str
    camera_role: str | None = None
    board_name: str | None = None
    width: int | None = None
    height: int | None = None
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    git_revision: str | None = field(default_factory=git_revision)
    notes: dict = field(default_factory=dict)
    complete: bool = False


class CaptureSession:
    """Writes a capture directory; also reads one back for later stages."""

    def __init__(self, path: Path, meta: SessionMeta | None = None):
        self.path = Path(path)
        self.frames_dir = self.path / "frames"
        self.meta_file = self.path / "session.json"
        self.obs_file = self.path / "observations.jsonl"

        if meta is not None:
            self.meta = meta
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self._write_meta()
        else:
            if not self.meta_file.is_file():
                raise FileNotFoundError(f"no session.json in {self.path}")
            raw = json.loads(self.meta_file.read_text())
            version = raw.get("schema_version")
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"{self.path}: schema version {version}, expected "
                    f"{SCHEMA_VERSION}. Recapture or migrate.")
            self.meta = SessionMeta(**raw)

    def _write_meta(self) -> None:
        self.meta_file.write_text(json.dumps(asdict(self.meta), indent=2) + "\n")

    def add(self, image: np.ndarray, servos: dict[str, int] | None = None,
            detection: dict | None = None, extra: dict | None = None) -> int:
        """Append one observation. Returns its index."""
        index = self.count()
        name = f"{index:06d}.png"
        if not cv2.imwrite(str(self.frames_dir / name), image):
            raise RuntimeError(f"failed to write frame {name}")

        record = {
            "index": index,
            "frame": name,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "servos_raw": servos or {},
        }
        if detection is not None:
            record["corners"] = np.asarray(detection["corners"]).round(4).tolist()
            record["ids"] = np.asarray(detection["ids"]).astype(int).tolist()
            record["n_corners"] = int(detection["n"])
        if extra:
            record.update(extra)

        with self.obs_file.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return index

    def count(self) -> int:
        if not self.obs_file.is_file():
            return 0
        with self.obs_file.open() as fh:
            return sum(1 for line in fh if line.strip())

    def drop_last(self) -> bool:
        """Remove the most recent observation and its frame.

        Lets an operator undo a capture they can see was bad. Indices stay
        contiguous because the next add() reuses the freed index.
        """
        if not self.obs_file.is_file():
            return False
        lines = [ln for ln in self.obs_file.read_text().splitlines() if ln.strip()]
        if not lines:
            return False

        last = json.loads(lines[-1])
        frame = self.frames_dir / last.get("frame", "")
        if frame.is_file():
            frame.unlink()
        body = "\n".join(lines[:-1])
        self.obs_file.write_text(body + "\n" if body else "")
        return True

    def observations(self) -> list[dict]:
        if not self.obs_file.is_file():
            return []
        out = []
        for line in self.obs_file.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "corners" in rec:
                rec["corners"] = np.asarray(rec["corners"], dtype=float)
                rec["ids"] = np.asarray(rec["ids"], dtype=int)
            out.append(rec)
        return out

    def load_frame(self, record: dict) -> np.ndarray:
        img = cv2.imread(str(self.frames_dir / record["frame"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read {record['frame']}")
        return img

    def finish(self, **notes) -> None:
        self.meta.complete = True
        self.meta.notes.update(notes)
        self._write_meta()

    def describe(self) -> str:
        n = self.count()
        status = "complete" if self.meta.complete else "IN PROGRESS"
        lines = [f"  {self.path.name}: {n} observations, {status}",
                 f"    stage {self.meta.stage}, {self.meta.purpose}"]
        if self.meta.camera_role:
            lines.append(f"    camera {self.meta.camera_role} "
                         f"@ {self.meta.width}x{self.meta.height}")
        return "\n".join(lines)


def session_path(stage: str, tag: str | None = None) -> Path:
    name = f"{stage}_{tag}" if tag else stage
    return DATA_DIR / name


def archive_session(path: Path) -> Path | None:
    """Move an existing capture aside so a new run starts empty.

    Sessions live at a fixed path per stage and role, so without this a second
    run appends to the first. For intrinsics that silently mixes two sets of
    views, which is wrong the moment anything changed between them: refocusing,
    a different resolution, or the camera renumbering onto another device. The
    old capture is kept rather than deleted; it is often still the better data.
    """
    path = Path(path)
    if not (path / "session.json").is_file():
        return None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = path.parent / f"{path.name}.{stamp}"
    n = 1
    while dest.exists():
        n += 1
        dest = path.parent / f"{path.name}.{stamp}-{n}"
    path.rename(dest)
    return dest


def list_sessions() -> list[Path]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(p for p in DATA_DIR.iterdir() if (p / "session.json").is_file())


# ---- results -------------------------------------------------------------

def save_result(name: str, payload: dict) -> Path:
    """Write a stage result, converting numpy types to plain JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    body = dict(payload)
    body.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    body.setdefault("git_revision", git_revision())
    path.write_text(json.dumps(body, indent=2, default=json_default) + "\n")
    return path


def load_result(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def result_fingerprint(payload: dict) -> str:
    """Stable identity for result contents, ignoring write metadata."""
    body = {k: v for k, v in payload.items()
            if k not in ("saved_at", "git_revision")}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def json_default(obj):
    """Convert the numpy types that reach JSON from OpenCV and the solvers.

    Anything serving results over HTTP needs the same conversion as writing them
    to disk, so this is shared rather than duplicated: a `dumps` that lacks it
    raises TypeError deep inside a request handler, where it reads as a hang.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__}")
