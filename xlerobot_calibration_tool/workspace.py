"""Isolated result workspace and atomic metadata operations."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import results_readme

LEGACY_STAGE6_RESULTS = {"wrist_left", "wrist_right"}


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def open(cls, root: Path | str) -> "Workspace":
        instance = cls(Path(root).expanduser().resolve())
        instance.results.mkdir(parents=True, exist_ok=True)
        instance.runs.mkdir(parents=True, exist_ok=True)
        instance.sessions.mkdir(parents=True, exist_ok=True)
        return instance

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    def result_path(self, name: str) -> Path:
        return self.results / f"{name}.json"

    def load_result(self, name: str) -> dict[str, Any] | None:
        path = self.result_path(name)
        return json.loads(path.read_text()) if path.is_file() else None

    def save_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def fingerprint(self, name: str) -> str | None:
        path = self.result_path(name)
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def import_legacy(self, legacy_results: Path | str) -> dict[str, str]:
        source = Path(legacy_results).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"The previous results directory does not exist: {source}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs / f"legacy_import_{stamp}"
        imported: dict[str, str] = {}
        archived: dict[str, str] = {}
        artifacts: dict[str, str] = {}
        run_dir.mkdir(parents=True)
        for path in sorted(source.glob("*.json")):
            target = run_dir / path.name
            shutil.copy2(path, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if path.stem in LEGACY_STAGE6_RESULTS:
                archived[path.stem] = digest
            else:
                imported[path.stem] = digest
        artifact_sources = {
            "robot.yaml": source / "robot.yaml",
            "xlerobot_calib_fitted.xml": source.parent / "model" / "xlerobot_calib_fitted.xml",
        }
        for name, path in artifact_sources.items():
            if not path.is_file():
                continue
            target = run_dir / name
            shutil.copy2(path, target)
            artifacts[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = {
            "kind": "legacy_import",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": str(source),
            "files": imported,
            "artifacts": artifacts,
            "archived_legacy_stage6": archived,
            "source_is_read_only": True,
            "committed": True,
        }
        self.save_json_atomic(run_dir / "manifest.json", manifest)
        for name in imported:
            self.save_json_atomic(self.result_path(name), json.loads((run_dir / f"{name}.json").read_text()))
        for name in artifacts:
            shutil.copy2(run_dir / name, self.results / name)
        return imported

    def export_bundle(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = self.root / f"xlerobot_calibration_{stamp}.zip"
        readme = results_readme.write(self.results)
        manifest = {
            "kind": "calibration_export",
            "readme": readme.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "files": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(self.results.glob("*")) if path.is_file()
            },
        }
        manifest_path = self.root / ".export_manifest.json"
        self.save_json_atomic(manifest_path, manifest)
        try:
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(manifest_path, "manifest.json")
                for path in sorted(self.results.glob("*")):
                    if path.is_file():
                        archive.write(path, f"results/{path.name}")
                if self.state_file.is_file():
                    archive.write(self.state_file, "workflow.json")
        finally:
            manifest_path.unlink(missing_ok=True)
        return output

    @property
    def state_file(self) -> Path:
        return self.root / "workflow.json"

    @property
    def mounting(self) -> str:
        """How the operator declared the robot is standing.

        Read here rather than passed in because the summary layer is reached
        from several directions (the dashboard, the exporter, the 3D page) and
        threading a parameter through all of them invites exactly one caller
        to forget it. Forgetting is what went wrong before: `joint_label` takes
        the mounting optionally, so a missed argument silently produced the
        model's names instead of the operator's.

        Defaults to normal, which is what a workspace with no declaration was
        calibrated as, and which makes every conversion an identity.
        """
        try:
            declared = json.loads(self.state_file.read_text()).get("mounting")
        except (OSError, ValueError):
            return "normal"
        return declared if declared in ("normal", "flipped") else "normal"
