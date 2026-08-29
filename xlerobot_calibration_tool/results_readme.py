"""Human-readable guide describing every file in an exported result bundle."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

# (filename, stage number or None, description)
FILES: tuple[tuple[str, int | None, str], ...] = (
    ("board.json", 1,
     "Charuco board geometry: square size, dictionary, and the pose of each board in the world frame."),
    ("intrinsics_head.json", 2,
     "Head camera intrinsics: resolution, camera matrix K, distortion coefficients, and fit/holdout reprojection error."),
    ("intrinsics_left_wrist.json", 2,
     "Left wrist camera intrinsics, same fields as above."),
    ("intrinsics_right_wrist.json", 2,
     "Right wrist camera intrinsics, same fields as above."),
    ("senses.json", 3,
     "Rotation sense (+1 / -1) of every joint: which way the joint turns as the encoder count increases."),
    ("head.json", 5,
     "Head mechanism calibration: camera pose relative to the tilt joint, pan/tilt base pose, and lever-arm lengths."),
    ("zeros.json", 4,
     "Rough zero offsets and measured travel ranges (in counts) for the arm joints."),
    ("touch.json", 6,
     "Raw observations from the arm touch calibration: every capture and the zero offsets used."),
    ("zeros_refined.json", 6,
     "Joint zero offsets refined by the stage {number} fusion."),
    ("head_zero.json", 7,
     "Head calibration expressed in the normalized body frame."),
    ("touch_zero.json", 7,
     "Normalized arm-root poses. Note that T_B_A is a correction relative to the nominal arm root in the XML, not an absolute position."),
    ("zeros_zero.json", 7,
     "Final joint zeros after normalization, with the gauge freedom between arm-root yaw and shoulder pan resolved."),
    ("validation.json", 8,
     "Verification result: per-camera residuals, shared drift, the overall pass/fail flag, and a human-readable report."),
    ("robot_yaml.json", 8,
     "Machine-readable roll-up of every calibrated quantity (the JSON form of robot.yaml)."),
    ("robot.yaml", 8,
     "The final calibration result: the main file for downstream programs to read."),
    ("xlerobot_calib_fitted.xml", None,
     "MuJoCo model with the calibration baked in. It references meshes/ by relative path, so open it from calibration/model/."),
    ("robot_overview_3d.html", None,
     "Rotatable 3D overview. Open it directly in a browser to inspect camera and arm-root positions and orientations. The file path shown in the result summary is a link: click it to open."),
    ("manifest.json", None,
     "Export manifest: the export timestamp and a SHA-256 checksum for every file."),
    ("workflow.json", None,
     "Workflow state: the completion status and run records of every stage."),
)

_TEXT = {
    "title": "XLeRobot calibration results",
    "generated": "Generated",
    "intro": "This directory is the output of one complete calibration. "
             "The table below explains what each file is for; the "
             "\"Stage\" column is the stage that produced it.",
    "col_file": "File",
    "col_stage": "Stage",
    "col_desc": "Description",
    "none": "—",
    "start_title": "Where to start",
    "start": [
        "`robot.yaml` — the final calibration result; this is what "
        "downstream programs read.",
        "`validation.json` — check the `passed` field to confirm the "
        "calibration met the acceptance criteria.",
        "`robot_overview_3d.html` — open in a browser to sanity-check the "
        "geometry visually.",
    ],
    "xml_title": "How to open xlerobot_calib_fitted.xml",
    "xml": "The XML references its mesh files by relative path, so opening "
           "it from this directory fails because meshes/ is not here. Copy "
           "it into `calibration/model/` in the repository (which has "
           "meshes/), then open it with MuJoCo:",
    "xml_cmd": "python -m mujoco.viewer --mjcf=xlerobot_calib_fitted.xml",
    "frames_title": "Conventions",
    "frames": [
        "Lengths are metres and angles are radians unless the field name "
        "says otherwise (e.g. `_mm`, `_deg`).",
        "`T_A_B` is the pose that maps points in frame B into frame A.",
        "Joint zeros are stored in encoder counts; one full turn is 4096.",
    ],
}


def render(present: set[str] | None = None) -> str:
    """Build the README text, marking which files are actually present."""
    text = _TEXT
    lines = [
        f"# {text['title']}",
        "",
        f"{text['generated']}: {datetime.now().isoformat(timespec='seconds')}",
        "",
        text["intro"],
        "",
        f"| {text['col_file']} | {text['col_stage']} | {text['col_desc']} |",
        "| --- | --- | --- |",
    ]
    for name, stage, description in FILES:
        if present is not None and name not in present:
            continue
        stage_cell = str(stage) if stage is not None else text["none"]
        # Descriptions may cite their own stage number; fill it from the same
        # column the table shows, so the two can never disagree.
        desc = description.replace("{number}", stage_cell)
        lines.append(f"| `{name}` | {stage_cell} | {desc} |")
    lines += ["", f"## {text['start_title']}", ""]
    lines += [f"- {item}" for item in text["start"]]
    lines += ["", f"## {text['xml_title']}", "", text["xml"], "",
              "```bash", f"cd calibration/model", text["xml_cmd"], "```",
              "", f"## {text['frames_title']}", ""]
    lines += [f"- {item}" for item in text["frames"]]
    lines.append("")
    return "\n".join(lines)


def write(results_dir: Path) -> Path:
    """Write README.md next to the result files."""
    present = {path.name for path in results_dir.glob("*") if path.is_file()}
    target = results_dir / "README.md"
    target.write_text(render(present), encoding="utf-8")
    return target
