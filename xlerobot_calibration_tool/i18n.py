"""Every phrase the operator reads, keyed by a stable identifier.

Nothing else in the package should hold a user-facing sentence: modules emit
keys, and the text is looked up when it is shown. Keys are grouped by a prefix
naming the thing they describe -- `stage.`, `prompt.`, `joint.`, `summary.` --
so a missing key is obvious in a diff.

The tool speaks English only. Text lived here in two languages until the
operator settled on one; the indirection stayed because it is what keeps
sentences out of the logic, which is worth having on its own.
"""
from __future__ import annotations

from typing import Any

# The joint names the operator sees, keyed by the identifier the firmware uses.
JOINTS: dict[str, str] = {
    "left_arm_shoulder_pan": "Left shoulder pan",
    "left_arm_shoulder_lift": "Left shoulder lift",
    "left_arm_elbow_flex": "Left elbow flex",
    "left_arm_wrist_flex": "Left wrist flex",
    "left_arm_wrist_roll": "Left wrist roll",
    "left_arm_gripper": "Left gripper",
    "right_arm_shoulder_pan": "Right shoulder pan",
    "right_arm_shoulder_lift": "Right shoulder lift",
    "right_arm_elbow_flex": "Right elbow flex",
    "right_arm_wrist_flex": "Right wrist flex",
    "right_arm_wrist_roll": "Right wrist roll",
    "right_arm_gripper": "Right gripper",
    "head_motor_1": "Head pan",
    "head_motor_2": "Head tilt",
}

TEXT: dict[str, str] = {
    # -- shell -----------------------------------------------------------
    "app.title": "XLeRobot Calibration",
    "app.heading": "Guided calibration",
    "app.export": "Download results",

    # -- how the robot is standing ---------------------------------------
    "mounting.title": "Robot mounting",
    "mounting.normal": "Normal",
    "mounting.flipped": "Back-to-front",
    "mounting.now": "{label}. Every prompt names the arm you can point at.",
    "mounting.confirmTitle": "Switch mounting and discard finished stages?",
    "mounting.confirmBody":
        "Every result is measured in a frame that depends on which way the "
        "chassis faces, so nothing measured under {current} can be reused under "
        "{target}. These finished stages will be archived and must be run again:",
    "mounting.confirmNone":
        "Nothing has been measured yet, so switching costs nothing.",
    "mounting.confirmGo": "Switch and start again",
    "mounting.cancel": "Cancel",
    "mounting.switched": "Mounting set to {label}.",

    # -- startup checks --------------------------------------------------
    "startup.running": "Running robot startup checks...",
    "startup.pending": "Preflight and camera identification not finished yet.",
    "startup.pass": "Passed",
    "startup.recheck": "Re-run checks",
    "startup.failed": "Startup checks failed",

    # -- stage list and status -------------------------------------------
    "status.blocked": "Blocked by an earlier stage",
    "status.ready": "Ready",
    "status.completed": "Completed",
    "status.completedLargeError": "Completed with a large error",
    "status.stale": "Needs redoing",
    "status.authorized": "Authorized",
    "status.running": "Running",
    "status.review": "Saving",
    "status.failed": "Failed",
    "status.cancelled": "Cancelled",

    "stage.why": "Why",
    "stage.what": "What to do",
    "stage.when": "When it is done",
    "stage.outputs": "This stage produces",
    "stage.defaultPurpose": "Establish a trustworthy result for the next stage.",
    "stage.start": "Start stage {number}",
    "stage.redo": "Redo stage {number}",
    "stage.cancel": "Cancel this run",
    "stage.viewResult": "View saved result",
    "stage.number": "Stage {number}",
    "stage.resultTitle": "Stage {number} result - {title}",
    "stage.autoSave": "The result is saved automatically and unlocks the next stage",
    "stage.missing": "Missing earlier results: {items}",

    # -- interaction -----------------------------------------------------
    "interaction.heading": "Your turn",
    "interaction.waiting": "Waiting for the robot...",
    "interaction.stale": "This step has moved on; follow the latest prompt.",
    "interaction.log": "Run log",

    # -- live ranges table -----------------------------------------------
    "ranges.heading": "Joint travel, live",
    "ranges.blurb": "Current, minimum and maximum are cumulative angles from the starting pose, tracked continuously so they stay correct across the 0/4095 seam.",
    "ranges.joint": "Joint",
    "ranges.raw": "Raw counts",
    "ranges.current": "Current",
    "ranges.min": "Minimum",
    "ranges.max": "Maximum",
    "ranges.covered": "Covered",
    "ranges.allowed": "Model allows",
    "ranges.status": "Status",
    "ranges.enough": "Enough",
    "ranges.short": "Not yet",
    "ranges.wrapped": "Crossed the 0/4095 seam",
    "ranges.overTurn": "More than a full turn - reduce the sweep",

    # -- measurements ----------------------------------------------------
    "measure.heading": "Measurements",
    "measure.start": "Start reading",
    "measure.end": "End reading",
    "measure.travel": "Travel",
    "measure.calculation": "How it is derived",

    # -- result summary --------------------------------------------------
    "summary.empty": "This stage has no result to show.",
    "summary.missing": "Result file missing",
    "summary.view3dTitle": "3D view (rotatable)",
    "summary.view3dFile": "File",
    "summary.view3dNote": "About",
    "summary.view3dHint": "Open in any browser to drag-rotate and zoom; a single file that needs no network.",
    "summary.view3dFailed": "Could not be generated: {error}",

    # -- errors ----------------------------------------------------------
    "error.generic": "Something went wrong: {message}",
    "error.offline": "Lost contact with the backend, retrying...",

    # -- dashboard ---------------------------------------------------------
    "ui.loading": "Loading the workflow...",
    "ui.startupHeading": "Robot startup checks",
    "ui.startupRerun": "Run the startup checks again",
    "ui.startupChecking": "Checking the bus, the servos and all three cameras, one moment...",
    "ui.checkFailed": "Failed",
    "ui.needFirst": "Finish these first",
    "ui.startupFirst": "Pass the startup checks first",
    "ui.notYet": "Not ready yet",
    "ui.archiveRerun": "Archive and run again",
    "ui.openTask": "Open the current task",
    "ui.endRun": "End this run",
    "ui.retryCommit": "Retry finishing this stage",
    "ui.viewTask": "View the current task",
    "ui.abandonRun": "Abandon this run",
    "ui.autoSaving": "Saving the result...",
    "ui.suggested": "Suggested next step",
    "ui.suggestedBody": "Finish stage {number}: {title}",
    "ui.rerunTitle": "Running again archives the current result first",
    "ui.rerunBody": "Nothing is deleted. Once confirmed, this run will not ask about overwriting again.",
    "ui.rerunConfirm": "Archive and run again",
    "ui.cancel": "Cancel",
    "ui.startupPending": "The startup checks at the top have not finished; please wait for them.",
    "ui.startupBlocked": "The startup checks did not pass. Fix the problem, or run the checks again.",
    "ui.authorized": "This run is authorized and will not ask about overwriting again.",
    "ui.resultSaved": "The result is saved and can be exported.",
    "ui.summaryFailed": "Could not read the result summary: {message}",
    "ui.loadingResult": "Loading the calibration result…",
    "ui.retry": "Retry",
    "ui.stageDone": "This stage is finished",
    "ui.stageDoneBody": "Saving the result and preparing the next stage.",
    "ui.stageTask": "Current stage task",
    "ui.showStages": "Show the stage list",
    "ui.identifyCameras": "Identify cameras",
    "ui.cameraToolTitle": "Camera identification",
    "ui.cameraToolNote": "Assign a role to each camera by looking at the picture, save, then run the checks again.",
    "ui.cameraToolStarting": "Opening the cameras...",
    "ui.cameraToolDone": "Done, run checks again",
    "ui.stageFrame": "Stage task",
    "ui.computing": "Computing",
    "ui.computingNote": "When it succeeds the result is saved and the next stage begins.",
    "ui.backToFlow": "Back to the workflow",
    "ui.preparing": "Preparing",
    "ui.shortcutHint": " (Enter / Space)",
    "ui.viewLog": "Diagnostic log",
    "ui.noLog": "No log yet",
    "ui.reference": "Reference",
    "ui.roughZeroTitle": "Rough zero readings",
    "ui.roughZeroNote": "These are the raw encoder counts just read from the servos; accepting them sets this arm's rough zero. Check that nothing moved while they were read.",
    "ui.name": "Name",
    "ui.liveTitle": "Live measurements",
    "ui.waitingRead": "Waiting",
    "ui.waitingMove": "Waiting for the move",
    "ui.waitingTravel": "Waiting",
    "ui.waitingSense": "Waiting",
    "ui.diffCounts": "Difference",
    "ui.diffDeg": "In degrees",
    "ui.noteJoin": "; ",
    "ui.commitFailed": "Stage {number} could not be saved: {message}. The result is still held in this run, so you can retry.",
    "ui.currentStage": "current",
    "ui.loadFailed": "Could not load the workflow: {message}",

    # -- stage reasons -----------------------------------------------------
    "reason.authorized": "This run is confirmed; waiting for the stage to start.",
    "reason.running": "The stage is running; reopen the task to carry on.",
    "reason.review": "The stage is finished and the result is being saved.",
    "reason.completed": "The result is complete; view it, or archive it and run again.",
    "reason.completedLargeError": "The stage is complete and its result is saved, but the measured error is too large to deploy.",
    "reason.blocked": "Finish the earlier stages first.",
    "reason.stale": "An earlier result changed; archive this one and run it again.",
    "reason.ready": "Everything it needs is ready.",

    # -- guided prompts: shared -------------------------------------------
    "prompt.thisJoint": "this joint",
    "prompt.leftArm": "the left arm",
    "prompt.rightArm": "the right arm",
    # Sentence-initial form: English needs the capital, Chinese does not care.
    "prompt.leftArmCap": "Left arm",
    "prompt.rightArmCap": "Right arm",
    "prompt.starting.title": "Starting this stage",
    "prompt.starting.instruction": "Reading the first step, one moment.",
    "prompt.startingSenses.title": "Starting the joint direction measurement",
    "prompt.startingSenses.instruction": "Connecting to the stage and reading the first joint step, one moment.",
    "prompt.startingRanges.title": "Starting the rough zero and travel measurement",
    "prompt.startingRanges.instruction": "Connecting to the stage and reading the first arm step, one moment.",

    # -- guided prompts: senses (joint directions) -------------------------
    "prompt.senses.start.title": "{joint}: set the start point",
    "prompt.senses.start.instruction": "Move the joint to the end opposite the positive direction, hold it steady, then read the start point.",
    "prompt.senses.start.yes": "In place - read the encoder",
    "prompt.senses.skip": "Skip this joint for now",
    "prompt.senses.end.title": "{joint}: move to the end point",
    "prompt.senses.end.instruction": "Move it by hand along the positive direction to a comfortable limit, hold it steady, then read the end point.",
    "prompt.senses.end.yes": "Moved - read the encoder",
    "prompt.senses.end.caution": "Keep the servos free while moving, and do not force the mechanical limit.",
    "prompt.readFailed.title": "Reading failed",
    "prompt.senses.readFailed.instruction": "Check the servo connection and the joint, then decide whether to read again.",
    "prompt.readFailed.retry": "Read again",
    "prompt.senses.readFailed.no": "Skip this joint",
    "prompt.busFault.title": "The servo bus check found a problem",
    "prompt.senses.busFault.instruction": "The bus reported a fault. Continue only if you are sure it is unrelated to this direction measurement.",
    "prompt.busFault.yes": "Continue anyway",
    "prompt.busFault.no": "Stop this stage",
    "prompt.senses.busFault.caution": "Better to stop and fix the bus first, so a wrong direction is not recorded.",
    "prompt.senses.again.title": "Some joints are still unfinished",
    "prompt.senses.again.instruction": "You can go back over every joint that was skipped or failed.",
    "prompt.senses.again.yes": "Redo the unfinished joints",
    "prompt.senses.again.no": "Finish and fail this stage",

    # -- guided prompt: replacing results a stage has already saved ---------
    # Shared by every interactive stage, because common.confirm_overwrite is.
    "prompt.overwrite.title": "Replace the saved {results}?",
    "prompt.overwrite.instruction": "This stage has already saved a result. Running it again replaces that result with what you measure now.",
    "prompt.overwrite.yes": "Measure again and replace",
    "prompt.overwrite.no": "Keep the saved result and stop",
    "prompt.overwrite.caution": "The stages after this one were solved from the saved result, so they will need redoing once it is replaced.",

    # -- guided prompts: arm_ranges (rough zeros and travel) ---------------
    "prompt.ranges.poseLeft.title": "Confirm the left arm rough zero",
    "prompt.ranges.poseRight.title": "Confirm the right arm rough zero",
    "prompt.ranges.pose.instruction": "Pose the arm to match the zero pose below and hold it steady. The servos are free, so the arm only stays put when balanced.",
    "prompt.ranges.poseLeft.yes": "Left arm is posed",
    "prompt.ranges.poseRight.yes": "Right arm is posed",
    "prompt.ranges.pose.no": "Not yet",
    "prompt.ranges.skip.title": "Skip {arm}?",
    "prompt.ranges.skip.instruction": "Skipping an arm means this stage cannot pass; normally you should go back and keep posing.",
    "prompt.ranges.skip.yes": "Skip {arm}",
    "prompt.ranges.skip.no": "Go back and pose it",
    "prompt.ranges.skip.caution": "Only skip if the hardware has failed and you are ending this run.",
    "prompt.ranges.accept.title": "Confirm the rough zero for {arm}",
    "prompt.ranges.accept.instruction": "Check that nothing moved while reading, then accept this rough zero.",
    "prompt.ranges.accept.yes": "Accept this rough zero",
    "prompt.ranges.accept.no": "Pose again and re-read",
    "prompt.ranges.sweep.title": "{arm}: sweep every joint's travel",
    "prompt.ranges.sweep.instruction": "In any order, take each joint to both limits and back near the start; you can move several at once. The table below shows each joint's current, minimum and maximum live - finish once they all pass.",
    "prompt.ranges.sweep.yes": "All joints swept",
    "prompt.ranges.sweep.no": "End this arm's sweep",
    "prompt.ranges.sweep.caution": "Move slowly and stop at the mechanical limit; do not force it or pull the cables.",
    "prompt.ranges.keep.title": "{arm}: some joints are still short",
    "prompt.ranges.keep.instruction": "Joints marked short have not covered half the range the model allows; it is worth sweeping those further.",
    "prompt.ranges.keep.yes": "Keep sweeping",
    "prompt.ranges.keep.no": "Keep what we have and finish",
    "prompt.ranges.readFailed.instruction": "Check the servo connection, then decide whether to read again.",
    "prompt.ranges.readFailed.no": "Stop this arm",
    "prompt.ranges.busFault.instruction": "The bus reported a fault. Continuing may give an incomplete zero or range.",
    "prompt.ranges.busFault.caution": "Better to stop and fix the bus first.",

    # -- reference blocks --------------------------------------------------
    "reference.zeroPose.title": "The model's zero pose",
    "reference.zeroPose.note": "Open the model and copy the pose you see: calibration/model/xlerobot_calib.xml, viewed with `python -m mujoco.viewer --mjcf=<that file>` (press space to pause, or gravity will pull the arms out of the pose). Both arms take the same pose. Get each joint within about 10 degrees, and note that a link pointing the opposite way is not the same zero.",
    "reference.direction.title": "This joint's positive direction",
    "reference.direction.note": "The positive direction comes from the model file calibration/model/xlerobot_calib.xml and is verified joint by joint against the model when the stage starts, independently of how the robot is wired.",
    "reference.direction.fallback": "the direction the stage states",

    # -- direction table ---------------------------------------------------
    "direction.shoulder_pan": "Seen from above, the whole arm swings clockwise",
    "direction.shoulder_lift": "The arm lifts up",
    "direction.elbow_flex": "The forearm swings down",
    "direction.wrist_flex": "The gripper swings down",
    "direction.wrist_roll": "Looking along the arm toward the fingertips, the gripper turns clockwise",
    "direction.gripper": "The gripper opens",
    "direction.head_motor_1": "Seen from above, the head turns counter-clockwise",
    "direction.head_motor_2": "The head looks down",

    # -- result summaries: shared -----------------------------------------
    "sum.none": "—",
    "sum.unrecorded": "Not recorded",
    "sum.camera": "Camera",
    "sum.joint": "Joint",
    "sum.arm": "Arm",
    "sum.status": "Status",
    "sum.item": "Item",
    "sum.valueCol": "Value",
    "sum.samples": "Samples",
    "sum.leftArm": "Left arm",
    "sum.rightArm": "Right arm",

    # -- summary: prepare (board) ------------------------------------------
    "sum.board.title": "Calibration board",
    "sum.board.board": "Board",
    "sum.board.size": "Layout",
    "sum.board.square": "Square mm",
    "sum.board.marker": "Marker mm",
    "sum.board.dictionary": "Dictionary",

    # -- summary: intrinsics -----------------------------------------------
    "sum.intr.title": "Camera intrinsics",
    "sum.intr.resolution": "Resolution",
    "sum.intr.views": "Views",
    "sum.intr.coverage": "Frame coverage",
    "sum.intr.fitRms": "Fit RMS",
    "sum.intr.holdoutRms": "Holdout RMS",

    # -- summary: senses ---------------------------------------------------
    "sum.senses.title": "Direction of all 14 joints",
    "sum.senses.sign": "Direction",
    "sum.senses.start": "Start counts",
    "sum.senses.end": "End counts",
    "sum.senses.travel": "Travel counts",

    # -- summary: head -----------------------------------------------------
    "sum.head.title": "Head fit",
    "sum.head.panSweep": "Pan sweep °",
    "sum.head.tiltSweep": "Tilt sweep °",
    "sum.head.fitRms": "Fit RMS mm",
    "sum.head.holdoutRms": "Holdout RMS mm",
    "sum.head.holdoutDeg": "Holdout RMS °",

    # -- summary: arm_ranges -----------------------------------------------
    "sum.ranges.title": "Arm rough zeros and travel",
    "sum.ranges.zero": "Rough zero counts",
    "sum.ranges.span": "Measured travel",
    "sum.ranges.recorded": "Joints recorded",

    # -- summary: arms -----------------------------------------------------
    "sum.arms.title": "Arm calibration",
    "sum.arms.fitViews": "Fit / holdout views",
    "sum.arms.failed": "Solve failed",
    "sum.arms.noHoldout": "No holdout data",
    "sum.arms.good": "Good",
    "sum.arms.acceptable": "Acceptable",
    "sum.arms.tooLarge": "Over the limit (> {limit} mm)",
    "sum.arms.zeroTitle": "Zero corrections against the rough zeros",
    "sum.arms.roughZero": "Rough zero counts",
    "sum.arms.refined": "Refined counts",
    "sum.arms.change": "Change counts",
    "sum.arms.angle": "In degrees",
    "sum.arms.notSolved": "Not solved",

    # -- summary: normalize ------------------------------------------------
    "sum.norm.title": "Zero corrections ({changed} changed)",
    "sum.norm.before": "Before counts",
    "sum.norm.after": "After counts",
    "sum.norm.unchanged": "Unchanged",
    "sum.norm.frameTitle": "Body frame redefinition",
    "sum.norm.yaw": "Body yaw (from arm-root symmetry)",
    "sum.norm.tilt": "Head tilt zero (optical axis along -X)",
    "sum.norm.root": "{arm} root mm",
    "sum.norm.heading": "{arm} forearm heading error (before correction)",
    "sum.norm.sharedFrame": "Shared body frame",
    "sum.norm.frameMismatch": "Inconsistent ({count} of them)",

    # -- summary: verify ---------------------------------------------------
    "sum.verify.title": "Independent verification, all three cameras",
    "sum.verify.passed": "Passed",
    "sum.verify.failed": "Failed: {gates}",
    "sum.verify.gateJoin": ", ",
    "sum.verify.gatePosition": "position",
    "sum.verify.gateRotation": "rotation",
    "sum.verify.gateSamples": "sample count",
    "sum.verify.posRms": "Position RMS / limit",
    "sum.verify.rotRms": "Rotation RMS / limit",
    "sum.verify.detailTitle": "Error distribution (RMS / P95 / max)",
    "sum.verify.posRmsPlain": "Position RMS",
    "sum.verify.posP95": "Position P95",
    "sum.verify.posMax": "Position max",
    "sum.verify.rotRmsPlain": "Rotation RMS",
    "sum.verify.rotP95": "Rotation P95",
    "sum.verify.rotMax": "Rotation max",
    "sum.verify.biasTitle": "Systematic bias and reprojection",
    "sum.verify.posBias": "Position bias XYZ mm",
    "sum.verify.rotBias": "Rotation bias XYZ °",
    "sum.verify.pixelRms": "Predicted pixel RMS",
    "sum.verify.pnpRms": "PnP reprojection RMS",

    # -- robot overview -------------------------------------------------------
    "ov.camera.head": "Head camera",
    "ov.camera.left_wrist": "Left wrist camera",
    "ov.camera.right_wrist": "Right wrist camera",
    "ov.arm.left_arm": "Left arm",
    "ov.arm.right_arm": "Right arm",
    "ov.frame.body": "Body",
    "ov.frame.gripper": "Gripper",
    "ov.cameras.title": "Camera positions and optical axes",
    "ov.cameras.posFrame": "Position frame",
    "ov.cameras.position": "Position XYZ (mm)",
    "ov.cameras.axisFrame": "Axis frame",
    "ov.cameras.axis": "Optical axis",
    "ov.azimuth": "Azimuth (°)",
    "ov.elevation": "Elevation (°)",
    "ov.arms.title": "Arm root positions and orientation",
    "ov.arms.xAxis": "X axis",
    "ov.notes.title": "Notes",
    "ov.note.frames": "Positions and orientations are in the calibrated body frame, with the head at exactly pan=tilt=0 and the arms at their calibrated zeros.",
    "ov.note.axis": "The optical axis is the camera's +Z (OpenCV convention). Azimuth is measured from +X about +Z; elevation is the angle above the horizontal.",
    "ov.note.wrist": "The wrist cameras are fixed to the grippers, so their body-frame position depends on the arm pose; position is given in the gripper frame, and the optical axis is its body-frame direction at the zero pose.",
    "ov.note.roots": "Arm root positions are the model's nominal positions after the calibrated correction.",
    "ov.note.frameId": "Body frame id: {value}",
    "ov.diagram.top": "Top view (robot front is up)",
    "ov.diagram.side": "Side view (robot front is right)",
    "ov.diagram.aria": "Diagram of the calibrated robot geometry",
    "ov.overview.title": "Robot overview",
    "ov.diagram.title": "Calibrated geometry",
    "ov.missing": "Missing result files: {items}",
    "ov.missingModel": "Model file missing",
    "ov.join": ", ",
    "ov.armCol": "Arm",

    # -- 3D view page ---------------------------------------------------------
    "view3d.title": "XLeRobot calibration - 3D view",
    "view3d.hint": "Drag to rotate, scroll to zoom. All values are millimetres in the calibrated body frame.",
    "view3d.spin": "Auto-rotate",
    "view3d.iso": "Isometric",
    "view3d.front": "Front",
    "view3d.side": "Right side",
    "view3d.top": "Top",
    "view3d.reset": "Reset",
    "view3d.object": "Object",
    "view3d.position": "Position XYZ (mm)",
    "view3d.heading": "Heading / optical axis",
    "view3d.note": "Note",
    "view3d.footer": "Axes: +X is the robot's front, +Y its left, +Z up. Camera optical axes are the +Z direction (OpenCV convention). The wrist cameras are fixed to the grippers and move with the arms, so they are drawn where they sit when the arms are at their calibrated zeros.",
    "view3d.axisX": "+X front",
    "view3d.axisY": "+Y left",
    "view3d.axisZ": "+Z up",
    "view3d.mounting.normal": "Mounted normally. Left and right below are both the robot's and yours; they agree.",
    "view3d.mounting.flipped": "Mounted back-to-front. Left and right below are YOURS, as you face the robot's working side; the saved results name each arm and wrist camera the opposite way. The head is drawn at pan = 180 deg, the posture whose encoder reading is the stored head zero on this mounting.",
    "view3d.directionOnly": " (direction)",
    "view3d.armNote": "Arm root (X axis is its heading)",
    "view3d.wristNote": "On the gripper; shown at the zero pose, moves with the arm",
    "view3d.wristNoteAnchored": "Offset in the gripper frame; drawn from the arm root, direction only",
    "view3d.bodyNote": "Body frame",

    # Refusals raised while driving the workflow. These reach the operator
    # through the dashboard's error banner, so they need translating like any
    # other on-screen phrase.
    "err.notActive": "That stage is not the active task, so it cannot be saved.",
    "err.runExpired": "This run has expired. Start the stage again.",
    "err.notReviewable": "The stage has not produced a complete result that passes its gates, so it cannot be saved.",
    "err.stageNotStarted": "The stage page has not started yet",
    "err.stageUnavailable": "The stage service is unavailable: {message}",
    "err.noPage": "No such page",
    "err.cameraToolStopped": "The camera identification tool is not running",
    "err.cameraToolMissing": "tools/cameras/identify.py is missing.",
    "err.cameraToolFailed": "The camera identification tool failed to start. Check that the cameras are plugged in and not in use by another program.",
    "err.noAction": "No such action",
    "err.startupNotPassed": "The startup checks at the top have not passed. Wait for them to finish, or select “Run the startup checks again”.",
    "err.otherStageRunning": "Another stage is already running. End that task first.",
    "err.startFailed": "The stage failed to start: {message}",
    "err.notActiveTask": "That stage is not the active task. Go back to the current task first.",
    "err.pageExpired": "The run this page belongs to has expired. Reopen the workflow.",
    "err.requiresMissing": "Earlier results are not finished yet: {items}",
    "err.resultExists": "This stage already has a result. View it, or archive it and run again.",
    "err.anotherRunning": "Another stage is running. Go back and end that task first.",
    "err.armsIncomplete": "The arm stage has not calibrated both arms, so it cannot be submitted",
    "err.armsMissing": "The arm stage is missing the left or right arm result, so it cannot be submitted",
    "err.notReviewState": "The stage must finish and reach the review state before it can be submitted",
    "err.runtimeLost": "The active task could not be restored after the dashboard restarted. Please run it again.",

    # Startup diagnostics. The detail (module names, device paths, preflight
    # output) is substituted in; only the wording around it is translated.
    "check.algorithmsFound": "The calibration algorithm directory is available",
    "check.algorithmsMissing": "The calibration algorithm directory was not found",
    "check.algorithmsFix": "Run from the directory this tool was unpacked into, or install it as an editable install.",
    "check.busFound": "The robot bus configuration is available",
    "check.busMissing": "tools/config/buses.py was not found",
    "check.busFix": "Keep the tools directory that ships with this tool, and configure the serial port.",
    "check.moduleFound": "{label} is available",
    "check.moduleMissing": "{label} is missing",
    "check.installLegacy": "pip install -e '.[legacy]'",
    "check.installFeetech": "pip install -e '.[legacy]', which includes the Feetech servo SDK.",
    "check.cameraModuleMissing": "The camera configuration module was not found",
    "check.cameraModuleUnloadable": "The camera configuration module could not load",
    "check.cameraFix": "Keep tools/config/cameras.py, then run tools/cameras/identify.py.",
    "check.cameraReadFailed": "The camera mapping could not be read: {message}",
    "check.cameraIdentifyFix": "Use the \"Identify cameras\" button above to identify the three cameras again.",
    "check.cameraRolesMissing": "Missing roles: {roles}",
    "check.cameraDevicesMissing": "These device nodes do not exist: {devices}",
    "check.cameraRolesAndDevices": "Missing roles: {roles}; these device nodes do not exist: {devices}",
    "check.cameraOk": "All three camera roles are mapped: {mapping}",
    "check.detailJoin": "; ",
    "check.preflightMissing": "tools/checks/preflight.py was not found",
    "check.preflightMissingFix": "Keep tools/checks/preflight.py; it ships with this tool.",
    "check.preflightTimeout": "The preflight check timed out (over 120 seconds)",
    "check.preflightTimeoutFix": "Check the serial port, servo, and camera connections, then run the startup checks again.",
    "check.preflightPassed": "Passed",
    "check.preflightExit": "preflight exited with status {code}",
    "check.preflightDetail": "{detail}",
    "check.preflightFix": "Review the errors above. If they report a missing dependency, install this tool's own: pip install -e '.[legacy]'",
}


def text(key: str, **fields: Any) -> str:
    """One phrase.

    An unknown key returns the key itself: a missing phrase should show up
    plainly in the interface rather than crash a calibration mid-run.
    """
    value = TEXT.get(key)
    if value is None:
        return key
    return value.format(**fields) if fields else value


def joint_label(joint: str, mounting_name: str | None = None) -> str:
    """The operator-facing name of a joint, or its raw identifier.

    The stored names say which arm of the MODEL a joint belongs to. Used
    back-to-front that is the opposite of the arm the operator is looking at,
    so when a mounting is given the label names the side they can SEE.

    Only that side is given. The operator is standing in front of one arm and
    needs to know which; showing the stored name beside it asks them to work out
    which of two opposite answers applies to them, which is the confusion the
    label exists to remove. The stored names stay in the saved files, where the
    reader is looking at the model rather than at the robot.
    """
    label = JOINTS.get(joint, joint)
    if mounting_name is None or label == joint:
        return label
    from . import mounting as mounting_mod
    if not mounting_mod.is_flipped(mounting_name):
        return label
    for arm in ("left_arm", "right_arm"):
        if joint.startswith(f"{arm}_"):
            side = mounting_mod.physical_side(arm, mounting_name)
            return f"{side.capitalize()}{label[label.index(' '):]}"
    return label


def catalog() -> dict[str, str]:
    """Every phrase, for the browser to render with."""
    return dict(TEXT)


class Localized(Exception):
    """A refusal that carries its phrase key rather than a formatted string.

    Raised deep in the workflow and turned into text at the point the message
    is put on the wire. Subclassing the builtin the caller already expects
    keeps existing handlers working.
    """

    builtin: type[Exception] = Exception

    def __init__(self, key: str, **fields: Any) -> None:
        self.key = key
        self.fields = fields
        super().__init__(text(key, **fields))

    def localized(self) -> str:
        return text(self.key, **self.fields)


class LocalizedValueError(Localized, ValueError):
    pass


class LocalizedRuntimeError(Localized, RuntimeError):
    pass


class LocalizedFileExistsError(Localized, FileExistsError):
    pass


def message_of(exc: BaseException) -> str:
    """What to show the operator for a failure."""
    return exc.localized() if isinstance(exc, Localized) else str(exc)

