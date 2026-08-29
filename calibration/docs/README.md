# Geometric calibration for XLeRobot

Solve where an XLeRobot's arms, head and cameras actually are, relative to the
kinematic model that a policy or an IK solver believes in — using nothing but
the robot's own cameras and a printed ChArUco board.

No laser tracker. No force/torque sensors. No ROS.

This is written for the XLeRobot specifically: two SO-101 arms, a pan/tilt
head, three cameras and the MuJoCo model in this repository. The maths is
general, but the code is not — it addresses these joints, these camera roles
and this model file by name.

## The problem this solves

`lerobot-calibrate` and its equivalents solve a **servo** problem: they write a
homing offset into each motor so a joint reads a known value at a known
physical position. That is what makes a leader and a follower agree, and it is
what policy transfer needs.

It does not solve a **geometric** problem. Nothing in that procedure knows
where each arm is bolted to the chassis, how far each joint's zero sits from
the *model's* zero, or where any camera is and which way it looks.

So the moment you ask for forward kinematics, inverse kinematics,
end-effector-space teleoperation or sim-to-real transfer, you are relying on
numbers nobody measured, and the gap shows up as an offset you cannot chase
down by hand.

This project measures those quantities directly and writes them back into the
model. A printed ChArUco board of known geometry stands in for the metrology
this hardware does not have: the robot's own cameras observe the board from
many postures, and that closes the kinematic loop for the cost of a sheet of
paper.

## What it produces

Per robot, written back into the MuJoCo model and a deployable YAML:

- `T_W_B` — chassis pose in the world (board) frame
- camera intrinsics `K` and distortion, for all three cameras
- `T_tilt_cam` — head camera mount, plus the head pan/tilt geometry
- `T_B_A` per arm — where each arm root really sits on the chassis
- five joint zero corrections per arm, relative to the model's zero
- `T_wrist_cam` per arm — wrist camera mount on the gripper
- a joint-sense table: which way each of the fourteen servos actually turns

## How to use it

Everything runs from one guided interface:

```bash
python -m xlerobot_calibration_tool     # then open http://127.0.0.1:8422
```

The tool works out what is next, refuses to start a stage whose inputs are
missing, and refuses to advance past one that failed its check. Stages that
need a camera show a live image, the corners being detected and the coverage
still missing, so you can see what you are collecting as you collect it.

### Where the results go

Each calibration lives in a **workspace** — a directory holding the results,
the run history and the captured frames. By default that is
`~/.xlerobot/calibration`; name your own to keep runs apart:

```bash
python -m xlerobot_calibration_tool --workspace new_calibration_5
```

| option | default | |
| --- | --- | --- |
| `--workspace` | `~/.xlerobot/calibration` | where results are written; a relative path is taken from the current directory |
| `--port` | `8422` | change it to run two instances side by side |
| `--host` | `127.0.0.1` | `0.0.0.0` to reach the page from another machine |

The directory does not have to exist — it is created on start, and pointing at
a new one gives you an empty workspace. Pointing at an existing one **resumes**
it rather than starting over: finished stages stay finished, and the tool
carries on from wherever that run stopped. To redo a stage that is already
complete, rerun it from the interface. So give a fresh run a fresh name unless
you mean to continue an old one.

The workspace path is printed on startup — worth a glance before you begin:

```
XLeRobot calibration tool: http://127.0.0.1:8422
Workspace: /path/to/new_calibration_5
```

### Before the stages: startup checks and camera roles

On launch the tool runs its **preflight** — servo bus, cameras, model files,
dependencies — and then asks you to do **camera identification**: it shows a
picture from each camera and you say which one it is (head, left wrist, right
wrist). Nothing else will start until both are green, because every later
stage is addressed to a camera by role rather than by device number.

### Tell it which way the robot is standing

XLeRobot can be assembled with the head and arms facing either way round, so
the interface asks for the **mounting** up front. The choice is exactly about
whether physical left and right agree with the model:

- **Normal** — the arm on your left is the one the XML calls `left_arm`. The
  physical sides and the model's sides are the same.
- **Back-to-front** — the arm on your left is the one the XML calls
  `right_arm`. The two are exactly opposite, on both arms and both wrist
  cameras.

That is the whole difference, and it is why the setting exists: the names in
the model never move, but which side of the room they are on does. The tool
keeps the two straight for you — every prompt names the arm you can actually
point at, while everything saved stays in model terms, so the results mean the
same thing under either mounting.

Set it before you start. Changing it later archives everything already
measured, because each result lives in a frame that depends on which way the
chassis faces.

### The eight stages

| Stage | What it does |
|---|---|
| 1. Setup and calibration board | Measure the board squares, lock focus on all three lenses, check the base and board can be held still |
| 2. Camera intrinsics | Focal length, principal point and distortion, one camera at a time |
| 3. Joint directions | Move each joint by hand the way the interface asks, so real encoder directions map onto the model's axes |
| 4. Arm rough zeros and travel | Pose each arm near its zero, then sweep each joint through its usable range |
| 5. Head, world frame and head camera | Hold the robot and board still, move pan and tilt by hand; solves the world frame and head geometry together |
| 6. Arm calibration | Point each wrist camera at the board in turn; solves arm mounting, joint zeros and wrist camera mount together |
| 7. Body frame and zero conventions | Collects nothing new — derives the body's forward direction from arm symmetry and fixes the zero conventions |
| 8. Independent verification and export | Fresh poses, no parameter fitted, then export |

Stages 1–4 are the ones where you move the robot around freely. From stage 5
onward the base and the board must not move at all.

**The board we use.** Any ChArUco board works — stage 1 accepts any size,
square and marker dimensions, and dictionary. But the one in this repository
is the only one we have tested on, and it is what every stage's accuracy
numbers were measured with, so it is what we recommend:
[`boards/ChArUco_A4_10x14_20mm_15mm_DICT_4X4.pdf`](../boards/ChArUco_A4_10x14_20mm_15mm_DICT_4X4.pdf)
— 10 × 14 squares, 20 mm squares with 15 mm markers, `DICT_4X4_1000`, legacy
layout.

Print it on A4 **at 100% scale**: "fit to page" rescales the squares, and that
becomes a proportional error on every distance the calibration reports. Then
measure a square with calipers and enter what you measured in stage 1, because
printers are not exact even when told not to scale. Mount it flat on something
rigid — a curled board is a shape error the solver will absorb into the joint
zeros.

Bringing your own is fine, and if you are unsure of its parameters the tool
can identify them from a photo. See [`boards/README.md`](../boards/README.md),
which also covers the `legacy` flag — get that wrong and every marker still
detects while not a single ChArUco corner does.

## Accuracy

Stage 8 is held-out validation: the robot is re-posed by hand into ten fresh
postures per camera, the frozen model predicts where each camera should be, and
that is compared against what the camera actually sees. **No parameter is
fitted at this stage.**

Across full runs in both mounting orientations, camera pose error comes out
**within about a centimetre**, with the head camera typically around 3–6 mm and
rotation errors of a few degrees at most. For context, the community baseline
for FK/IK error on this class of hardware is 10–15 mm, usually unvalidated.

Those numbers come from ordinary runs, not showcase ones — a careful operator
collecting well-spread, corner-rich views does better. Most of the remaining
error traces to individual frames captured with few ChArUco corners visible;
stage 6 advises on corner count while you capture, and stage 8 will catch a
thin view afterwards.

### Known limits

- **The floor is servo noise, not method.** Zero error is very nearly
  proportional to servo angle noise, and is exactly zero when noise is zero.
  Past a certain number of well-spread views there is nothing left to win:
  what remains is what the servos themselves cannot report.
- **Currently shaped around one robot.** It assumes the XLeRobot layout: two
  SO-101 arms, a pan/tilt head, three cameras, a MuJoCo model. The maths is
  general; the wiring is not yet.

## Getting a good calibration

Nothing here is subtle, but the difference between a careless run and a
careful one is most of the error budget.

**Get corners, not just frames.** This matters more than anything else on the
list. A view where few ChArUco corners are visible gives a weak pose, and it
does not look weak: the reprojection error stays *low*, because there is
little left to disagree with. Measured on real runs, views with under 12
corners averaged 8.7 mm of pose error against 3.9 mm for views with 20 or
more. Fill the frame with the board and keep it fully inside the image.

**Collect more views than feels necessary.** Each stage tells you when it has
enough, but "enough" is a floor, not a target. Extra views cost seconds and
average away servo noise, which is the error floor you are working against.

**Spread the postures out.** Views that resemble each other constrain the same
directions over and over. Vary distance, angle and where the board sits in the
frame; for the arms, move shoulder, elbow and wrist rather than one joint.
Coverage is what makes a parameter observable at all — a stage can fit its
data beautifully and still be badly determined if every view was taken from
the same place.

**Keep the board and the base still.** From stage 5 onward, neither the board
nor the robot base may move at all — every result from there on is expressed
relative to that board.

**Do not touch the lenses after stage 2.** These are manual-focus modules with
no software readback, and intrinsics are only valid at the focus they were
solved at.

**Hold still while a capture is taken.** Motion blur costs corners, which
costs pose quality, and the frame will still look fine.

## Documentation

- [`ALGORITHM.md`](ALGORITHM.md) — parameterisation, residuals, observability
  analysis and gauge fixing, per stage, with derivations.
- [`../boards/README.md`](../boards/README.md) — the calibration board: what to
  print, at what size, and how to check it came out right.

## Licence

Apache-2.0, matching [LeRobot](https://github.com/huggingface/lerobot) and
[XLeRobot](https://github.com/Vector-Wangel/XLeRobot). Built on
[SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100).
