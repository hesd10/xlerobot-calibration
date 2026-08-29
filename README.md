# XLeRobot calibration

Guided calibration that measures where this robot's cameras and arms actually
are, and writes the result back into the MuJoCo model, so that a pose commanded
in simulation lands in the same place on the bench.

Bugs are likely — this has been run on one bench by one operator, and yours will
differ. Issues and pull requests are welcome.

## Before you start

You need an assembled XLeRobot, wired and powered, with its servo IDs already
configured and its three cameras plugged in. Setting that up is
[XLeRobot's own business](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble.html),
and this tool assumes it is done: it measures a robot, it does not build one.

What it does not assume is XLeRobot's software stack. Calibration reads raw
encoder counts straight off the servos and never loads a policy, so neither
`lerobot` nor `torch` is needed here, and the install below does not pull them
in. If you already have them, the startup checks will say so and carry on.

## Start here

Python 3.10 or newer:

```bash
pip install -e '.[legacy]'
python -m xlerobot_calibration_tool
```

Then open <http://127.0.0.1:8422> and work through the eight stages in order.
The tool blocks a stage whose inputs are missing, so the order is not something
you have to remember.

That one install command brings in everything, MuJoCo included. MuJoCo is not
optional here: every stage evaluates forward kinematics through the model to
turn joint angles into frames, and the final export writes the measured geometry
back into the XML. Stage 4 also asks you to open the model and copy the pose you
see, which opens a viewer window, so run the tool somewhere with a display
rather than over a plain SSH session.

To keep a run apart from earlier ones, name its workspace:

```bash
python -m xlerobot_calibration_tool --workspace ~/my_calibration
```

## The documentation

- **[`calibration/docs/README.md`](calibration/docs/README.md)** — what the tool
  measures, how to mount the robot, what to print, how to run each stage and
  how accurate the result is. **Read this first.**
- [`calibration/docs/ALGORITHM.md`](calibration/docs/ALGORITHM.md) — the
  parameterisation, residuals, observability analysis and gauge fixing behind
  each stage, with derivations.
- [`calibration/boards/README.md`](calibration/boards/README.md) — the ChArUco
  board: what to print, at what size, and how to check it came out right.

## What is in here

```
calibration/core      the geometry: board detection, models, solvers, gates
calibration/stages    one script per stage, run by the dashboard
calibration/boards    the ChArUco board to print, and its description
calibration/model     the nominal MuJoCo model and its meshes
tools/                camera identification and the hardware preflight checks
xlerobot_calibration_tool  the dashboard: workflow, guidance, result storage
```

Results are written to the workspace, never into this directory. Each finished
calibration leaves `xlerobot_calib_fitted.xml` — the model with the measured
geometry baked in — alongside a `README.md` explaining every file it produced.

## Licence

Apache-2.0. Built on [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100)
and [LeRobot](https://github.com/huggingface/lerobot).
