"""Serial bus mapping for this XLeRobot unit.

Both Feetech adapters are CH343 chips (1a86:55d3) with distinct burned-in USB
serial numbers, so udev rules in /etc/udev/rules.d/99-xlerobot.rules pin each one
to a stable symlink. Unlike the cameras, this mapping survives any replug order.

Verified layout (16 servos total):
    bus A / port1 -> left arm  ID 1-6 + head   ID 7, 8
    bus B / port2 -> right arm ID 1-6 + wheels ID 9, 10
"""

import os
from pathlib import Path

PORT1 = "/dev/xlerobot_bus_A"  # left arm + head, USB serial 5B8E114602
PORT2 = "/dev/xlerobot_bus_B"  # right arm + wheels, USB serial 5B3D042633

BUSES: dict[str, str] = {
    "port1 (left arm + head)": PORT1,
    "port2 (right arm + wheels)": PORT2,
}

USB_SERIALS = {
    PORT1: "5B8E114602",
    PORT2: "5B3D042633",
}

# Servo IDs expected on each bus, for the 2-wheel differential drive variant.
EXPECTED_IDS: dict[str, dict[int, str]] = {
    PORT1: {
        1: "left_arm_shoulder_pan",
        2: "left_arm_shoulder_lift",
        3: "left_arm_elbow_flex",
        4: "left_arm_wrist_flex",
        5: "left_arm_wrist_roll",
        6: "left_arm_gripper",
        7: "head_motor_1",
        8: "head_motor_2",
    },
    PORT2: {
        1: "right_arm_shoulder_pan",
        2: "right_arm_shoulder_lift",
        3: "right_arm_elbow_flex",
        4: "right_arm_wrist_flex",
        5: "right_arm_wrist_roll",
        6: "right_arm_gripper",
        9: "base_left_wheel",
        10: "base_right_wheel",
    },
}

ROBOT_TYPE = "xlerobot_2wheels"
ROBOT_ID = "xlerobot_desk"


def verify_buses(strict: bool = True) -> list[str]:
    """Check both udev symlinks exist and are readable/writable."""
    problems: list[str] = []
    for label, port in BUSES.items():
        p = Path(port)
        if not p.exists():
            problems.append(
                f"{label}: {port} missing "
                f"(udev rule not loaded, or adapter unplugged)"
            )
            continue
        try:
            # Character devices are not seekable, so use a raw fd rather than open().
            fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            os.close(fd)
        except PermissionError:
            problems.append(f"{label}: {port} exists but is not writable (check dialout group / udev MODE)")
        except OSError as exc:
            problems.append(f"{label}: {port} open failed: {exc}")

    if problems and strict:
        raise RuntimeError("Bus verification failed:\n  " + "\n  ".join(problems))
    return problems


def describe() -> str:
    lines = []
    for label, port in BUSES.items():
        target = Path(port).resolve().name if Path(port).exists() else "MISSING"
        ids = sorted(EXPECTED_IDS[port])
        lines.append(f"  {label:<28} {port:<22} -> {target:<10} IDs {ids}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"XLeRobot bus mapping (robot type: {ROBOT_TYPE}, id: {ROBOT_ID})")
    print(describe())
    problems = verify_buses(strict=False)
    print("\nOK: both buses present and writable" if not problems
          else "\nPROBLEMS:\n  " + "\n  ".join(problems))
