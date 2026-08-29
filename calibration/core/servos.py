"""Raw servo access for calibration.

The document's model is q = s * (2*pi/4096) * (r - r0), where r must be the bare
encoder reading. lerobot's connect() applies the firmware Homing_Offset and
returns calibrated values, so using it here would make r0 and the firmware offset
mutually dependent and the result meaningless.

This module therefore talks to the Feetech bus directly, reads Present_Position
without any calibration layer, and never enables torque. Reading is always safe;
the arms stay backdrivable so a human can pose them by hand.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from config.buses import EXPECTED_IDS, PORT1, PORT2  # noqa: E402

COUNTS_PER_TURN = 4096

# Feetech STS3215 control table, the subset needed for read-only access.
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63
ADDR_TORQUE_ENABLE = 40
ADDR_HOMING_OFFSET = 31

# Motors that must never be commanded during calibration.
WHEEL_MOTORS = ("base_left_wheel", "base_right_wheel")


class RawServoBus:
    """One serial bus, read directly with the Feetech SDK."""

    def __init__(self, port: str, protocol: int = 0):
        import scservo_sdk as scs

        self.scs = scs
        self.port_name = port
        self.motors = EXPECTED_IDS[port]
        self.port = scs.PortHandler(port)
        self.packet = scs.PacketHandler(protocol)
        if not self.port.openPort():
            raise RuntimeError(f"could not open {port}")
        if not self.port.setBaudRate(1_000_000):
            self.port.closePort()
            raise RuntimeError(f"could not set baud rate on {port}")

    def close(self) -> None:
        try:
            self.port.closePort()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _read_word(self, servo_id: int, addr: int) -> int | None:
        value, comm, err = self.packet.read2ByteTxRx(self.port, servo_id, addr)
        if comm != self.scs.COMM_SUCCESS or err != 0:
            return None
        return int(value)

    def _read_byte(self, servo_id: int, addr: int) -> int | None:
        value, comm, err = self.packet.read1ByteTxRx(self.port, servo_id, addr)
        if comm != self.scs.COMM_SUCCESS or err != 0:
            return None
        return int(value)

    def read_raw(self, name: str, retries: int = 3) -> int | None:
        """Raw encoder counts for one motor by name, 0..4095."""
        servo_id = self._id_of(name)
        for _ in range(retries):
            value = self._read_word(servo_id, ADDR_PRESENT_POSITION)
            if value is not None:
                return value & 0x0FFF
            time.sleep(0.002)
        return None

    def read_all_raw(self, retries: int = 3) -> dict[str, int | None]:
        return {name: self.read_raw(name, retries) for name in self.motors.values()}

    def read_health(self, name: str) -> dict:
        servo_id = self._id_of(name)
        volts = self._read_byte(servo_id, ADDR_PRESENT_VOLTAGE)
        temp = self._read_byte(servo_id, ADDR_PRESENT_TEMPERATURE)
        torque = self._read_byte(servo_id, ADDR_TORQUE_ENABLE)
        offset = self._read_word(servo_id, ADDR_HOMING_OFFSET)
        return {
            "id": servo_id,
            "raw": self.read_raw(name),
            "volts": None if volts is None else volts / 10.0,
            "temp_c": temp,
            "torque_on": None if torque is None else bool(torque),
            "firmware_homing_offset": offset,
        }

    def ping(self, name: str) -> bool:
        servo_id = self._id_of(name)
        _, comm, _ = self.packet.ping(self.port, servo_id)
        return comm == self.scs.COMM_SUCCESS

    def _id_of(self, name: str) -> int:
        for servo_id, motor in self.motors.items():
            if motor == name:
                return servo_id
        raise KeyError(f"{name} is not on {self.port_name}")


class RawRobot:
    """Both buses together, addressed purely by motor name."""

    def __init__(self):
        self.buses = [RawServoBus(PORT1), RawServoBus(PORT2)]
        self._owner: dict[str, RawServoBus] = {}
        for bus in self.buses:
            for name in bus.motors.values():
                self._owner[name] = bus

    def close(self) -> None:
        for bus in self.buses:
            bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def motor_names(self) -> list[str]:
        return list(self._owner)

    def read_raw(self, name: str) -> int | None:
        return self._owner[name].read_raw(name)

    def read_all_raw(self) -> dict[str, int | None]:
        out: dict[str, int | None] = {}
        for bus in self.buses:
            out.update(bus.read_all_raw())
        return out

    def read_arm_raw(self, side: str) -> dict[str, int | None]:
        prefix = f"{side}_arm_"
        return {n: self.read_raw(n) for n in self._owner if n.startswith(prefix)}

    def read_health_all(self) -> dict[str, dict]:
        return {name: bus.read_health(name)
                for bus in self.buses for name in bus.motors.values()}

    def verify(self) -> list[str]:
        """Confirm every expected servo answers and no torque is on."""
        problems = []
        for bus in self.buses:
            for servo_id, name in sorted(bus.motors.items()):
                if not bus.ping(name):
                    problems.append(f"{name} (id {servo_id} on {bus.port_name}) not responding")
                    continue
                health = bus.read_health(name)
                if health["raw"] is None:
                    problems.append(f"{name}: position read failed")
                if health["torque_on"]:
                    problems.append(f"{name}: torque is ENABLED, should be off for calibration")
                if health["volts"] is not None and health["volts"] < 9.0:
                    problems.append(f"{name}: supply {health['volts']:.1f} V looks low")
                if health["temp_c"] is not None and health["temp_c"] > 55:
                    problems.append(f"{name}: {health['temp_c']} C is hot, let it cool")
        return problems


def unwrap_delta(delta: int, counts: int = COUNTS_PER_TURN) -> int:
    """Shortest signed count difference, handling the 4095 -> 0 seam.

    A single-turn absolute encoder wraps, so a raw difference of +4080 really
    means -16. Valid whenever the true motion is under half a turn, which holds
    for every joint on this robot except head pan at its extremes.
    """
    return (int(delta) + counts // 2) % counts - counts // 2


def raw_to_rad(raw: float, zero: float, sign: int = 1,
               counts: int = COUNTS_PER_TURN) -> float:
    """Encoder counts -> radians, wrap-aware. The document's forward model."""
    import math

    return sign * (2.0 * math.pi / counts) * unwrap_delta(int(round(raw - zero)), counts)


def rad_to_raw(q: float, zero: float, sign: int = 1,
               counts: int = COUNTS_PER_TURN) -> float:
    """Radians -> encoder counts, the inverse of raw_to_rad."""
    import math

    return (sign * q * counts / (2.0 * math.pi) + zero) % counts


def zero_with_angle_correction(zero: int, correction_rad: float,
                               sign: int = 1,
                               counts: int = COUNTS_PER_TURN) -> int:
    """Move a raw zero so live angles gain a solved model-space correction."""
    import math

    correction_counts = int(round(
        sign * float(correction_rad) * counts / (2.0 * math.pi)))
    return (int(zero) - correction_counts) % counts


def settled(samples: list[int], tolerance: int = 2) -> bool:
    """True when a run of readings is stable enough to record."""
    if len(samples) < 2:
        return False
    spread = max(unwrap_delta(s - samples[0]) for s in samples)
    low = min(unwrap_delta(s - samples[0]) for s in samples)
    return (spread - low) <= tolerance


def wait_until_still(robot: RawRobot, names: list[str], tolerance: int = 2,
                     window: int = 5, timeout: float = 10.0) -> bool:
    """Block until the listed motors stop moving, so frames are not smeared."""
    history: dict[str, list[int]] = {n: [] for n in names}
    deadline = time.time() + timeout
    while time.time() < deadline:
        for name in names:
            value = robot.read_raw(name)
            if value is not None:
                history[name].append(value)
                history[name] = history[name][-window:]
        if all(len(h) >= window and settled(h, tolerance) for h in history.values()):
            return True
        time.sleep(0.05)
    return False
