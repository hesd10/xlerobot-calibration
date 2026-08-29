"""Scan both Feetech buses and report which servo IDs respond on each.

Read-only: pings servos without enabling torque, so nothing moves. Use this after
rewiring, after changing servo IDs, or when a limb stops responding.

Uses the Feetech SDK directly rather than lerobot's motor bus. The two reach the
same wire, but lerobot is an optional dependency here: calibration reads raw
encoder counts and never loads a policy, so requiring it would drag a CUDA stack
into an install that will not use it. Since preflight calls this script and
preflight must run on a calibration-only machine, the import had to go.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.buses import BUSES, EXPECTED_IDS, USB_SERIALS, verify_buses  # noqa: E402

BAUD_RATE = 1_000_000
ADDR_PRESENT_POSITION = 56

# main() reports servos answering on an ID it did not expect, which is the
# symptom of a servo configured with the wrong ID after a rewire. Probing only
# the expected IDs would make that class of fault invisible, so sweep the whole
# range the configuration tool hands out.
ID_RANGE = range(1, 21)


def scan(port: str) -> dict[int, int]:
    """Return {servo_id: 1} for servos answering on this port.

    lerobot's broadcast_ping returned model numbers; nothing downstream reads
    them, only the set of keys, so a per-ID ping gives the same answer. It is
    slower -- one round trip per ID rather than one broadcast -- but this runs
    once at preflight over 20 IDs, and each miss costs only the SDK's timeout.
    """
    import scservo_sdk as scs

    handler = scs.PortHandler(port)
    packet = scs.PacketHandler(0)
    if not handler.openPort():
        raise RuntimeError(f"could not open {port}")
    try:
        if not handler.setBaudRate(BAUD_RATE):
            raise RuntimeError(f"could not set baud rate on {port}")
        found = {}
        for servo_id in ID_RANGE:
            _, comm, err = packet.read2ByteTxRx(
                handler, servo_id, ADDR_PRESENT_POSITION)
            if comm == scs.COMM_SUCCESS and err == 0:
                found[servo_id] = 1
        return found
    finally:
        handler.closePort()


def main() -> int:
    for problem in verify_buses(strict=False):
        print(f"WARNING: {problem}")

    all_ok = True
    for label, port in BUSES.items():
        expected = EXPECTED_IDS[port]
        print(f"\n=== {label} ===")
        print(f"    {port}  (USB serial {USB_SERIALS[port]})")

        try:
            found = scan(port)
        except Exception as exc:
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            all_ok = False
            continue

        ids = set(found)
        if not ids:
            print("    no servos responded (check power switch and bus wiring)")
            all_ok = False
            continue

        for sid in sorted(ids):
            print(f"    ID {sid:>2}: {expected.get(sid, '(UNEXPECTED)')}")

        missing = sorted(set(expected) - ids)
        extra = sorted(ids - set(expected))
        if missing:
            print(f"    MISSING: {[f'{i}={expected[i]}' for i in missing]}")
            all_ok = False
        if extra:
            print(f"    UNEXPECTED IDs: {extra}")
            all_ok = False
        if not missing and not extra:
            print(f"    OK: all {len(expected)} servos present")

    print()
    print("All buses match the expected layout." if all_ok
          else "Layout does not match. Check wiring, power, and servo IDs.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
