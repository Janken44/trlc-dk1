#!/usr/bin/env python3
"""Sequential Dynamixel ID assignment over the transparent USB bridge — chained.

Build the leader chain ONE MOTOR AT A TIME without ever unplugging: connect the
motor nearest the controller first (→ ID 1), then the next (→ ID 2), and so on,
matching the joint order in DK1Leader (joint_1..gripper == IDs 1..7).

How it stays collision-free
---------------------------
Factory-fresh XL330s are ID 1 @ 57600 baud. Each motor is moved to 1 Mbps the
moment it's assigned, so:
  * a broadcast-ping @ 1 Mbps only reaches already-configured motors,
  * a broadcast-ping @ 57600 only reaches the freshly plugged one.
Adding one fresh motor at a time therefore yields exactly one 57600 responder —
no two motors ever share an ID *on the same baud*, even though all ship as ID 1.

Reused motors already at 1 Mbps + ID 1 are the exception: they'd collide with an
assigned ID-1 motor. The script detects the ambiguity and asks you to isolate it.

Usage
-----
    uv run python leader_v1_MVP/assign_motor_ids.py --port /dev/tty.usbmodemXXXX
"""
import argparse
import sys

from serial.tools import list_ports

from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig

# Bauds to probe each step. 57600 (factory) and 1 Mbps (configured) are the common
# cases; the rest catch motors left at some other rate by a previous setup.
SCAN_BAUDS = [57600, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 115_200, 9_600]
TARGET_BAUD = 1_000_000  # DK1Leader / lerobot default


def autodetect_port():
    """Return the sole usbmodem serial port, or raise if not exactly one.

    Only one leader arm is ever connected when assigning IDs, so a single
    usbmodem port is unambiguous — no need to run lerobot-find-port.
    """
    ports = [p.device for p in list_ports.comports() if "usbmodem" in p.device]
    if not ports:
        raise LookupError(
            "No usbmodem port found. Is the leader plugged in and powered?"
        )
    if len(ports) > 1:
        raise LookupError(
            f"Multiple usbmodem ports: {ports}. Pass one explicitly with --port."
        )
    return ports[0]


def scan(bus):
    """Return {(baud, id): model} for every motor currently answering, any baud."""
    found = {}
    for baud in SCAN_BAUDS:
        bus.set_baudrate(baud)
        resp = bus.broadcast_ping()  # {id: model} or None
        if resp:
            for mid, model in resp.items():
                found[(baud, mid)] = model
    return found


def detect_new_motor(bus, assigned_finals):
    """Find the single newly plugged motor.

    A motor is 'already configured' if it answers at TARGET_BAUD with an ID we've
    already assigned. Everything else is a candidate for the new motor.
    Returns (baud, id) or raises with a human-readable reason.
    """
    present = scan(bus)
    candidates = [
        (baud, mid)
        for (baud, mid) in present
        if not (baud == TARGET_BAUD and mid in assigned_finals)
    ]
    if not candidates:
        raise LookupError("No new motor detected. Is it plugged in and powered?")
    if len(candidates) > 1:
        raise LookupError(
            f"Ambiguous — {len(candidates)} unexpected motors on the bus: {candidates}. "
            "A reused motor may already be at 1 Mbps with a duplicate ID; isolate it "
            "and set it on its own, or factory-reset it first."
        )
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", default=None,
                    help="serial port of the ESP32 bridge, e.g. /dev/tty.usbmodemXXXX "
                         "(default: auto-detect the sole usbmodem port)")
    args = ap.parse_args()

    port = args.port
    if port is None:
        try:
            port = autodetect_port()
        except LookupError as e:
            print(f"! {e}")
            sys.exit(1)
        print(f"Auto-detected port: {port}\n")

    leader = DK1Leader(DK1LeaderConfig(port=port))
    bus = leader.bus
    bus.connect(handshake=False)  # motors aren't at their target IDs yet

    print("Build the chain one motor at a time — do NOT unplug motors already set.\n")
    assigned_finals = set()
    try:
        for motor in bus.motors:                 # joint_1..gripper == IDs 1..7, in order
            target = bus.motors[motor].id
            while True:
                input(f"→ Connect the '{motor}' motor (ID {target}) to the end of the "
                      "chain, then press Enter ... ")
                try:
                    baud, cur_id = detect_new_motor(bus, assigned_finals)
                except LookupError as e:
                    print(f"   ! {e}\n     Try again (Ctrl-C to abort).\n")
                    continue

                # Reuse lerobot's tested path: disable torque, write ID, set 1 Mbps,
                # and verify the model number matches xl330-m077.
                bus.setup_motor(motor, initial_baudrate=baud, initial_id=cur_id)
                assigned_finals.add(target)
                print(f"   ✓ '{motor}' set to ID {target} @ {TARGET_BAUD} "
                      f"(was ID {cur_id} @ {baud})\n")
                break

        # Final check: every motor should now answer at the target baud.
        bus.set_baudrate(TARGET_BAUD)
        present = set(bus.broadcast_ping() or {})
        missing = sorted(assigned_finals - present)
        print(f"Done. Responding IDs @ {TARGET_BAUD}: {sorted(present)}")
        if missing:
            print(f"   ! Not responding: {missing} — recheck those connections.")
        else:
            print("   ✓ All motors assigned and responding.")
    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
