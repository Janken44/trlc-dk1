"""
Cartesian-space WASD teleop for the TRLC-DK1.

Smooth velocity-based control on top of the joint impedance controller. Each
keypress latches a Cartesian velocity for a short TTL; terminal auto-repeat
keeps motion alive while a key is held; release decays smoothly to zero.

Controls (terminal must have focus):

    Translation — world frame
        w / s        +X / -X     (forward / back)
        a / d        +Y / -Y     (left / right)
        r / f        +Z / -Z     (up / down)

    Rotation — tool frame  (rolls/pitches/yaws about the gripper's own axes)
        u / o        roll  -X / +X
        i / k        pitch +Y / -Y
        j / l        yaw   +Z / -Z

    Speed adjust
        +  /  -      increase / decrease translation + rotation speed
        1            slow preset    (3 cm/s, 0.3 rad/s)
        2            normal preset  (10 cm/s, 1.0 rad/s)
        3            fast preset    (20 cm/s, 2.0 rad/s)

    Gripper
        space        toggle open / closed

    Misc
        h            home target to current measured pose
        p            print current target + EE pose
        esc          quit  (ctrl-c also works)

Run:
    uv run python examples/cartesian_wasd.py --port /dev/tty.usbmodemXXXX
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
from contextlib import contextmanager

import numpy as np

from trlc_dk1_control import (
    CartesianController,
    CartesianControllerConfig,
    DK1_DEFAULT_CONFIG,
    DK1Robot,
)


# Speed presets (lin m/s, rot rad/s)
PRESETS = {
    "slow":   (0.03, 0.3),
    "normal": (0.10, 1.0),
    "fast":   (0.20, 2.0),
}

# Translation: world frame.   value = (linear_unit_vec, "world")
# Rotation:    tool frame.    value = (angular_unit_vec, "tool")
TRANSLATE_KEYS = {
    "w": np.array([+1.0, 0.0, 0.0]),
    "s": np.array([-1.0, 0.0, 0.0]),
    "a": np.array([0.0, +1.0, 0.0]),
    "d": np.array([0.0, -1.0, 0.0]),
    "r": np.array([0.0, 0.0, +1.0]),
    "f": np.array([0.0, 0.0, -1.0]),
}
ROTATE_KEYS = {
    "u": np.array([-1.0, 0.0, 0.0]),   # roll -
    "o": np.array([+1.0, 0.0, 0.0]),   # roll +
    "i": np.array([0.0, +1.0, 0.0]),   # pitch +
    "k": np.array([0.0, -1.0, 0.0]),   # pitch -
    "j": np.array([0.0, 0.0, +1.0]),   # yaw +
    "l": np.array([0.0, 0.0, -1.0]),   # yaw -
}

ESC = "\x1b"
CTRL_C = "\x03"


@contextmanager
def cbreak_stdin():
    """Put stdin in cbreak mode — get keystrokes without waiting for newline."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key(timeout: float) -> str | None:
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return None
    return sys.stdin.read(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="DK1 follower serial port")
    parser.add_argument(
        "--mode", choices=("impedance", "position"), default="impedance",
        help="impedance = MIT PD + gravity comp (compliant); "
             "position = motor onboard POS_VEL PID (stiff, no gravity comp)",
    )
    parser.add_argument("--preset", choices=PRESETS.keys(), default="normal",
                        help="Initial speed preset")
    parser.add_argument("--control-hz", type=float, default=100.0)
    args = parser.parse_args()

    print(__doc__)
    print("-" * 64)

    cfg = DK1_DEFAULT_CONFIG(args.port)
    cfg.control_mode = args.mode
    robot = DK1Robot(cfg)
    print("Connecting robot...")
    robot.connect()

    cart_cfg = CartesianControllerConfig(control_hz=args.control_hz)
    cart = CartesianController(robot, cart_cfg)
    cart.start()

    lin_speed, rot_speed = PRESETS[args.preset]
    print(f"Mode:   {args.mode}")
    print(f"Preset: {args.preset}  ({lin_speed*100:.1f} cm/s, {rot_speed:.2f} rad/s)")
    pos, _ = cart.get_target()
    print(f"Initial target pos = {np.round(pos, 4)}")
    print("Ready. WASD/RF translate, UIOJKL rotate, space=gripper, esc=quit.\n")

    gripper_closed = False

    try:
        with cbreak_stdin():
            while True:
                key = read_key(timeout=0.05)
                if key is None:
                    continue
                if key in (ESC, CTRL_C):
                    break

                k = key.lower()

                if k in TRANSLATE_KEYS:
                    cart.set_velocity(
                        lin=TRANSLATE_KEYS[k] * lin_speed,
                        rot=(0.0, 0.0, 0.0),
                        lin_frame="world",
                        rot_frame="tool",
                    )
                elif k in ROTATE_KEYS:
                    cart.set_velocity(
                        lin=(0.0, 0.0, 0.0),
                        rot=ROTATE_KEYS[k] * rot_speed,
                        lin_frame="world",
                        rot_frame="tool",
                    )
                elif k == " ":
                    gripper_closed = not gripper_closed
                    cart.set_gripper(1.0 if gripper_closed else 0.0)
                    print(f"gripper -> {'closed' if gripper_closed else 'open'}")
                elif k == "h":
                    cart.reset_target_to_current()
                    tp, _ = cart.get_target()
                    print(f"target reset to current: {np.round(tp, 4)}")
                elif k == "p":
                    tp, _ = cart.get_target()
                    ep, _ = cart.get_ee_pose()
                    print(
                        f"target={np.round(tp, 4)}  ee={np.round(ep, 4)}  "
                        f"err={np.round((tp - ep) * 1000, 1)} mm"
                    )
                elif k in ("+", "="):
                    lin_speed = min(lin_speed * 1.25, 0.5)
                    rot_speed = min(rot_speed * 1.25, 4.0)
                    print(f"speed: {lin_speed*100:.1f} cm/s, {rot_speed:.2f} rad/s")
                elif k in ("-", "_"):
                    lin_speed = max(lin_speed / 1.25, 0.005)
                    rot_speed = max(rot_speed / 1.25, 0.05)
                    print(f"speed: {lin_speed*100:.1f} cm/s, {rot_speed:.2f} rad/s")
                elif k == "1":
                    lin_speed, rot_speed = PRESETS["slow"]
                    print(f"preset: slow ({lin_speed*100:.1f} cm/s, {rot_speed:.2f} rad/s)")
                elif k == "2":
                    lin_speed, rot_speed = PRESETS["normal"]
                    print(f"preset: normal ({lin_speed*100:.1f} cm/s, {rot_speed:.2f} rad/s)")
                elif k == "3":
                    lin_speed, rot_speed = PRESETS["fast"]
                    print(f"preset: fast ({lin_speed*100:.1f} cm/s, {rot_speed:.2f} rad/s)")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        cart.stop()
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
