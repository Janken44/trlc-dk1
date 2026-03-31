import sys
import threading
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig
from trlc_dk1_control.rerun_viz import ArmViz


follower_config = DK1FollowerConfig(
    port="/dev/tty.usbmodem00000000050C1",
    control_mode="impedance",
    current_sensor_port="/dev/tty.usbserial-BG0038JI"
)

leader_config = DK1LeaderConfig(
    port="/dev/tty.usbmodem59700734821"
)

leader = DK1Leader(leader_config)
leader.connect()

follower = DK1Follower(follower_config)
follower.connect()

arm_viz = ArmViz()

init_rerun(session_name="teleop")

_joints = [f"joint_{i}" for i in range(1, 7)]
rr.send_blueprint(rrb.Blueprint(
    rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial3DView(
                name="Arm 3D",
                contents=["arm/**"],
            ),
            rrb.TimeSeriesView(
                name="Tool0 Position (mm)",
                contents=["tool0.x_mm", "tool0.y_mm", "tool0.z_mm",
                           "tool0.reach_mm"],
            ),
            row_shares=[2, 1],
        ),
        rrb.Grid(
            rrb.TimeSeriesView(
                name="Torques (Nm)",
                contents=[f"observation.{j}.torque" for j in _joints]
                        + ["observation.gripper.torque"],
            ),
            rrb.TimeSeriesView(
                name="Temperatures (°C)",
                contents=[f"observation.{j}.temp_motor" for j in _joints]
                        + [f"observation.{j}.temp_mos" for j in _joints],
            ),
            rrb.TimeSeriesView(
                name="Supply Current (A)",
                contents=["observation.external_current_a"],
            ),
            rrb.TimeSeriesView(
                name="Positions (rad)",
                contents=[f"observation.{j}.pos" for j in _joints]
                        + ["observation.gripper.pos"],
            ),
            grid_columns=2,
        ),
        column_shares=[1, 2],
    ),
    collapse_panels=False,
))

# ---------------------------------------------------------------------------
# State: freeze, joint selection, nudge offsets
# ---------------------------------------------------------------------------
_frozen        = False
_frozen_action: dict | None = None
_selected_joint: int | None = None   # 0-based
_nudge_offsets  = np.zeros(6)        # rad, added on top of frozen position

def _print_status() -> None:
    sel = f"J{_selected_joint + 1}" if _selected_joint is not None else "none"
    mode = "FROZEN" if _frozen else "LIVE"
    offsets_str = "  ".join(
        f"J{i+1}={_nudge_offsets[i]:+.4f}" for i in range(6)
        if _nudge_offsets[i] != 0.0
    ) or "all zero"
    print(f"[{mode}  sel={sel}  nudges: {offsets_str}]")

# ---------------------------------------------------------------------------
# stdin command thread — only active when terminal is focused + Enter pressed
# ---------------------------------------------------------------------------
def _stdin_loop() -> None:
    global _frozen, _selected_joint
    while True:
        try:
            line = input("> ").strip().lower()
        except EOFError:
            break
        if not line:
            continue

        # Freeze / unfreeze
        if line == "f":
            _frozen = not _frozen
            if not _frozen:
                _nudge_offsets[:] = 0.0
            _print_status()

        # Joint selection: "1" through "6"
        elif line in "123456" and len(line) == 1:
            _selected_joint = int(line) - 1
            _print_status()

        # Nudge: "+" / "-" for fine, "++" / "--" for coarse
        elif line in ("+", "++", "-", "--"):
            if _selected_joint is None:
                print("Select a joint first (1-6)")
                continue
            step = 0.1 if len(line) == 2 else 0.01
            sign = 1.0 if "+" in line else -1.0
            _nudge_offsets[_selected_joint] += sign * step
            _print_status()

        # Set a specific joint angle: "j4 0.15" or "j4=0.15"
        elif line.startswith("j") and len(line) > 2:
            try:
                parts = line[1:].replace("=", " ").split()
                jnum = int(parts[0])
                val = float(parts[1])
                if 1 <= jnum <= 6:
                    _nudge_offsets[jnum - 1] = val
                    _selected_joint = jnum - 1
                    _print_status()
                else:
                    print("Joint must be 1-6")
            except (ValueError, IndexError):
                print("Usage: j4 0.15  or  j4=0.15")

        # Zero selected joint nudge
        elif line == "z":
            if _selected_joint is not None:
                _nudge_offsets[_selected_joint] = 0.0
            _print_status()

        # Zero all nudges
        elif line == "zz":
            _nudge_offsets[:] = 0.0
            _print_status()

        # Print full state
        elif line == "p":
            print(f"Nudge offsets (rad): {_nudge_offsets.tolist()}")

        # Help
        elif line in ("h", "help", "?"):
            print(
                "Commands:\n"
                "  f         — freeze / unfreeze (clears nudges on unfreeze)\n"
                "  1-6       — select joint for nudging\n"
                "  + / -     — nudge selected joint ±0.01 rad\n"
                "  ++ / --   — nudge selected joint ±0.1 rad\n"
                "  j4 0.15   — set J4 nudge offset directly\n"
                "  z         — zero selected joint nudge\n"
                "  zz        — zero all nudges\n"
                "  p         — print all nudge offsets\n"
                "  h         — this help\n"
                "  Ctrl-C    — quit"
            )
        else:
            print(f"Unknown command: '{line}' (h for help)")


_cmd_thread = threading.Thread(target=_stdin_loop, daemon=True)
_cmd_thread.start()

# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
freq      = 200  # Hz — control loop rate
viz_freq  = 60   # Hz — Rerun logging rate
viz_every = max(1, round(freq / viz_freq))

print("Teleop running — type 'h' + Enter for commands, Ctrl-C to quit.")
print("> ", end="", flush=True)

try:
    step = 0
    while True:
        if not _frozen:
            action = leader.get_action()
            _frozen_action = action
        # Apply per-joint nudge offsets on top of the frozen position
        if _frozen and np.any(_nudge_offsets != 0.0):
            action_to_send = dict(_frozen_action)
            for i, offset in enumerate(_nudge_offsets):
                if offset != 0.0:
                    k = f"joint_{i + 1}.pos"
                    action_to_send[k] = action_to_send[k] + offset
        else:
            action_to_send = _frozen_action
        follower.send_action(action_to_send)
        if step % viz_every == 0:
            obs = follower.get_observation()
            log_rerun_data(observation=obs, action=action_to_send)
            qpos = np.array([obs[f"joint_{i}.pos"] for i in range(1, 7)])
            arm_viz.log(qpos)
        step += 1
        time.sleep(1 / freq)
except KeyboardInterrupt:
    print("\nStopping teleop...")
finally:
    leader.disconnect()
    follower.disconnect()
