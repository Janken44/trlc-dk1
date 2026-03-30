import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pynput import keyboard as kb

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
# Freeze toggle — SPACE holds the last commanded position so you can observe
# steady-state torques at a fixed reach while logging continues uninterrupted.
# ---------------------------------------------------------------------------
_frozen       = False
_frozen_action: dict | None = None

def _on_key_press(key: kb.Key) -> None:
    global _frozen
    if key == kb.Key.space:
        _frozen = not _frozen
        state = "FROZEN — holding position" if _frozen else "LIVE — resuming teleop"
        print(f"\n[{state}]")

_kb_listener = kb.Listener(on_press=_on_key_press)
_kb_listener.start()

# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
freq      = 200  # Hz — control loop rate
viz_freq  = 60   # Hz — Rerun logging rate
viz_every = max(1, round(freq / viz_freq))

print("Running — SPACE to freeze/unfreeze, Ctrl-C to quit.")

try:
    step = 0
    while True:
        if not _frozen:
            action = leader.get_action()
            _frozen_action = action
        follower.send_action(_frozen_action)
        if step % viz_every == 0:
            obs = follower.get_observation()
            log_rerun_data(observation=obs, action=_frozen_action)
            qpos = np.array([obs[f"joint_{i}.pos"] for i in range(1, 7)])
            arm_viz.log(qpos)
        step += 1
        time.sleep(1 / freq)
except KeyboardInterrupt:
    print("\nStopping teleop...")
finally:
    _kb_listener.stop()
    leader.disconnect()
    follower.disconnect()
