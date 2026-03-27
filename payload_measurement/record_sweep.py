"""
Record a reach sweep trajectory for payload testing.

Teleop the arm through the sweep once — this trajectory will be replayed
identically for each payload mass increment in replay_sweep.py.

Usage:
    uv run examples/payload_measurement/record_sweep.py

Output:
    outputs/payload_measurement/sweep_trajectory/

Controls:
    RIGHT ARROW  — finish recording / confirm start position
    LEFT ARROW   — re-record (during recording only)
"""

from pathlib import Path
import shutil
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.processor import make_default_processors
import threading

import rerun.blueprint as rrb

from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig

# --- Configuration -----------------------------------------------------------

FOLLOWER_PORT       = "/dev/tty.usbmodem00000000050C1"
LEADER_PORT         = "/dev/tty.usbmodem59700734821"
CURRENT_SENSOR_PORT = "/dev/tty.usbserial-BG0038JI"

FPS             = 60
MAX_DURATION_S  = 600   # hard cutoff — press right arrow to finish early
OUTPUT_DIR      = Path("outputs/payload_measurement/sweep")

# -----------------------------------------------------------------------------

follower_config = DK1FollowerConfig(
    port=FOLLOWER_PORT,
    control_mode="impedance",
    current_sensor_port=CURRENT_SENSOR_PORT,
)
leader_config = DK1LeaderConfig(port=LEADER_PORT)

follower = DK1Follower(follower_config)
leader   = DK1Leader(leader_config)

obs_features    = hw_to_dataset_features(follower.observation_features, "observation")
action_features = hw_to_dataset_features(follower.action_features, "action")

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

dataset = LeRobotDataset.create(
    repo_id="trlc/sweep_trajectory",
    fps=FPS,
    features={**obs_features, **action_features},
    robot_type=follower.name,
    root=OUTPUT_DIR,
)

_, events = init_keyboard_listener()
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

follower.connect()
leader.connect()

def teleop_to_position(prompt: str) -> None:
    """Run teleop until the user presses RIGHT ARROW to confirm position."""
    events["exit_early"] = False
    print(prompt)
    while not events["exit_early"]:
        follower.send_action(leader.get_action())
        log_rerun_data(observation=follower.get_observation())
        time.sleep(1 / FPS)


def _rerun_monitor(stop: threading.Event) -> None:
    """Background thread: log observations to Rerun while record_loop runs."""
    while not stop.is_set():
        log_rerun_data(observation=follower.get_observation())
        time.sleep(1 / FPS)


print("\n--- Payload sweep recorder ---")
init_rerun(session_name="sweep_recording")
import rerun as rr
_joints = [f"joint_{i}" for i in range(1, 7)]
rr.send_blueprint(rrb.Blueprint(
    rrb.Grid(
        rrb.TimeSeriesView(
            name="Torques (Nm)",
            contents=[f"observation.{j}.torque" for j in _joints] + ["observation.gripper.torque"],
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
            contents=[f"observation.{j}.pos" for j in _joints] + ["observation.gripper.pos"],
        ),
    ),
    collapse_panels=False,
))
teleop_to_position("Teleop the arm to the START position, then press RIGHT ARROW to begin recording.")

while True:
    events["exit_early"] = False
    events["rerecord_episode"] = False

    print("Recording — perform the reach sweep, then press RIGHT ARROW to finish.")
    _stop = threading.Event()
    threading.Thread(target=_rerun_monitor, args=(_stop,), daemon=True).start()
    record_loop(
        robot=follower,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=leader,
        dataset=dataset,
        control_time_s=MAX_DURATION_S,
        single_task="payload reach sweep",
    )
    _stop.set()

    num_frames = dataset.episode_buffer["size"]
    duration_s = num_frames / FPS
    print(f"\nRecorded {num_frames} frames ({duration_s:.1f} s).")

    if events["rerecord_episode"]:
        dataset.clear_episode_buffer()
        teleop_to_position("Move back to start position, then press RIGHT ARROW to re-record.")
        continue

    # Accept or re-record via arrow keys
    events["exit_early"] = False
    events["rerecord_episode"] = False
    print("Press RIGHT ARROW to accept, LEFT ARROW to re-record.")
    while not events["exit_early"] and not events["rerecord_episode"]:
        time.sleep(0.05)

    if events["rerecord_episode"]:
        dataset.clear_episode_buffer()
        teleop_to_position("Move back to start position, then press RIGHT ARROW to re-record.")
        continue

    break

dataset.save_episode()
dataset.finalize()
print(f"\nSweep saved → {OUTPUT_DIR}")
print("Run replay_sweep.py --mass-g <grams> to replay with a payload attached.")

follower.disconnect()
leader.disconnect()
