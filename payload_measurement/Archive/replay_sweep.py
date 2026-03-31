"""
Replay the recorded sweep trajectory with a payload attached.

Run this once per payload mass. The arm moves to the sweep start position,
waits for you to attach the payload, then replays the trajectory and saves
joint torques + supply current for post-processing in analyse.py.

Usage:
    uv run examples/payload_measurement/replay_sweep.py --mass-g 100

    Use --mass-g 0 for the unloaded baseline.

Output:
    outputs/payload_measurement/replays/mass_<N>g/

Controls:
    RIGHT ARROW  — start replay once payload is attached
"""

import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig

# --- Configuration -----------------------------------------------------------

FOLLOWER_PORT       = "/dev/tty.usbmodem00000000050C1"
CURRENT_SENSOR_PORT = "/dev/tty.usbserial-BG0038JI"

SWEEP_DIR   = Path("outputs/payload_measurement/sweep")
REPLAYS_DIR = Path("outputs/payload_measurement/replays")
SETTLE_S    = 2.0   # seconds to hold start position before waiting for user

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Replay sweep with attached payload.")
parser.add_argument("--mass-g", type=float, required=True, help="Payload mass in grams (0 for baseline)")
args = parser.parse_args()
mass_g = args.mass_g

# Load sweep trajectory (episode 0)
if not SWEEP_DIR.exists():
    raise FileNotFoundError(f"Sweep dataset not found at {SWEEP_DIR}. Run record_sweep.py first.")

sweep = LeRobotDataset(repo_id="trlc/sweep_trajectory", root=SWEEP_DIR, episodes=[0])
state_names  = sweep.features["observation.state"]["names"]
action_names = sweep.features["action"]["names"]
fps          = sweep.fps

# Build follower (no leader needed for replay)
follower_config = DK1FollowerConfig(
    port=FOLLOWER_PORT,
    control_mode="impedance",
    current_sensor_port=CURRENT_SENSOR_PORT,
)
follower = DK1Follower(follower_config)
obs_features    = hw_to_dataset_features(follower.observation_features, "observation")
action_features = hw_to_dataset_features(follower.action_features, "action")

# Prepare output dataset
out_dir = REPLAYS_DIR / f"mass_{mass_g:.0f}g"
if out_dir.exists():
    shutil.rmtree(out_dir)

dataset = LeRobotDataset.create(
    repo_id=f"trlc/payload_mass_{mass_g:.0f}g",
    fps=fps,
    features={**obs_features, **action_features},
    robot_type=follower.name,
    root=out_dir,
)

_, events = init_keyboard_listener()
init_rerun(session_name=f"replay_mass_{mass_g:.0f}g")

follower.connect()

try:
    # Extract start position from first frame
    start_action_arr = sweep[0]["action"].numpy()
    start_action = {name: float(v) for name, v in zip(action_names, start_action_arr)}

    print(f"\n--- Payload replay: {mass_g:.0f} g ---")
    print("Moving to sweep start position...")
    t_settle = time.perf_counter()
    while time.perf_counter() - t_settle < SETTLE_S:
        follower.send_action(start_action)
        time.sleep(1 / fps)

    # Hold start position until user confirms payload is attached
    events["exit_early"] = False
    label = f"{mass_g:.0f} g payload" if mass_g > 0 else "no payload (baseline)"
    print(f"Attach {label}, then press RIGHT ARROW to start replay.")
    while not events["exit_early"]:
        follower.send_action(start_action)
        obs = follower.get_observation()
        log_rerun_data(observation=obs)
        time.sleep(1 / fps)

    # Replay
    print(f"Replaying {len(sweep)} frames ({len(sweep) / fps:.1f} s)...")
    t_start = time.perf_counter()
    for i, frame in enumerate(sweep):
        t_frame = t_start + i / fps

        # Send action from recorded trajectory
        action_arr = frame["action"].numpy()
        action_dict = {name: float(v) for name, v in zip(action_names, action_arr)}
        follower.send_action(action_dict)

        # Record and visualise observation
        obs = follower.get_observation()
        state_vec = np.array([obs[name] for name in state_names], dtype=np.float32)
        dataset.add_frame({
            "observation.state": state_vec,
            "action": action_arr,
            "task": "payload reach sweep",
        })
        log_rerun_data(observation=obs, action=action_dict)

        # Pace to FPS
        sleep_s = t_frame + 1 / fps - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)

    dataset.save_episode()
    dataset.finalize()
    print(f"\nReplay saved → {out_dir}")
    print(f"{len(sweep)} frames recorded at {mass_g:.0f} g payload.")
    print("Run analyse.py to update the payload chart.")

finally:
    follower.disconnect()
