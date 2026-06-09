from lerobot.scripts.lerobot_replay import replay, ReplayConfig, DatasetReplayConfig
from lerobot_robot_trlc_dk1.follower import DK1FollowerConfig
import time
import rerun as rr
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data



# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_REPO_ID = ""
EPISODE_INDEX   = 0

TEST_DURATION   = 2 * 60 * 60    # Testdauer in Sekunden (2h)

cfg = ReplayConfig(
    robot=DK1FollowerConfig(
        port="/dev/tty.usbmodem00000000050C1",
        joint_velocity_scaling=1.0,
        disable_torque_on_disconnect=False,
    ),
    dataset=DatasetReplayConfig(
        repo_id=DATASET_REPO_ID,
        episode=EPISODE_INDEX,
    ),
    play_sounds=False,
)

# ─────────────────────────────────────────────
# STRESSTEST
# ─────────────────────────────────────────────

start_time     = time.time()
last_milestone = 0

print(f"Test will run for {TEST_DURATION / 3600:.0f}h")

while (time.time() - start_time) < TEST_DURATION:

    # Fortschritt alle 15 Minuten anzeigen
    passed_time = (time.time() - start_time) / 60
    if passed_time >= last_milestone + 15:
        last_milestone += 15
        print(f"Running for {last_milestone} min")

    try:
        replay(cfg)

    except KeyboardInterrupt:
        print("\nStopping the test...")
        break

    except Exception as e:
        min_until_failure = (time.time() - start_time) / 60
        print(f"Error after {min_until_failure:.1f} minutes: {type(e).__name__}: {e}")
        break

print("Test finished.")