from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.leader_usb import DK1LeaderUSB, DK1LeaderUSBConfig
import logging
import time

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

leader_config = DK1LeaderUSBConfig(
    port="/dev/tty.usbmodem21101",
)

follower_config = DK1FollowerConfig(
    port="/dev/tty.usbmodem00000000050C1",
    joint_velocity_scaling=0.2,
)

print("Connecting leader...", flush=True)
leader = DK1LeaderUSB(leader_config)
leader.connect()  # sends START, blocks until first packet received
print("Leader connected.", flush=True)

print("Connecting follower...", flush=True)
follower = DK1Follower(follower_config)
follower.connect()
print("Follower connected.", flush=True)

print("Streaming. Ctrl-C to stop.\n", flush=True)

freq = 50  # Hz

try:
    while True:
        action = leader.get_action()
        follower.send_action(action)
        time.sleep(1 / freq)
except KeyboardInterrupt:
    print("\nStopping.")
except Exception as e:
    print(f"\nError: {e}")
finally:
    leader.disconnect()
    follower.disconnect()
