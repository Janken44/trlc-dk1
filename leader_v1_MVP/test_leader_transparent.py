"""Standalone test for the transparent USB leader — no follower required.
Connects via DK1Leader (DynamixelMotorsBus) exactly as teleop.py would,
prints joint positions at ~10 Hz.
"""
import logging
import time

from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

leader = DK1Leader(DK1LeaderConfig(port="/dev/tty.usbmodemDCB4D93A6B541"))
leader.connect()
print("Connected. Move the arm. Ctrl-C to stop.\n")

try:
    while True:
        action = leader.get_action()
        parts = [f"{k.split('.')[0]}={v:+.3f}" for k, v in action.items()]
        print("  ".join(parts))
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    leader.disconnect()
