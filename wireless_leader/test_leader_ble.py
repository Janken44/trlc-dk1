"""Quick test: connect to DK1-Leader BLE and print positions at ~10 Hz."""
import time
from lerobot_robot_trlc_dk1 import DK1LeaderBLE, DK1LeaderBLEConfig

leader = DK1LeaderBLE(DK1LeaderBLEConfig())
leader.connect()

try:
    while True:
        action = leader.get_action()
        parts = [f"{k.split('.')[0]}={v:+.3f}" for k, v in action.items()]
        print("  ".join(parts))
        time.sleep(0.03)
except KeyboardInterrupt:
    pass
finally:
    leader.disconnect()
