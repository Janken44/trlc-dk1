from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.leader_ble import DK1LeaderBLE, DK1LeaderBLEConfig
import time


follower_config = DK1FollowerConfig(
    port="/dev/tty.usbmodem00000000050C1",
    joint_velocity_scaling=0.2,
)

leader_config = DK1LeaderBLEConfig()

leader = DK1LeaderBLE(leader_config)
leader.connect()

follower = DK1Follower(follower_config)
follower.connect()

freq = 200 # Hz

try:
    while True:
        action = leader.get_action()
        follower.send_action(action)    
        time.sleep(1/freq)
except KeyboardInterrupt:
    print("\nStopping teleop...")
    leader.disconnect()
    follower.disconnect()
