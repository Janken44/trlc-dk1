from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import *

import serial
import time


follower_config = DK1FollowerConfig(
    port="/dev/tty.usbmodem00000000050C1",
    control_mode="pos_vel",
)
follower = DK1Follower(follower_config)

follower.connect()

joint2 = follower._motors["joint_2"]
follower._control.set_zero_position(joint2)
print(f"joint_2 ({joint2.MotorType.name}) set to zero position.")
    
follower._control.serial_.close()