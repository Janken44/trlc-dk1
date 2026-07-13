"""Standalone test for the transparent USB leader — no follower required.
Connects via DK1Leader (DynamixelMotorsBus) exactly as teleop.py would, prints
joint positions plus the handle button-node state (ID 8) at ~10 Hz.

The button node is a separate Protocol 2.0 slave on the same bus (see
sketch_handle-buttonnode). It is NOT a configured motor, so we read it directly
off the bus's packet handler rather than through DK1Leader's sync_read.
"""
import logging
import time

from dynamixel_sdk import COMM_SUCCESS
from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

BUTTON_ID = 8        # handle button-node Dynamixel ID
ADDR_BUTTONS = 100   # 1 byte: bit0 = button 1 (D7), bit1 = button 2 (D8)

leader = DK1Leader(DK1LeaderConfig(port="/dev/tty.usbmodem5A460835011"))
leader.connect()
print("Connected. Move the arm and press the button. Ctrl-C to stop.\n")


def read_buttons() -> str:
    """Read the packed button byte off the shared bus. Returns 'b1=.. b2=..' or '?'."""
    val, comm, err = leader.bus.packet_handler.read1ByteTxRx(
        leader.bus.port_handler, BUTTON_ID, ADDR_BUTTONS
    )
    if comm != COMM_SUCCESS or err != 0:
        return "btn=?"
    b1 = "DOWN" if val & 0b01 else "up"
    b2 = "DOWN" if val & 0b10 else "up"
    return f"b1={b1} b2={b2}"


try:
    while True:
        action = leader.get_action()
        parts = [f"{k.split('.')[0]}={v:+.3f}" for k, v in action.items()]
        parts.append(read_buttons())
        print("  ".join(parts))
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    leader.disconnect()
