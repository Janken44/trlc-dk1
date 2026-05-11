"""Step through Dynamixel operating modes on Enter for current measurement.
Usage: uv run python wireless_leader/measure_current.py --port /dev/tty.usbmodem...
"""
import argparse
from dynamixel_sdk import PortHandler, PacketHandler

BAUD    = 1_000_000
ID      = 1
PROTOCOL = 2.0

ADDR_TORQUE_ENABLE   = 64
ADDR_OPERATING_MODE  = 11
ADDR_GOAL_CURRENT    = 102
ADDR_GOAL_POSITION   = 116
ADDR_CURRENT_LIMIT   = 38

MODE_CURRENT          = 0
MODE_VELOCITY         = 1
MODE_POSITION         = 3
MODE_EXT_POSITION     = 4
MODE_CURRENT_POSITION = 5
MODE_PWM              = 16

STEPS = [
    ("Torque DISABLED (standby)",         lambda ph, port: torque(ph, port, 0)),
    ("Position mode, torque ON, stationary", lambda ph, port: set_mode(ph, port, MODE_POSITION)),
    ("Current mode, Goal_Current = 0",    lambda ph, port: set_mode(ph, port, MODE_CURRENT)),
    ("Current mode, Goal_Current = 50mA", lambda ph, port: set_current(ph, port, 50)),
    ("Current mode, Goal_Current = 100mA",lambda ph, port: set_current(ph, port, 100)),
    ("Current-position mode, torque ON",  lambda ph, port: set_mode(ph, port, MODE_CURRENT_POSITION)),
]


def torque(ph, port, val):
    ph.write1ByteTxRx(port, ID, ADDR_TORQUE_ENABLE, val)

def set_mode(ph, port, mode):
    torque(ph, port, 0)
    ph.write1ByteTxRx(port, ID, ADDR_OPERATING_MODE, mode)
    torque(ph, port, 1)

def set_current(ph, port, ma):
    ph.write2ByteTxRx(port, ID, ADDR_GOAL_CURRENT, ma)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    args = parser.parse_args()

    port = PortHandler(args.port)
    ph   = PacketHandler(PROTOCOL)

    port.openPort()
    port.setBaudRate(BAUD)
    print(f"Connected to {args.port}\n")

    for i, (label, action) in enumerate(STEPS):
        input(f"[{i+1}/{len(STEPS)}] Press Enter to apply: {label}")
        action(ph, port)
        print(f"  → Applied. Measure now.\n")

    print("Done. Disabling torque.")
    torque(ph, port, 0)
    port.closePort()


if __name__ == "__main__":
    main()
