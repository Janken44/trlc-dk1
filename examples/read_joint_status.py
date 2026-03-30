from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import *
import serial
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--joint', type=int, required=True, help="Joint number (1-7)")
args = parser.parse_args()

# Calculate the hex values based on the input
# If joint is 4, id_val becomes 0x04 (4) and type_val becomes 0x14 (20)
id_val = args.joint
type_val = args.joint + 0x10

motor=Motor(DM_Motor_Type.DM4340, id_val, type_val)

serial_device = serial.Serial('/dev/tty.usbmodem00000000050C1', 921600, timeout=0.5)
time.sleep(0.5)

control=MotorControl(serial_device)
control.addMotor(motor)

print("\n--- Live Status ---")
control.refresh_motor_status(motor)
# Give extra time for response, then drain the serial buffer
time.sleep(0.05)
control.recv()

print(f"Position  : {motor.getPosition():.4f} rad")
print(f"Velocity  : {motor.getVelocity()} rad/s")
print(f"Torque    : {motor.getTorque()} Nm")
print(f"Temp MOS  : {motor.getTemperatureMOS()} °C")
print(f"Temp Motor: {motor.getTemperatureMotor()} °C")

err = motor.getError()
err_meanings = {
    0x0: "Disabled",
    0x1: "Enabled (Normal)",
    0x3: "Output Shaft Calibration Error",
    0x4: "Sensor Output Abnormal",
    0x5: "Motor Encoder Calibration Error",
    0x8: "Overvoltage",
    0x9: "Undervoltage",
    0xA: "Overcurrent",
    0xB: "MOS Overheating",
    0xC: "Coil Overheating",
    0xD: "Communication Loss",
    0xE: "Overload",
}
print(f"Error Code: 0x{err:02X} ({err_meanings.get(err, 'Unknown')})")
