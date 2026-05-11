"""Debug test: USB leader only — tracks packet rate, seq gaps, and reconnect behaviour."""
import logging
import time
import serial

from lerobot_robot_trlc_dk1.leader_usb import DK1LeaderUSB, DK1LeaderUSBConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s.%(msecs)03d %(message)s",
                    datefmt="%H:%M:%S")

PORT = "/dev/tty.usbmodemDCB4D93A6B541"

leader = DK1LeaderUSB(DK1LeaderUSBConfig(port=PORT))
leader.connect()
print("Connected.\n")

t_start    = time.perf_counter()
t_report   = t_start
last_count = 0

try:
    while True:
        now = time.perf_counter()

        # print packet rate every 2 s
        if now - t_report >= 2.0:
            with leader._lock:
                count = leader._pkt_count
                age_ms = (now - leader._latest_time) * 1e3
            hz = (count - last_count) / (now - t_report)
            elapsed = now - t_start
            print(f"t={elapsed:6.1f}s  rate={hz:5.1f} Hz  total={count}  last_age={age_ms:.0f} ms")
            last_count = count
            t_report = now

        try:
            action = leader.get_action()
        except Exception as e:
            with leader._lock:
                age_ms = (time.perf_counter() - leader._latest_time) * 1e3
                count  = leader._pkt_count
            elapsed = time.perf_counter() - t_start
            print(f"\n*** FAILURE at t={elapsed:.1f}s after {count} packets, last packet {age_ms:.0f} ms ago")
            print(f"    Error: {e}")

            # check if port is still present (ESP32 reset vs. port stolen)
            time.sleep(0.5)
            try:
                s = serial.Serial(PORT, 115200, timeout=0.5)
                s.close()
                print("    Port still openable → ESP32 reset and re-enumerated (same path)")
            except serial.SerialException as se:
                print(f"    Port not openable: {se}")
            break

        time.sleep(0.02)  # 50 Hz poll

except KeyboardInterrupt:
    elapsed = time.perf_counter() - t_start
    with leader._lock:
        count = leader._pkt_count
    print(f"\nStopped after {elapsed:.1f}s, {count} packets total.")
finally:
    leader.disconnect()
