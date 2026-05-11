import struct
import threading
import time
import logging
from dataclasses import dataclass

import serial

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

_MOTOR_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
_MAGIC       = bytes([0xAA, 0x55])
_PAYLOAD_FMT = "<Bfffffff"   # seq(u8) + 7 × float32  — same layout as BLE version
_PAYLOAD_LEN = struct.calcsize(_PAYLOAD_FMT)   # 29 bytes
_PKT_LEN     = len(_MAGIC) + _PAYLOAD_LEN      # 31 bytes


@TeleoperatorConfig.register_subclass("dk1_leader_usb")
@dataclass
class DK1LeaderUSBConfig(TeleoperatorConfig):
    port: str = "/dev/tty.usbmodem00000000050C1"
    baudrate: int = 115200
    gripper_open_pos: int = 2280
    gripper_closed_pos: int = 1670
    timeout_ms: float = 500.0


class DK1LeaderUSB(Teleoperator):
    config_class = DK1LeaderUSBConfig
    name = "dk1_leader_usb"

    def __init__(self, config: DK1LeaderUSBConfig):
        super().__init__(config)
        self.config  = config
        self._ser: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._lock        = threading.Lock()
        self._latest: dict | None = None
        self._latest_time = 0.0
        self._running     = False
        self._last_seq: int | None = None
        self._pkt_count   = 0

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in _MOTOR_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._running and self._ser is not None and self._ser.is_open

    def connect(self, calibrate: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self._ser = serial.Serial(self.config.port, self.config.baudrate, timeout=1.0)
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        time.sleep(0.1)  # let serial settle before sending command
        self._ser.write(bytes([0xAA, 0x01]))  # START
        self._ser.flush()

        # wait for first packet (confirms ESP32 is streaming)
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            with self._lock:
                if self._latest is not None:
                    break
            time.sleep(0.02)
        else:
            self._running = False
            self._ser.close()
            raise RuntimeError(f"{self}: no data received after START — check port and wiring")

        logger.info(f"{self} connected on {self.config.port}")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass  # ESP32 configures motors at boot

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._lock:
            if self._latest is None:
                raise DeviceNotConnectedError(f"{self}: no data received yet")
            age_ms = (time.perf_counter() - self._latest_time) * 1e3
            if age_ms > self.config.timeout_ms:
                raise DeviceNotConnectedError(f"{self}: data timeout ({age_ms:.0f} ms)")
            return dict(self._latest)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._ser.write(bytes([0xAA, 0x00]))  # STOP — ESP32 zeros pushback and goes idle
        self._ser.flush()
        time.sleep(0.05)
        self._running = False
        self._ser.close()
        logger.info(f"{self} disconnected.")

    # ── Background reader ─────────────────────────────────────────────────────

    _KEEPALIVE_INTERVAL = 15.0  # seconds; prevents ESP32-S3 USB JTAG watchdog (~45s timeout)

    def _reader(self) -> None:
        buf = bytearray()
        txt = bytearray()  # accumulates ESP32 printf text
        last_keepalive = time.perf_counter()
        while self._running:
            if time.perf_counter() - last_keepalive > self._KEEPALIVE_INTERVAL:
                try:
                    self._ser.write(bytes([0xFF]))  # ignored by firmware, resets USB watchdog
                except Exception:
                    pass
                last_keepalive = time.perf_counter()
            try:
                chunk = self._ser.read(self._ser.in_waiting or 1)
            except Exception as e:
                logger.warning(f"{self} serial read error: {e}")
                break
            if not chunk:
                continue
            buf.extend(chunk)

            # scan for magic header; accumulate printable leading bytes as ESP32 log lines
            while len(buf) >= 2 and buf[:2] != _MAGIC:
                b = buf.pop(0)
                if b == ord('\n'):
                    line = txt.decode(errors="replace").strip()
                    if line:
                        logger.warning(f"[ESP32] {line}")
                    txt.clear()
                elif b != ord('\r'):
                    txt.append(b)

            # parse complete packets
            while len(buf) >= _PKT_LEN:
                if buf[:2] != _MAGIC:
                    buf.pop(0)
                    continue
                payload = bytes(buf[2 : 2 + _PAYLOAD_LEN])
                buf = buf[_PKT_LEN:]

                unpacked = struct.unpack(_PAYLOAD_FMT, payload)
                seq = unpacked[0]
                raw = unpacked[1:]

                with self._lock:
                    if self._last_seq is not None:
                        expected = (self._last_seq + 1) & 0xFF
                        if seq != expected:
                            dropped = (seq - expected) & 0xFF
                            logger.warning(f"[seq] gap: expected {expected}, got {seq} ({dropped} dropped)")
                    self._last_seq = seq
                    self._pkt_count += 1

                gripper_range = self.config.gripper_open_pos - self.config.gripper_closed_pos
                action = {f"{m}.pos": raw[i] for i, m in enumerate(_MOTOR_NAMES) if m != "gripper"}
                action["gripper.pos"] = 1.0 - (raw[6] - self.config.gripper_closed_pos) / gripper_range

                with self._lock:
                    self._latest = action
                    self._latest_time = time.perf_counter()
