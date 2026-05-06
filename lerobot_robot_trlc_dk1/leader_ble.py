import asyncio
import struct
import threading
import time
import logging
from dataclasses import dataclass

from bleak import BleakClient, BleakScanner

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

_SERVICE_UUID   = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_POSITIONS_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567891"

_MOTOR_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
_PACKET_FMT  = "<Bfffffff"  # seq(u8) + 7 × float32


@TeleoperatorConfig.register_subclass("dk1_leader_ble")
@dataclass
class DK1LeaderBLEConfig(TeleoperatorConfig):
    device_name: str = "DK1-Leader"
    gripper_open_pos: int = 2280
    gripper_closed_pos: int = 1670
    timeout_ms: float = 100.0


class DK1LeaderBLE(Teleoperator):
    config_class = DK1LeaderBLEConfig
    name = "dk1_leader_ble"

    def __init__(self, config: DK1LeaderBLEConfig):
        super().__init__(config)
        self.config = config
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._client: BleakClient | None = None
        self._lock          = threading.Lock()
        self._latest: dict | None = None
        self._latest_time   = 0.0
        self._connected     = False

    # ── LeRobot interface ─────────────────────────────────────────────────────

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in _MOTOR_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None and self._client.is_connected

    def connect(self, calibrate: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        future = asyncio.run_coroutine_threadsafe(self._async_connect(), self._loop)
        future.result(timeout=30)
        logger.info(f"{self} connected.")

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
                raise DeviceNotConnectedError(f"{self}: no BLE data received yet")
            age_ms = (time.perf_counter() - self._latest_time) * 1e3
            if age_ms > self.config.timeout_ms:
                raise DeviceNotConnectedError(f"{self}: BLE notification timeout ({age_ms:.0f}ms)")
            logger.debug(f"{self} action age: {age_ms:.1f}ms")
            return dict(self._latest)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        future = asyncio.run_coroutine_threadsafe(self._async_disconnect(), self._loop)
        future.result(timeout=10)
        logger.info(f"{self} disconnected.")

    # ── Async internals ───────────────────────────────────────────────────────

    async def _async_connect(self) -> None:
        logger.info(f"Scanning for '{self.config.device_name}'…")
        device = await BleakScanner.find_device_by_name(self.config.device_name, timeout=10.0)
        if device is None:
            raise RuntimeError(f"BLE device '{self.config.device_name}' not found")
        self._client = BleakClient(device, disconnected_callback=self._on_ble_disconnect)
        await self._client.connect()
        await self._client.start_notify(_POSITIONS_UUID, self._on_notification)
        self._connected = True

    async def _async_disconnect(self) -> None:
        self._connected = False
        if self._client and self._client.is_connected:
            await self._client.stop_notify(_POSITIONS_UUID)
            await self._client.disconnect()

    def _on_notification(self, _sender, data: bytearray) -> None:
        if len(data) != struct.calcsize(_PACKET_FMT):
            logger.warning(f"Unexpected BLE packet length {len(data)}")
            return
        unpacked = struct.unpack(_PACKET_FMT, data)
        # unpacked[0] = seq, unpacked[1:8] = positions
        raw = unpacked[1:]
        gripper_range = self.config.gripper_open_pos - self.config.gripper_closed_pos
        action = {f"{m}.pos": raw[i] for i, m in enumerate(_MOTOR_NAMES) if m != "gripper"}
        action["gripper.pos"] = 1.0 - (raw[6] - self.config.gripper_closed_pos) / gripper_range
        with self._lock:
            self._latest = action
            self._latest_time = time.perf_counter()

    def _on_ble_disconnect(self, _client: BleakClient) -> None:
        self._connected = False
        logger.warning(f"{self} BLE disconnected unexpectedly")
