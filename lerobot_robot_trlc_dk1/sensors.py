"""
External sensor readers for the TRLC-DK1.
"""

from __future__ import annotations

import logging
import threading

import serial

logger = logging.getLogger(__name__)


class ACS712Sensor:
    """
    Reads supply current from an ACS712 sensor connected via an Arduino Nano over USB serial.

    The Arduino sketch (examples/arduino/acs712_current_sensor.ino) samples the ACS712
    analog output and sends one ASCII float per line (current in Amperes) at ~100 Hz::

        1.2345\\n

    A background thread continuously reads the latest value so that ``get_current()``
    never blocks the observation loop.

    Args:
        port:      Serial port the Arduino Nano is on (e.g. ``"/dev/ttyUSB0"``).
        baud_rate: Must match the sketch (default 115200).
    """

    def __init__(self, port: str, baud_rate: int = 115200) -> None:
        self._port = port
        self._baud_rate = baud_rate
        self._serial: serial.Serial | None = None
        self._current_a: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def is_connected(self) -> bool:
        return self._running and self._serial is not None and self._serial.is_open

    def connect(self) -> None:
        if self.is_connected:
            return
        self._serial = serial.Serial(self._port, self._baud_rate, timeout=0.1)
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="acs712-reader"
        )
        self._thread.start()
        logger.info("ACS712Sensor connected on %s at %d baud", self._port, self._baud_rate)

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        logger.info("ACS712Sensor disconnected")

    def get_current(self) -> float:
        """Return the latest measured supply current in Amperes."""
        with self._lock:
            return self._current_a

    def _read_loop(self) -> None:
        while self._running:
            try:
                line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    value = float(line)
                    with self._lock:
                        self._current_a = value
            except ValueError:
                pass  # malformed line — keep last value
            except serial.SerialException as e:
                logger.warning("ACS712Sensor serial error: %s", e)
                break
