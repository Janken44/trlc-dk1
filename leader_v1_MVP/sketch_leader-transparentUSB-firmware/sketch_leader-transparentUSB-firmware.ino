// Transparent USB ↔ Dynamixel half-duplex TTL bridge
//
// Drop-in replacement for the Waveshare USB-to-half-duplex-TTL board.
// The ESP32-S3 passes bytes between USB CDC and the Dynamixel bus with no
// protocol awareness — the host Dynamixel SDK sees a normal serial port.
//
// Wiring: single wire on GPIO7 (D8) for both TX and RX.
//         uart_set_mode RS485_HALF_DUPLEX suppresses TX echo in silicon.
//
// IMPORTANT: flash with Arduino IDE → Tools → USB Mode → "USB-OTG (TinyUSB)"
//            USB Serial/JTAG mode has a ~45 s JTAG watchdog that resets the chip.
//
// Baud rate: auto-detected from USB CDC line coding set by the host.
//            lerobot DynamixelMotorsBus default = 1 000 000 baud.

#include "driver/uart.h"

#define DXL_PIN      7         // GPIO7 = D8, single-wire half-duplex
#define DEFAULT_BAUD 1000000   // used until host sets CDC line coding

HardwareSerial DxlSerial(1);
static uint32_t activeBaud = 0;

static void setUartBaud(uint32_t baud) {
    if (baud == 0 || baud == activeBaud) return;
    activeBaud = baud;
    DxlSerial.begin(baud, SERIAL_8N1, DXL_PIN, DXL_PIN);
    uart_set_mode(UART_NUM_1, UART_MODE_RS485_HALF_DUPLEX);
}

void setup() {
    Serial.begin(0);          // baud rate is irrelevant for USB CDC
    setUartBaud(DEFAULT_BAUD);
}

void loop() {
    // keep UART baud in sync with whatever the host set on the CDC port
    uint32_t hostBaud = Serial.baudRate();
    if (hostBaud > 0) setUartBaud(hostBaud);

    // USB CDC → Dynamixel bus
    int n = Serial.available();
    if (n > 0) {
        uint8_t buf[256];
        int count = 0;
        while (count < n && count < (int)sizeof(buf) && Serial.available()) {
            buf[count++] = Serial.read();
        }
        if (count > 0) {
            DxlSerial.write(buf, count);
            DxlSerial.flush();  // wait for TX before motors start responding
        }
    }

    // Dynamixel bus → USB CDC
    n = DxlSerial.available();
    if (n > 0) {
        uint8_t buf[256];
        int count = 0;
        while (count < n && count < (int)sizeof(buf) && DxlSerial.available()) {
            buf[count++] = DxlSerial.read();
        }
        if (count > 0) {
            Serial.write(buf, count);
        }
    }

    yield();  // let TinyUSB process USB transactions between iterations
}
