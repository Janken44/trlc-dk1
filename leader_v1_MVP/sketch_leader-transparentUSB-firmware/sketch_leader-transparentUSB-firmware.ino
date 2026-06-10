// Transparent USB ↔ Dynamixel half-duplex TTL bridge + HID keyboard button
//
// Drop-in replacement for the Waveshare USB-to-half-duplex-TTL board.
// The ESP32-S3 passes bytes between USB CDC and the Dynamixel bus with no
// protocol awareness — the host Dynamixel SDK sees a normal serial port.
//
// Also enumerates as a USB HID keyboard (composite CDC + HID device).
// Button on GPIO9 (D10, active-low) sends RIGHT ARROW — triggers lerobot-record
// episode advance without touching the keyboard.
//
// Wiring: single wire on GPIO7 (D8) for both TX and RX.
//         uart_set_mode RS485_HALF_DUPLEX suppresses TX echo in silicon.
//         Button: GPIO9 (D10) → switch → GND. Internal pullup, no resistor needed.
//
// IMPORTANT: flash with Arduino IDE → Tools → USB Mode → "USB-OTG (TinyUSB)"
//            USB Serial/JTAG mode has a ~45 s JTAG watchdog that resets the chip.
//
// Baud rate: auto-detected from USB CDC line coding set by the host.
//            lerobot DynamixelMotorsBus default = 1 000 000 baud.

#include "USB.h"
#include "USBHIDKeyboard.h"
#include "driver/uart.h"

#define DXL_PIN      7         // GPIO7 = D8, single-wire half-duplex
#define DEFAULT_BAUD 1000000   // used until host sets CDC line coding
#define BUTTON_PIN   9         // GPIO9 = D10, active-low
#define DEBOUNCE_MS  50

HardwareSerial DxlSerial(1);
USBHIDKeyboard Keyboard;

static uint32_t activeBaud      = 0;
static bool     lastRaw         = HIGH;
static bool     buttonState     = HIGH;
static uint32_t lastDebounceTime = 0;

static void setUartBaud(uint32_t baud) {
    if (baud == 0 || baud == activeBaud) return;
    activeBaud = baud;
    DxlSerial.begin(baud, SERIAL_8N1, DXL_PIN, DXL_PIN);
    uart_set_mode(UART_NUM_1, UART_MODE_RS485_HALF_DUPLEX);
}

static void handleButton() {
    bool raw = digitalRead(BUTTON_PIN);
    if (raw != lastRaw) {
        lastDebounceTime = millis();
        lastRaw = raw;
    }
    if (millis() - lastDebounceTime > DEBOUNCE_MS && raw != buttonState) {
        buttonState = raw;
        if (buttonState == LOW) {   // falling edge — button pressed
            Keyboard.press(KEY_RIGHT_ARROW);
            delay(10);
            Keyboard.releaseAll();
        }
    }
}

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    Keyboard.begin();
    Serial.begin(0);           // baud rate is irrelevant for USB CDC
    USB.begin();
    setUartBaud(DEFAULT_BAUD);
}

void loop() {
    handleButton();

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
