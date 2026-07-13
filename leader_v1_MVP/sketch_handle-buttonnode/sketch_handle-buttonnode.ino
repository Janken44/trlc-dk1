// Leader-handle button node — Dynamixel Protocol 2.0 SLAVE on the TTL bus.
//
// Sits on the same half-duplex TTL bus as the XL330 motors and answers to ID 8.
// A button wired  GND -> button -> D7  is exposed in a readable register; the bus
// MASTER (your base bridge → PC lerobot) polls it. Dynamixel is master-polled, so
// the node cannot push unsolicited — it responds only when READ/PINGed (otherwise
// two devices would drive the bus at once).
//
// Wiring:
//   D5 (GPIO6)  — single-wire half-duplex TTL DATA (TX+RX), same as the base bridge
//   D7 (GPIO44) — button, INPUT_PULLUP, pressed = LOW (GND -> button -> D7)
//   VDD/GND     — tap the motor-chain power at the handle
//
// Read from the master (Protocol 2.0):  READ  id=8  addr=100  len=1
//   returns 1 = pressed, 0 = released.
//
// IMPORTANT: flash in Arduino IDE → Tools → USB Mode → "USB-OTG (TinyUSB)".
// Requires the "Dynamixel2Arduino" library (Library Manager).
//
// NOTE: this is a prototyping stand-in on a spare ESP32-S3. The production handle
// node was speced as a bare tinyAVR; the slave logic ports over directly.

#include <Dynamixel2Arduino.h>
#include "driver/uart.h"

using namespace DYNAMIXEL;

#define DXL_PIN       6           // GPIO6 = D5, single-wire half-duplex TTL
#define BUTTON_PIN    44          // GPIO44 = D7, wired GND -> button -> D7 (active low)
#define DXL_BAUD      1000000     // leader bus runs at 1 Mbps
#define DXL_ID        8
#define DXL_MODEL_NUM 0x5005      // arbitrary custom-node model number
#define DXL_FW_VER    1

// Custom control-table register the master reads for the button state.
#define ADDR_BUTTON   100         // 1 byte: 1 = pressed, 0 = released

HardwareSerial DxlSerial(1);
SerialPortHandler dxl_port(DxlSerial);   // dir_pin defaults to -1: HW RS485 handles direction
Slave dxl(dxl_port, DXL_MODEL_NUM);

uint8_t button_state = 0;                // registered at ADDR_BUTTON; master reads this

// Debounce
static const uint32_t DEBOUNCE_MS = 20;
static uint8_t  lastRaw = HIGH;
static uint8_t  stable  = HIGH;
static uint32_t lastChange = 0;

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);      // off (active-low); lights while pressed
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    // Bring up UART1 as single-wire half-duplex at the bus baud.
    // dxl_port.begin() marks the port handler open but uses UART1's default pins, so we
    // re-begin with our pin and enable hardware RS485 half-duplex afterwards (last wins).
    dxl_port.begin(DXL_BAUD);
    DxlSerial.begin(DXL_BAUD, SERIAL_8N1, DXL_PIN, DXL_PIN);
    uart_set_mode(UART_NUM_1, UART_MODE_RS485_HALF_DUPLEX);

    dxl.setPortProtocolVersion(2.0);
    dxl.setFirmwareVersion(DXL_FW_VER);
    dxl.setID(DXL_ID);
    dxl.addControlItem(ADDR_BUTTON, button_state);   // stores &button_state; READ returns live value
}

void loop() {
    // Debounced read. GND -> button -> pin, so LOW = pressed.
    uint8_t raw = digitalRead(BUTTON_PIN);
    if (raw != lastRaw) { lastChange = millis(); lastRaw = raw; }
    if (millis() - lastChange > DEBOUNCE_MS && raw != stable) {
        stable = raw;
    }
    button_state = (stable == LOW) ? 1 : 0;
    digitalWrite(LED_BUILTIN, button_state ? LOW : HIGH);   // LED on while pressed

    // Respond to any master instruction addressed to us (PING, or READ of ADDR_BUTTON).
    // Non-blocking: services whatever bytes have arrived and replies when a full packet lands.
    dxl.processPacket();
}
