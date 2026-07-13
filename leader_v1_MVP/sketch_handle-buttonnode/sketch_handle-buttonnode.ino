// Leader-handle button node — Dynamixel Protocol 2.0 SLAVE on the TTL bus.
//
// Sits on the same half-duplex TTL bus as the XL330 motors and answers to ID 8.
// Two buttons are packed into one readable register; the bus MASTER (base bridge →
// PC lerobot) polls it. Dynamixel is master-polled, so the node cannot push
// unsolicited — it responds only when READ/PINGed.
//
// Wiring:
//   D5 (GPIO6)  — single-wire half-duplex TTL DATA (TX+RX), same as the base bridge
//   D7 (GPIO44) — button 1, INPUT_PULLUP, pressed = LOW (GND -> button -> D7)
//   D9 (GPIO8)  — button 2, INPUT_PULLUP, pressed = LOW (GND -> button -> D9)
//                 (moved off D8/GPIO7 — some PCBs pull D8 to GND permanently)
//   VDD/GND     — tap the motor-chain power at the handle. GND MUST be common with
//                 the bus AND with the node's USB supply if powered separately.
//
// Read from the master (Protocol 2.0):  READ  id=8  addr=100  len=1
//   bit 0 = button 1 (D7), bit 1 = button 2 (D9).  1 = pressed.
//   0 = none, 1 = btn1, 2 = btn2, 3 = both.
//
// LED DIAGNOSTIC (onboard LED, active-low):
//   • solid ON while being polled (a READ of addr 100 arrived in the last 300 ms)
//   • slow heartbeat blink (~50 ms/s) when alive but NOT being read
//   • fully dark  → firmware not running / not powered
//
// USB mode is irrelevant here — this firmware never uses USB for data, only UART1,
// so USB-Serial/JTAG mode (with its ~60 s reset) is fine for debugging.
// Requires the "Dynamixel2Arduino" library (Library Manager).

#include <Dynamixel2Arduino.h>
#include "driver/uart.h"

using namespace DYNAMIXEL;

#define DXL_PIN       6           // GPIO6 = D5, single-wire half-duplex TTL
#define DXL_BAUD      1000000     // leader bus runs at 1 Mbps
#define DXL_ID        8
#define DXL_MODEL_NUM 0x5005      // arbitrary custom-node model number
#define DXL_FW_VER    1

// Custom control-table register the master reads for the button bits.
#define ADDR_BUTTONS  100         // 1 byte: bit0 = button 1 (D7), bit1 = button 2 (D8)

// Buttons: pin + bit position in the register. Both active-low (GND -> button -> pin).
#define NUM_BUTTONS   2
static const uint8_t BTN_PIN[NUM_BUTTONS] = {44, 8};   // D7 = GPIO44, D9 = GPIO8
static const uint8_t BTN_BIT[NUM_BUTTONS] = {0, 1};

HardwareSerial DxlSerial(1);
SerialPortHandler dxl_port(DxlSerial);   // dir_pin = -1: HW RS485 handles direction
Slave dxl(dxl_port, DXL_MODEL_NUM);

uint8_t buttons = 0;                      // registered at ADDR_BUTTONS; master reads this

// Per-button debounce state.
static const uint32_t DEBOUNCE_MS = 20;
static uint8_t  lastRaw[NUM_BUTTONS]    = {HIGH, HIGH};
static uint8_t  stable[NUM_BUTTONS]     = {HIGH, HIGH};
static uint32_t lastChange[NUM_BUTTONS] = {0, 0};

// Diagnostics: last master READ time + total read count (for the USB status print).
static uint32_t lastReadMs = 0;
static uint32_t readCount = 0;

static void onRead(uint16_t item_addr, uint8_t &err, void *arg) {
    if (item_addr == ADDR_BUTTONS) {
        lastReadMs = millis();
        readCount++;
    }
}

void setup() {
    Serial.begin(115200);                 // USB debug (separate from the UART1 bus)
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);      // off (active-low)
    for (int i = 0; i < NUM_BUTTONS; i++) pinMode(BTN_PIN[i], INPUT_PULLUP);

    // Begin UART1 ONCE on GPIO6 with hardware RS485 half-duplex (proven bridge init),
    // then mark the library's port handler open. Do NOT call dxl_port.begin() — it
    // re-begins UART1 on default pins, and begin-twice crashes the UART.
    DxlSerial.begin(DXL_BAUD, SERIAL_8N1, DXL_PIN, DXL_PIN);
    uart_set_mode(UART_NUM_1, UART_MODE_RS485_HALF_DUPLEX);
    dxl_port.setOpenState(true);

    dxl.setPortProtocolVersion(2.0);
    dxl.setFirmwareVersion(DXL_FW_VER);
    dxl.setID(DXL_ID);
    dxl.addControlItem(ADDR_BUTTONS, buttons);       // stores &buttons; READ returns live value
    dxl.setReadCallbackFunc(onRead);
}

void loop() {
    // Debounce each button (GND -> button -> pin, so LOW = pressed) and pack the bits.
    uint32_t now = millis();
    uint8_t bits = 0;
    for (int i = 0; i < NUM_BUTTONS; i++) {
        uint8_t raw = digitalRead(BTN_PIN[i]);
        if (raw != lastRaw[i]) { lastChange[i] = now; lastRaw[i] = raw; }
        if (now - lastChange[i] > DEBOUNCE_MS && raw != stable[i]) {
            stable[i] = raw;
        }
        if (stable[i] == LOW) bits |= (1 << BTN_BIT[i]);
    }
    buttons = bits;

    // Respond to any master instruction addressed to us (PING, or READ of ADDR_BUTTONS).
    dxl.processPacket();

    // LED diagnostic (active-low: LOW = lit). May do nothing on boards with an RGB LED.
    if (now - lastReadMs < 300) {
        digitalWrite(LED_BUILTIN, LOW);                            // being polled → solid on
    } else {
        digitalWrite(LED_BUILTIN, (now % 1000) < 50 ? LOW : HIGH); // alive → heartbeat
    }

    // USB status print once/sec — LED-independent view of what's happening.
    // GUARD: only print when the USB TX buffer has room. In USB-Serial/JTAG mode
    // Serial.printf() BLOCKS if the buffer fills and no host is draining it (e.g. no
    // Serial Monitor open), which would freeze the loop and stop bus responses.
    static uint32_t lastPrint = 0;
    if (now - lastPrint >= 1000) {
        lastPrint = now;
        if (Serial.availableForWrite() >= 48) {
            Serial.printf("[node] up=%lus  master_reads=%lu  btn1=%s  btn2=%s\n",
                          now / 1000, readCount,
                          (buttons & 0b01) ? "DOWN" : "up",
                          (buttons & 0b10) ? "DOWN" : "up");
        }
    }
}
