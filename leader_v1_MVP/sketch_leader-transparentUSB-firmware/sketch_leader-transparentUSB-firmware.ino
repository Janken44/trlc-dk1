// Transparent USB ↔ Dynamixel half-duplex TTL bridge
//
// Drop-in replacement for the Waveshare USB-to-half-duplex-TTL board.
// The ESP32-S3 passes bytes between USB CDC and the Dynamixel bus with no
// protocol awareness — the host Dynamixel SDK sees a normal serial port.
//
// Wiring: single wire on DXL_PIN for both TX and RX.
//         uart_set_mode RS485_HALF_DUPLEX suppresses TX echo in silicon.
//
// IMPORTANT: flash with Arduino IDE → Tools → USB Mode → "USB-OTG (TinyUSB)"
//            USB Serial/JTAG mode has a ~45 s JTAG watchdog that resets the chip.
//
// Baud rate: relayed from the USB-CDC host baud (SET_LINE_CODING) to UART1.
//            A dedicated USB-UART chip (Waveshare/CH343P) does this in silicon;
//            on the ESP32 the USB-CDC and UART1 are separate peripherals, so we
//            relay the baud ourselves via the CDC line-coding event (below).
//            lerobot scans 57600 (factory) .. 1 000 000 during motor ID setup.

#include "driver/uart.h"

#define DXL_PIN      6         // GPIO6 = D5, single-wire half-duplex
#define DEFAULT_BAUD 1000000   // used until the host sets a CDC line coding

HardwareSerial DxlSerial(1);
static uint32_t activeBaud = 0;
static volatile uint32_t requestedBaud = DEFAULT_BAUD;

static void setUartBaud(uint32_t baud) {
    if (baud == 0 || baud == activeBaud) return;
    activeBaud = baud;
    DxlSerial.begin(baud, SERIAL_8N1, DXL_PIN, DXL_PIN);
    uart_set_mode(UART_NUM_1, UART_MODE_RS485_HALF_DUPLEX);
}

// Fires whenever the host changes the CDC line coding (baud) — the event a CH343P
// handles in hardware. We relay the new baud to UART1. The LED toggles on every
// such event so you can VISUALLY confirm the relay fires during a lerobot baud scan;
// if it never blinks, SET_LINE_CODING isn't reaching the CDC (an upstream problem
// no relay code can fix).
static void onUsbCdcEvent(void *arg, esp_event_base_t base, int32_t id, void *event_data) {
    if (id == ARDUINO_USB_CDC_LINE_CODING_EVENT) {
        auto *d = (arduino_usb_cdc_event_data_t *) event_data;
        requestedBaud = d->line_coding.bit_rate;
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    }
}

void setup() {
    // GPIO6 (D5) is a plain GPIO — no U0TXD eviction needed (unlike D6/D7 = GPIO43/44,
    // the UART0 debug pins, where RS485 half-duplex on UART1 fails).
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);  // LED off (active-low); only blinks on baud change
    Serial.onEvent(onUsbCdcEvent);   // relay host baud changes to UART1
    Serial.begin(0);                 // baud rate is irrelevant for USB CDC
    setUartBaud(DEFAULT_BAUD);
}

void loop() {
    // Apply the most recent host-requested baud (set by the line-coding event).
    setUartBaud(requestedBaud);

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
