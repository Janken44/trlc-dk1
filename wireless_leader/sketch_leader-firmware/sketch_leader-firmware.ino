#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── BLE ───────────────────────────────────────────────────────────────────────
#define SERVICE_UUID        "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
#define POSITIONS_CHAR_UUID "a1b2c3d4-e5f6-7890-abcd-ef1234567891"

// ── UART pins (full-duplex to Waveshare H2 header) ───────────────────────────
#define DXL_RX_PIN  44      // GPIO44 = D7 → Waveshare RX (straight: RX-RX)
#define DXL_TX_PIN   7      // GPIO7  = D8 → Waveshare TX (straight: TX-TX)
#define DXL_BAUD    1000000 // LeRobot configures XL330 to 1 Mbps

// ── Dynamixel motor IDs ───────────────────────────────────────────────────────
#define N_MOTORS    7       // IDs 1–7 (joints 1–6 + gripper)
#define ID_JOINT3   3
#define ID_JOINT4   4
#define ID_GRIPPER  7

// ── Control table addresses (XL330, Protocol 2.0) ────────────────────────────
#define ADDR_CURRENT_LIMIT   38
#define ADDR_OPERATING_MODE  11
#define ADDR_TORQUE_ENABLE   64
#define ADDR_GOAL_CURRENT    102
#define ADDR_GOAL_POSITION   116
#define ADDR_PRESENT_POS     132

#define MODE_CURRENT           0
#define MODE_CURRENT_POSITION  5

// ── Gripper config ────────────────────────────────────────────────────────────
#define GRIPPER_OPEN_POS     2280   // ticks, matches leader.py default
#define GRIPPER_CURRENT_LIMIT 100   // mA

// ── Pushback spring ───────────────────────────────────────────────────────────
#define JOINT3_UPPER_RAD  2.0f
#define JOINT4_LOWER_RAD -1.1f
#define PUSHBACK_K        100.0f   // mA / rad
#define PUSHBACK_MAX_MA   100

// ── Packet sizes ──────────────────────────────────────────────────────────────
// SyncRead: 4(hdr)+1(id)+2(len)+1(inst)+2(addr)+2(dlen)+N(ids)+2(crc)
#define SYNCREAD_PKT_LEN  (N_MOTORS + 14)
// Status per motor: 4(hdr)+1(id)+2(len)+1(0x55)+1(err)+4(data)+2(crc)
#define STATUS_PKT_LEN    15
// SyncWrite Goal_Current for 2 joints: 4+1+2+1+2+2+2*(1+2)+2 = 20
#define SYNCWRITE_CUR_LEN 20

HardwareSerial DxlSerial(1);

BLEServer*         pServer        = nullptr;
BLECharacteristic* pPositionsChar = nullptr;
bool               connected      = false;

struct __attribute__((packed)) PositionPacket {
    uint8_t seq;
    float   positions[N_MOTORS];  // joints 1–6: radians, gripper: raw ticks
};

// ── CRC-16 (Dynamixel Protocol 2.0) ──────────────────────────────────────────
static const uint16_t CRC_TABLE[256] = {
    0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011,
    0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027, 0x0022,
    0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072,
    0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041,
    0x80C3, 0x00C6, 0x00CC, 0x80C9, 0x00D8, 0x80DD, 0x80D7, 0x00D2,
    0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1,
    0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1,
    0x8093, 0x0096, 0x009C, 0x8099, 0x0088, 0x808D, 0x8087, 0x0082,
    0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192,
    0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1,
    0x01E0, 0x81E5, 0x81EF, 0x01EA, 0x81FB, 0x01FE, 0x01F4, 0x81F1,
    0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2,
    0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151,
    0x8173, 0x0176, 0x017C, 0x8179, 0x0168, 0x816D, 0x8167, 0x0162,
    0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
    0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101,
    0x8303, 0x0306, 0x030C, 0x8309, 0x0318, 0x831D, 0x8317, 0x0312,
    0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321,
    0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371,
    0x8353, 0x0356, 0x035C, 0x8359, 0x0348, 0x834D, 0x8347, 0x0342,
    0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1,
    0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2,
    0x83A3, 0x03A6, 0x03AC, 0x83A9, 0x03B8, 0x83BD, 0x83B7, 0x03B2,
    0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381,
    0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291,
    0x82B3, 0x02B6, 0x02BC, 0x82B9, 0x02A8, 0x82AD, 0x82A7, 0x02A2,
    0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2,
    0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1,
    0x8243, 0x0246, 0x024C, 0x8249, 0x0258, 0x825D, 0x8257, 0x0252,
    0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261,
    0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231,
    0x8213, 0x0216, 0x021C, 0x8219, 0x0208, 0x820D, 0x8207, 0x0202
};

uint16_t computeCRC(uint8_t *data, uint16_t len) {
    uint16_t crc = 0;
    for (uint16_t j = 0; j < len; j++) {
        uint16_t i = ((crc >> 8) ^ data[j]) & 0xFF;
        crc = (crc << 8) ^ CRC_TABLE[i];
    }
    return crc;
}

// ── Low-level write to single motor (no response read) ────────────────────────
void writeReg(uint8_t id, uint16_t addr, uint8_t *data, uint8_t dlen) {
    // Length = INST(1) + ADDR(2) + DATA(dlen) + CRC(2)
    uint16_t pkt_len = dlen + 5;
    uint8_t  buf[20];
    buf[0] = 0xFF; buf[1] = 0xFF; buf[2] = 0xFD; buf[3] = 0x00;
    buf[4] = id;
    buf[5] = pkt_len & 0xFF; buf[6] = (pkt_len >> 8) & 0xFF;
    buf[7] = 0x03;  // Write instruction
    buf[8] = addr & 0xFF; buf[9] = (addr >> 8) & 0xFF;
    for (int i = 0; i < dlen; i++) buf[10 + i] = data[i];
    uint16_t crc = computeCRC(buf, 10 + dlen);
    buf[10 + dlen]     = crc & 0xFF;
    buf[11 + dlen]     = (crc >> 8) & 0xFF;
    DxlSerial.write(buf, 12 + dlen);
    DxlSerial.flush();
    delay(5);
    while (DxlSerial.available()) DxlSerial.read();  // discard status packet
}

void write1(uint8_t id, uint16_t addr, uint8_t val) {
    writeReg(id, addr, &val, 1);
}

void write2(uint8_t id, uint16_t addr, int16_t val) {
    uint8_t d[2] = { (uint8_t)(val & 0xFF), (uint8_t)((val >> 8) & 0xFF) };
    writeReg(id, addr, d, 2);
}

void write4(uint8_t id, uint16_t addr, int32_t val) {
    uint8_t d[4] = {
        (uint8_t)(val & 0xFF), (uint8_t)((val >> 8) & 0xFF),
        (uint8_t)((val >> 16) & 0xFF), (uint8_t)((val >> 24) & 0xFF)
    };
    writeReg(id, addr, d, 4);
}

// ── Configure operating modes at startup ──────────────────────────────────────
void configureDynamixel() {
    // Joints 3 & 4: current mode for pushback spring
    for (uint8_t id : {ID_JOINT3, ID_JOINT4}) {
        write1(id, ADDR_TORQUE_ENABLE, 0);
        write1(id, ADDR_OPERATING_MODE, MODE_CURRENT);
        write1(id, ADDR_TORQUE_ENABLE, 1);
        write2(id, ADDR_GOAL_CURRENT, 0);
    }

    // Gripper: current-based position mode, holds open against light resistance
    write1(ID_GRIPPER, ADDR_TORQUE_ENABLE, 0);
    write1(ID_GRIPPER, ADDR_OPERATING_MODE, MODE_CURRENT_POSITION);
    write2(ID_GRIPPER, ADDR_CURRENT_LIMIT, GRIPPER_CURRENT_LIMIT);
    write1(ID_GRIPPER, ADDR_TORQUE_ENABLE, 1);
    write4(ID_GRIPPER, ADDR_GOAL_POSITION, GRIPPER_OPEN_POS);

    Serial.println("Dynamixel configured");
}

// ── SyncWrite Goal_Current to joints 3 & 4 ───────────────────────────────────
void syncWriteCurrent(int16_t cur3, int16_t cur4) {
    // Params: ADDR(2) + DLEN(2) + [ID3 + 2 bytes] + [ID4 + 2 bytes] = 10
    // Length = 10 + 3 = 13
    uint8_t buf[SYNCWRITE_CUR_LEN];
    buf[0] = 0xFF; buf[1] = 0xFF; buf[2] = 0xFD; buf[3] = 0x00;
    buf[4] = 0xFE;   // broadcast
    buf[5] = 13; buf[6] = 0x00;
    buf[7] = 0x83;   // SyncWrite instruction
    buf[8]  = ADDR_GOAL_CURRENT & 0xFF;
    buf[9]  = (ADDR_GOAL_CURRENT >> 8) & 0xFF;
    buf[10] = 0x02; buf[11] = 0x00;  // data length per motor = 2
    buf[12] = ID_JOINT3;
    buf[13] = cur3 & 0xFF; buf[14] = (cur3 >> 8) & 0xFF;
    buf[15] = ID_JOINT4;
    buf[16] = cur4 & 0xFF; buf[17] = (cur4 >> 8) & 0xFF;
    uint16_t crc = computeCRC(buf, 18);
    buf[18] = crc & 0xFF; buf[19] = (crc >> 8) & 0xFF;
    DxlSerial.write(buf, SYNCWRITE_CUR_LEN);
    DxlSerial.flush();
    // SyncWrite has no response packet
}

// ── SyncRead Present_Position from all motors ─────────────────────────────────
bool readPositions(float *positions) {
    static uint8_t pkt[SYNCREAD_PKT_LEN];
    static bool    built = false;
    if (!built) {
        pkt[0] = 0xFF; pkt[1] = 0xFF; pkt[2] = 0xFD; pkt[3] = 0x00;
        pkt[4] = 0xFE;
        uint16_t length = N_MOTORS + 7;
        pkt[5] = length & 0xFF; pkt[6] = (length >> 8) & 0xFF;
        pkt[7] = 0x82;
        pkt[8]  = ADDR_PRESENT_POS & 0xFF;
        pkt[9]  = (ADDR_PRESENT_POS >> 8) & 0xFF;
        pkt[10] = 4; pkt[11] = 0x00;
        for (int i = 0; i < N_MOTORS; i++) pkt[12 + i] = i + 1;  // IDs 1–7
        uint16_t crc = computeCRC(pkt, SYNCREAD_PKT_LEN - 2);
        pkt[SYNCREAD_PKT_LEN - 2] = crc & 0xFF;
        pkt[SYNCREAD_PKT_LEN - 1] = (crc >> 8) & 0xFF;
        built = true;
    }

    while (DxlSerial.available()) DxlSerial.read();  // flush stale bytes

    DxlSerial.write(pkt, SYNCREAD_PKT_LEN);
    DxlSerial.flush();

    uint8_t  buf[STATUS_PKT_LEN];
    uint32_t deadline = millis() + 50;

    for (int m = 0; m < N_MOTORS; m++) {
        int rx = 0;
        while (rx < STATUS_PKT_LEN && millis() < deadline) {
            if (DxlSerial.available()) buf[rx++] = DxlSerial.read();
        }
        if (rx < STATUS_PKT_LEN) {
            Serial.printf("Motor %d timeout: got %d bytes\n", m + 1, rx);
            return false;
        }
        if (buf[0] != 0xFF || buf[1] != 0xFF || buf[2] != 0xFD ||
            buf[3] != 0x00 || buf[7] != 0x55) {
            Serial.printf("Motor %d bad header:", m + 1);
            for (int i = 0; i < STATUS_PKT_LEN; i++) Serial.printf(" %02X", buf[i]);
            Serial.println();
            return false;
        }

        uint8_t motor_id = buf[4];
        if (motor_id < 1 || motor_id > N_MOTORS) return false;

        int32_t ticks = (int32_t)(buf[9] | (buf[10] << 8) | (buf[11] << 16) | (buf[12] << 24));
        int     idx   = motor_id - 1;

        if (idx < N_MOTORS - 1) {
            positions[idx] = (ticks / 4096.0f) * 2.0f * M_PI - M_PI;
        } else {
            positions[idx] = (float)ticks;  // gripper: raw ticks, PC normalises
        }
    }
    return true;
}

// ── Pushback spring ───────────────────────────────────────────────────────────
void applyPushback(float *positions) {
    auto spring = [](float pos, float lo, float hi) -> int16_t {
        float mA = 0.0f;
        if (pos < lo) mA =  PUSHBACK_K * (lo - pos);
        if (pos > hi) mA = -PUSHBACK_K * (pos - hi);
        if (mA >  PUSHBACK_MAX_MA) mA =  PUSHBACK_MAX_MA;
        if (mA < -PUSHBACK_MAX_MA) mA = -PUSHBACK_MAX_MA;
        return (int16_t)mA;
    };
    int16_t cur3 = spring(positions[2], -1e9f, JOINT3_UPPER_RAD);
    int16_t cur4 = spring(positions[3], JOINT4_LOWER_RAD,  1e9f);
    syncWriteCurrent(cur3, cur4);
}

// ── BLE callbacks ─────────────────────────────────────────────────────────────
class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer*) override {
        connected = true;
        Serial.println("BLE client connected");
    }
    void onDisconnect(BLEServer*) override {
        connected = false;
        Serial.println("BLE client disconnected — re-advertising");
        BLEDevice::startAdvertising();
    }
};

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    DxlSerial.begin(DXL_BAUD, SERIAL_8N1, DXL_RX_PIN, DXL_TX_PIN);
    delay(100);
    configureDynamixel();

    BLEDevice::init("DK1-Leader");
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());
    BLEService* pService = pServer->createService(SERVICE_UUID);
    pPositionsChar = pService->createCharacteristic(
        POSITIONS_CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    pPositionsChar->addDescriptor(new BLE2902());
    pService->start();

    BLEAdvertising* pAdv = BLEDevice::getAdvertising();
    pAdv->addServiceUUID(SERVICE_UUID);
    pAdv->setScanResponse(true);
    BLEDevice::startAdvertising();

    Serial.println("DK1-Leader ready");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    static uint8_t       seq      = 0;
    static unsigned long lastSend = 0;

    if (millis() - lastSend < 10) return;  // 100 Hz cap
    lastSend = millis();

    float positions[N_MOTORS] = {};
    if (!readPositions(positions)) return;

    applyPushback(positions);

    if (connected) {
        PositionPacket pkt;
        pkt.seq = seq++;
        memcpy(pkt.positions, positions, sizeof(positions));
        pPositionsChar->setValue((uint8_t*)&pkt, sizeof(pkt));
        pPositionsChar->notify();
    }
}
