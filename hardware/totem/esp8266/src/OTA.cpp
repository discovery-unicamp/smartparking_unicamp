#include "OTA.h"


WiFiClientSecure OTA::client;
unsigned int OTA::last_check;


void OTA::setup() {
    client.setInsecure(); // TODO: Puxar fingerprint do endpoint
    ESPhttpUpdate.rebootOnUpdate(false);
    ESPhttpUpdate.setAuthorization(OTA_USERNAME, OTA_PASSWORD);
}

void OTA::check() {
    last_check = millis();

    HTTPUpdateResult ret = ESPhttpUpdate.update(
        client,
        OTA_ADDRESS + String("/update?currentVersion=") + String(VERSION)
    );
    client.stop();

    switch (ret) {
        case HTTP_UPDATE_OK:
            Serial.println("Found new version. Rebooting to update...");
            delay(5000);
            EspClass::restart();
            return;
        case HTTP_UPDATE_FAILED:
            Serial.println("Update failed with error.");
            return;
        case HTTP_UPDATE_NO_UPDATES:
            Serial.println("ESP is updated.");
    }
}

bool OTA::should_check() {
    // 15min
    if (millis() - last_check > 60 * 15 * 1000) return true;
    return false;
}
