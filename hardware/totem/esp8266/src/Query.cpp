#include "Query.h"

#include <Credentials.h>

unsigned int Query::last_ok_read = 0;
unsigned int Query::max_down_time_s = 60 * 5;

WiFiClientSecure Query::espClient;
HTTPClient Query::client;

bool Query::read_last(int *res, bool backup) {
    espClient.setInsecure(); // TODO: Puxar fingerprint do endpoint
    client.begin(espClient, backup ? BACKUP_API_ADDRESS : API_ADDRESS);

    client.addHeader("Device-Name", "Totem-IC2");
    client.addHeader("Device-FreeHeap", String(ESP.getFreeHeap()));

    int httpCode = client.GET();
    String payload;

    Serial.println("HTTP code: " + String(httpCode));

    if (httpCode >= 200 && httpCode < 300) {
        payload = client.getString();
        Serial.println("resposta:");
        Serial.println(payload);

        last_ok_read = millis();
    } else {
        Serial.printf("erro no GET: %s\n", client.errorToString(httpCode).c_str());
        client.end();

        if (!backup) {
            Serial.println("tentando server backup...");
            return read_last(res, true);
        }

        return false;
    }

    client.end();

    *res = payload.toInt();
    return true;
}

bool Query::is_updated() {
    return millis() - last_ok_read <= max_down_time_s * 1000;
}
