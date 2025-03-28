#include "Wifi.h"

void Wifi::setup() {
    delay(10);

    Serial.print("Connecting to ");
    Serial.println(WIFI_ID);

    WiFi.begin(WIFI_ID, WIFI_PASSWORD);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println("");
    Serial.println("WiFi conectado");
    Serial.println("Endereco de IP: ");
    Serial.println(WiFi.localIP());
}