#include <Arduino.h>
#include <OTA.h>

#include "Display.h"
#include "Wifi.h"
#include "Query.h"

unsigned int read_interval_s = 10;
unsigned long last_read = 0;

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);
    Serial.begin(115200);

    Serial.println(String("Current Version: ") + String(VERSION));

    Display::setup_pins();
    Wifi::setup();
    OTA::setup();

    OTA::check();

    last_read = millis() - read_interval_s * 1000;
}

void loop() {
    if (millis() - last_read >= read_interval_s * 1000) {
        int count = 0;
        if (Query::read_last(&count)) {
            if (count > 16 || count < 0) Display::disable();
            else Display::write(count);
        }

        last_read = millis();
    }

    if (!Query::is_updated()) Display::disable();
    if (OTA::should_check()) OTA::check();
}
