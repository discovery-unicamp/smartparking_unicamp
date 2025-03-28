#include <Arduino.h>

#include "Display.h"
#include "Wifi.h"
#include "HttpClient.h"
#include "Influx.h"
//#include "OTA.h"

unsigned int check_interval_s = 30;
unsigned long last_check = 0;

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);
    Serial.begin(115200);

    Display::setup_pins();
    Wifi::setup();
//    OTA::setup();

    last_check = millis() - check_interval_s * 1000;
}

void loop() {
    if (millis() - last_check >= check_interval_s * 1000) {
        int count = Influx::read_last();

        if (count > 16 || count < 0) Display::disable();
        else Display::write(count);

        last_check = millis();
    }

    if (!Influx::is_updated()) Display::disable();
//    OTA::check();
}