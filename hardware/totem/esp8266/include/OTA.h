#ifndef TOTEM_OTA_H
#define TOTEM_OTA_H

#include <ESP8266WiFi.h>
#include <ESP8266httpUpdate.h>
#include "Credentials.h"

class OTA {

private:
    static WiFiClientSecure client;
    static unsigned int last_check;

public:
    static void setup();

    static void check();

    static bool should_check();

};


#endif
