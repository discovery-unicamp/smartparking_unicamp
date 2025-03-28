#ifndef TOTEM_INFLUX_H
#define TOTEM_INFLUX_H

#include "Arduino.h"
#include "Credentials.h"

#include <ESP8266WiFi.h>
#include <ArduinoHttpClient.h>

class Influx {

private:
    static WiFiClient espClient;
    static HttpClient client;

    static unsigned int last_ok_read;
    static unsigned int max_down_time_s;

    typedef struct QueryResponse {
        int statusCode;
        String response;
    } QueryResponse;

    static QueryResponse query(String q);

public:

    static int read_last();

    static bool is_updated();

};


#endif
