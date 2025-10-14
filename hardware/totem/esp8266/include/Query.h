#ifndef QUERY_H
#define QUERY_H

#include <ESP8266HTTPClient.h>
#include <WiFiClientSecure.h>
#include <Arduino.h>

class Query {

private:
    static WiFiClientSecure espClient;
    static HTTPClient client;

    static unsigned int last_ok_read;
    static unsigned int max_down_time_s;

public:

    static bool read_last(int *res, bool backup = false);

    static bool is_updated();

};



#endif //QUERY_H
