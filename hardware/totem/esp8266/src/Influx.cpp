#include "Influx.h"

WiFiClient Influx::espClient;
HttpClient Influx::client = HttpClient(espClient, INFLUXDB_ADDRESS, INFLUXDB_PORT);

unsigned int Influx::last_ok_read = 0;
unsigned int Influx::max_down_time_s = 30 * 5;

Influx::QueryResponse Influx::query(String query) {

    Serial.println("Sending query: " + query);

    String api_string = "/api/v2/query?org=" + String(INFLUXDB_ORG);
    String auth_string = "Token " + String(INFLUXDB_TOKEN);

    client.beginRequest();
    client.post(api_string);
    client.sendHeader("Accept", "application/csv");
    client.sendHeader("Content-Type", "application/vnd.flux");
    client.sendHeader("Content-Length", query.length());
    client.sendHeader("Authorization", auth_string);
    client.endRequest();

    client.print(query);

    int statusCode = client.responseStatusCode();
    if (statusCode == 200) Serial.println("Successfully asked");
    else Serial.println("Something unexpected happened: STATUS CODE " + String(statusCode));

    String response = "";
    while (client.available()) {
        char c = client.read();
        response += c;
    }

    return QueryResponse{
            .statusCode = statusCode,
            .response = response
    };


}

int Influx::read_last() {
    String q =
            "from(bucket: \"" + String(INFLUXDB_BUCKET) + "\")" +
            "   |> range(start: -60d)" +
            "   |> filter(fn: (r) => r[\"_measurement\"] == \"detected_cars\")" +
            "   |> filter(fn: (r) => r[\"_field\"] == \"cars\")" +
            "   |> keep(columns: [\"_value\"])" +
            "   |> last()";

    Influx::QueryResponse qr = query(q);
    Serial.println("Response: " + qr.response);

    if (qr.statusCode == 200) last_ok_read = millis();

    int start = qr.response.lastIndexOf(",") + 1;
    int end = qr.response.lastIndexOf("\n");
    int value = qr.response.substring(start, end).toInt();

    Serial.println("Last _value: " + String(value));
    return value;
}

bool Influx::is_updated() {
    return millis() - last_ok_read <= max_down_time_s * 1000;
}
