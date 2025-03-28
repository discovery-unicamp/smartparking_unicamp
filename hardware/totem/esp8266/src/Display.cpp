#include "Display.h"

void Display::setup_pins() {
    pinMode(UNITS_A_PIN, OUTPUT);
    pinMode(UNITS_B_PIN, OUTPUT);
    pinMode(UNITS_C_PIN, OUTPUT);
    pinMode(UNITS_D_PIN, OUTPUT);

    pinMode(TENS_A_PIN, OUTPUT);
    pinMode(TENS_B_PIN, OUTPUT);
    pinMode(TENS_C_PIN, OUTPUT);
    pinMode(TENS_D_PIN, OUTPUT);
}

void Display::write(unsigned int number) {
    unsigned int units = number % 10;
    unsigned int tens = number / 10;

    bool units_A = units >> 0 & 1;
    bool units_B = units >> 1 & 1;
    bool units_C = units >> 2 & 1;
    bool units_D = units >> 3 & 1;

    bool tens_A = tens >> 0 & 1;
    bool tens_B = tens >> 1 & 1;
    bool tens_C = tens >> 2 & 1;
    bool tens_D = tens >> 3 & 1;

    digitalWrite(UNITS_A_PIN, units_A);
    digitalWrite(UNITS_B_PIN, units_B);
    digitalWrite(UNITS_C_PIN, units_C);
    digitalWrite(UNITS_D_PIN, units_D);

    digitalWrite(TENS_A_PIN, tens_A);
    digitalWrite(TENS_B_PIN, tens_B);
    digitalWrite(TENS_C_PIN, tens_C);
    digitalWrite(TENS_D_PIN, tens_D);
}

void Display::disable() {
    digitalWrite(UNITS_A_PIN, 0);
    digitalWrite(UNITS_B_PIN, 1);
    digitalWrite(UNITS_C_PIN, 0);
    digitalWrite(UNITS_D_PIN, 1);

    digitalWrite(TENS_A_PIN, 0);
    digitalWrite(TENS_B_PIN, 1);
    digitalWrite(TENS_C_PIN, 0);
    digitalWrite(TENS_D_PIN, 1);
}
