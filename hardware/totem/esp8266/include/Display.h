#ifndef TOTEM_DISPLAY_H
#define TOTEM_DISPLAY_H

#include "Arduino.h"

#define UNITS_A_PIN D3
#define UNITS_B_PIN D0
#define UNITS_C_PIN D1
#define UNITS_D_PIN D2

#define TENS_A_PIN D8
#define TENS_B_PIN D5
#define TENS_C_PIN D6
#define TENS_D_PIN D7

class Display {

public:
    static void setup_pins();

    static void write(unsigned int number);

    static void disable();

};


#endif
