# ESP8266 2-Digit 7-Segment Display Controller

## Introduction
This project is designed to run on an **ESP8266** microcontroller, which controls a **2-digit 7-segment display** using a **CD4511BE** BCD to 7-segment decoder. The ESP8266 retrieves data from an **Web Server** and updates the display accordingly.

## Table of Contents
- [ESP8266 2-Digit 7-Segment Display Controller](#esp8266-2-digit-7-segment-display-controller)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [How It Works](#how-it-works)
  - [File Overview](#file-overview)
    - [**Core Code Files**](#core-code-files)
    - [**PlatformIO Configuration**](#platformio-configuration)
  - [Configuration](#configuration)

## Installation
1. Install [PlatformIO](https://platformio.org/).
2. Clone this repository or copy the files to your local system.
3. Modify the `Credentials.h` file to include your WiFi credentials and InfluxDB information.
4. Compile and upload the code to your ESP8266 using PlatformIO.


## How It Works
1. The ESP8266 connects to WiFi and fetches data from an InfluxDB database.
2. The retrieved data (a number between **0-16**) is sent to the **CD4511BE** in **BCD format**.
3. The **CD4511BE** decodes the BCD signals and controls the **7-segment display**.
4. The display updates every **30 seconds**. If no new data is available, the display turns off.

## File Overview

### **Core Code Files**
- **`src/main.cpp`**  
  - Initializes WiFi and display.
  - Reads data from InfluxDB and updates the display every 30 seconds.
  - If it receives a invalid number (negative, greater than 16), or if it hasn't been able to retrieve updated information for more than a defined time, the display is turned off.

- **`src/Display.cpp`**  
  - Converts a number (0-99) into **Binary-Coded Decimal (BCD)**.
  - Sends the BCD signals to the **CD4511BE** decoder.
  - Can also turn off the display by sending a invalid BCD code (by sending the number 10 of example, which cannot be displayed on a single digit, the multiplexer turns all segments off).

- **`src/Wifi.cpp`**  
  - Manages ESP8266's connection to WiFi.

- **`src/Query.cpp`**  
  - Fetches data from an Web Server.
  - Also stores the last time an successful read was made so it can calculate the down time.

- **`src/OTA.cpp`**
  - Check for firmware updates
  - Updates itself when new version available

- **`include/Credentials.h`**  
  - Stores WiFi credentials and Web Server and Update Server address.

### **PlatformIO Configuration**
- **`platformio.ini`** – Configures PlatformIO build settings for ESP8266.

## Configuration
Modify **`include/Credentials.h`** to set your **WiFi SSID and password** before flashing the ESP8266.