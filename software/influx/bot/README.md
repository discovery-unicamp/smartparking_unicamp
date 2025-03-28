# Raspberry Pi InfluxDB Monitor  

## 📌 Overview  
This script monitors a Raspberry Pi (Pi 3) and ensures it is sending data to an **InfluxDB** database. If data stops coming in for a specified period, it sends an **alert to a Telegram chat**.  

## 🚀 Features  
- Periodically checks the latest data in InfluxDB  
- Sends a Telegram notification if no data is received for a configurable time  
- Supports reading from both **InfluxDB** and **Parquet files**  
- Provides basic **Telegram bot commands** (`/report`, `/status`)  

## 🌍 Hosting Information  

### Telegram Bot Hosting  
- The bot runs **locally** on your machine or a Raspberry Pi.  
- For **24/7 availability**, consider hosting on a **cloud server (AWS, DigitalOcean, etc.)** or using a **Raspberry Pi** with a process manager like `systemd` or `pm2`.  

## ⚙️ Installation  

## 📦 Dependencies  
The script requires Python libraries. Tested with Python 3.11  

Install them using:  
```sh
pip install -r requirements.txt
```  

### Set Up InfluxDB  
- Create an **InfluxDB account** (if not already set up)  
- Generate an **API token** with write and read permissions  
- Get the **organization name** and **bucket name**  

### Set Up Telegram Bot  
- Create a **Telegram Bot** using [@BotFather](https://t.me/BotFather)  
- Get the **bot token**  
- Find your **Telegram Chat ID** (e.g., using `@userinfobot`)  

### Store Tokens in Files  
Save your **InfluxDB token** and **Telegram bot token** in separate files:  
```sh
echo "your_influxdb_token" > influx_token.txt
echo "your_telegram_bot_token" > bot_token.txt
```  

## 🔧 Configuration  
Modify the following variables in the script:  

| Variable          | Description                                       | Example Value |
|------------------|------------------------------------------------|--------------|
| `token`         | InfluxDB API token (read from `influx_token.txt`) | `"your_influxdb_token"` |
| `bot_token`     | Telegram Bot API token (read from `bot_token.txt`) | `"your_telegram_bot_token"` |
| `org`           | InfluxDB Organization Name                         | `"Your Influx Organization"` |
| `url`           | InfluxDB URL                                      | `"http://localhost:8086"` |
| `bucket`        | InfluxDB Bucket Name                              | `"Your Bucket"` |
| `chat_id`       | Telegram Chat ID                                  | `"Your Chat ID"` |
| `days`          | How many days of data to check                    | `5` |
| `minutes_tolerance` | Maximum allowed delay before sending an alert | `5` |
| `seconds`       | Interval between checks (in seconds)              | `600` (10 minutes) |

## ▶️ Usage  
Run the script with:  
```sh
python monitor.py
```  
The script will:  
1. Retrieve the latest data from InfluxDB  
2. Check if the last data point is older than `minutes_tolerance`  
3. Send a Telegram alert if data is missing  

### 🔹 Telegram Bot Commands  
The bot also supports simple commands:  
- `/status` → Check if the bot is running  
- `/report` → Get a summary of recent data  
