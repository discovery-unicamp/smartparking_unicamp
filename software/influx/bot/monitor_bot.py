import pandas as pd
import numpy as np
import plotly.graph_objs as go
from threading import Timer
import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
import logging
import requests
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")


def read_token(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.error(f"Token file not found: {file_path}")
        sys.exit(1)

# Read token
token = read_token('influx_token.txt')
bot_token = read_token('bot_token.txt')
org = "Your Influx Organization"
url = 'Your Influx URL'
chat_id = "Your Chat ID of Telegram"
bucket = "Your Bucket"
write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
days = '5'
from_parquet = False
export = False
seconds = 10 * 60

minutes_tolerance = 5

def send_message_to_telegram(timestamp,minutes_diff, bot_token, chat_id):
    # Convert the list to a string message
    # message = "Last time Pi3 was alive:\n" + "\n".join(timestamp)
    message = f"Pi 3 is not sending data to InfluxDB. Last data was received {round(minutes_diff,2)} minutes ago. Last timestamp: {timestamp}"
    # URL for the Telegram Bot API
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # Data payload to send the message
    payload = {
        'chat_id': chat_id,
        'text': message
    }

    # Send the message using the requests library
    response = requests.post(url, data=payload)
    print(response.text)
    # Check if the request was successful
    if response.status_code == 200:
        print("Message sent successfully")
    else:
        print(f"Failed to send message. Error: {response.status_code}")

def get_updates():
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    response = requests.get(url)
    return response.json()

def get_latest_update(updates):
    user_messages = {}
    for update in updates['result']:
        user_id = update['message']['from']['id']
        user_messages[user_id] = update
    return list(user_messages.values())


def process_command(command):
    if command.startswith("/report"):
        return f"Report: In the last 24 hours, {len(df_health)} inferences were sent to InfluxDB."
    elif command.startswith("/status"):
        return "Status: Bot is running."
    else:
        return "Invalid command. Please try again."


while True:
    
    
    print('retrieving data\n')
    if from_parquet == True:
        df_health =  pd.read_parquet(f'{bucket}.parquet')
    else:


        query_api = write_client.query_api()
        query = f"""from(bucket: {bucket})
        |> range(start:-{days}d )
        """
        # df_health_bot / health_bucket
        df_health = query_api.query_data_frame(query, org="Discovery")
        if export == True:
            table = pa.Table.from_pandas(df_health, preserve_index=True)
            pq.write_table(table, f'{bucket}.parquet')

    print(f'df_health aquired\n')
    # print(df_health.columns)
    print(df_health[['_time','_value']])
    # Step 1: Convert '_time' column to datetime format
    # df_health['_time'] = pd.to_datetime(df_health['_time'])
    timestamp = df_health['_time'].iloc[-1]

    now_plus_3h = pd.Timestamp(datetime.utcnow(), tz="UTC")

    # Calculate the difference
    time_diff = now_plus_3h - timestamp

    print("Time Difference:", time_diff)
    print("Time Difference in seconds:", time_diff.total_seconds())
    minutes_diff = time_diff.total_seconds() / 60
    print("Time Difference in minutes:", minutes_diff)

    
    if minutes_diff > minutes_tolerance:
        print("Missing data. Check the Pi3.")
        print(f'Sending message to Telegram \n')

        

        send_message_to_telegram(timestamp,minutes_diff, bot_token, chat_id)
    else:
        print("Data is up to date.")

    print(f'waiting for {seconds} seconds \n')
    time.sleep(seconds)
