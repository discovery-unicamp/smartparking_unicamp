# 📊 InfluxDB Data Viewer  

## 📌 Overview  
This project provides tools for **monitoring and analyzing data** stored in an **InfluxDB** database. It includes:  
- A **Jupyter Notebook** for querying and visualizing data from InfluxDB  
- A **Telegram monitoring bot** (located in [bot folder](./bot/)) that alerts if data stops updating  

## 🚀 Features  
- Connects to **InfluxDB** to fetch historical data  
- Supports **time range filtering** and **device-based analysis**  
- Generates **interactive and static plots**  
- Highlights **weekends, holidays, and key events**  
- Saves visualizations as PDFs  
- Includes a **monitoring bot** that sends alerts via Telegram  

## 🌍 Hosting Information  

### InfluxDB Hosting  
The InfluxDB instance can be hosted in different ways:  
- **Self-hosted**: Install and run InfluxDB on a local machine (`http://localhost:8086`).  
- **Cloud-based**: Use [InfluxDB Cloud](https://www.influxdata.com/products/influxdb-cloud/) for a managed database.  



## 📦 Installation  

### Dependencies  
Ensure you have Python 3.11 installed and install required libraries:  
```sh
pip install -r requirements.txt
```  

### Set Up InfluxDB  
1. Create an **InfluxDB account** (if not already set up).  
2. Generate an **API token** with **read permissions**.  
3. Retrieve the **organization name** and **bucket name**.  

### 🔑 Store Token  
Save your **InfluxDB API token** in a file:  
```sh
echo "your_influxdb_read_token" > token_read.txt
```  

## 🔧 Configuration  
Modify the following variables in the notebook (`influx_viewer.ipynb`):  

| Variable          | Description                                       | Example Value |
|------------------|------------------------------------------------|--------------|
| `token`         | InfluxDB API token (read from `token_read.txt`)  | `"your_influxdb_read_token"` |
| `org`           | InfluxDB Organization Name                         | `"Your Influx Organization"` |
| `url`           | InfluxDB URL                                      | `"http://localhost:8086"` |
| `days`          | Number of days to fetch from InfluxDB             | `77` |
| `from_parquet`  | Load data from a Parquet file instead of InfluxDB | `False` |
| `export`        | Export retrieved data to a Parquet file           | `True` |

## ▶️ Usage  

Run `influx_viewer.ipynb`.  


### 📊 Visualizations  
The script generates:  
- A **line plot** showing detected cars over time.  
- Highlights for **weekends, holidays, and key dates**.  
- A **PDF report** (`cars_influx.pdf`) with the final visualization.  

### 📄 Saving Results  
The script saves the visualization as a **PDF file** (`cars_influx.pdf`).  

## 🛠️ Notes  
- Modify the **time range** in `start_time_filter` and `end_time_filter` to focus on specific periods.  
- Update `exclude_ids` to filter out_
