#!/usr/bin/python3
import os
import sys
from datetime import datetime
from PIL import Image,ImageDraw
import time
import psutil
import redis
from threading import Timer
import argparse
import numpy as np
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
from tflite_runtime.interpreter import Interpreter
# import imageio.v2 as iio
import traceback
import shutil
from logging.handlers import TimedRotatingFileHandler
import picamera
# import subprocess

# Setup logging
log_file = 'inference_prod.log'
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

handler = TimedRotatingFileHandler(
    log_file, 
    when='midnight',  # Rotate logs daily
    interval=1,       # Interval in days
    backupCount=7     # Keep logs for 7 days
)
handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# Set up constants
inference_interval = 60 #every 60 seconds a new inference is started
TIME_TO_SEND = inference_interval/2 #+ 5#* 2
DEVICE_NAME = "pi3_ic2"
INPUT_PATH = '/home/pi/parking_2024/camera_output/tempdir/latest.jpg'
DATA_QUEUE = 'inference_queue'
REDIS_HOSTNAME = "localhost"
INFERENCE_BUCKET = "ic2_parking"
MODEL = 'yolo11m_float16.tflite'

# create temporary directory
os.system('sudo mount -t tmpfs -o size=500m tmpfs /home/pi/parking_2024/camera_output/tempdir')

def read_token(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.error(f"Token file not found: {file_path}")
        sys.exit(1)

# Read token
token = read_token('token_write.txt')

def initialize_connections(org, token, url, redis_hostname):
    # Initialize InfluxDB client
    try:
        client = influxdb_client.InfluxDBClient(url=url, token=token, org=org,timeout=30000)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        logger.info("InfluxDB client initialized.")
    except Exception as e:
        logger.error(f"Error initializing InfluxDB client: {e}")
        return None, None
    
    # Initialize Redis connection
    try:
        pool = redis.ConnectionPool(host=redis_hostname, port=6379, db=0, decode_responses=True)
        global redisConn
        redisConn = redis.Redis(connection_pool=pool)
        logger.info("Connected to Redis successfully.")
        redisConn.flushall()  # Clear Redis database at the start
        logger.info("Redis database cleared at startup.")
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        return None, None
    
    return write_api, redisConn

def process_image(img_object):
    global type, width, height
    img = img_object.resize((width, height))
    img = np.expand_dims(np.array(img), axis=0)
    if 'float' in str(type):
        img = img / 256.0
        img = img.astype(type)
    return img

def process_data(input_data):
    global input_details
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

def nms(boxes, scores, iou_threshold=0.5):
    x = boxes[:, 0]
    y = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    
    areas = width * height
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + width[i], x[order[1:]] + width[order[1:]])
        yy2 = np.minimum(y[i] + height[i], y[order[1:]] + height[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep



def count_cars(input_data, mask,th=0.25):
    global output_details
    output = process_data(input_data)
    detected_boxes = []
    if output_details[0]['name'] == 'Identity':
        output = output[0].T
        scores = np.max(output[..., 4:], axis=1)
        classes = np.argmax(output[..., 4:], axis=1)
        boxes_xywh = output[..., :4]  # Assuming output format is [x, y, width, height, scores...]
        for i in range(len(scores)):
            if scores[i] >= th and classes[i] == 2:  # class 0: persons, 2 cars, 7 trucks
            # if scores[i] >= th and (classes[i] == 2 or classes[i] == 7):
                box = boxes_xywh[i]
                detected_boxes.append(box)
    else:
        for i in output[0]:
            if i[-1] == 1 and i[-2] >= th:  # Assuming [x, y, width, height, score, class_id]
                detected_boxes.append(i[:4])
                
    if detected_boxes:
        detected_boxes = np.array(detected_boxes)
        scores = detected_boxes[:, 3]  # Assuming the score is stored in the height dimension
        
        keep = nms(detected_boxes, scores, iou_threshold=0.5)
        detected_boxes = detected_boxes[keep]

        cars = count_cars_post(detected_boxes,mask)

        return cars, detected_boxes
    else:
        return len(detected_boxes), detected_boxes


def detection_matrix_modified(x,y,mask):
    mask = mask[:,:,0]
    # print(mask.shape)
    mask_x,mask_y = mask.shape
    # mask_x,mask_y = mask.shape[:2]
    # mask_x, mask_y = (480,640)
    x = x*mask_y
    y = y*mask_x
    pixel_value = mask[int(y),int(x)]
    # print(f"\n points are {x},{y} \n pixel value: {pixel_value} and mask shape is {mask.shape}\n mask_x = {mask_x}, mask_y = {mask_y}")
    if pixel_value == 255:
        # print("The point is outside the mask.")
        return False
    else:
        # print("The point is inside the mask.")
        return True

def count_cars_post(lines, mask,class_names_dict=0):

    car_count = 0
    truck_count = 0

    for line in lines:

        line_ = np.array(line)
        x_center, y_center, width, height = line_

        point_inside = detection_matrix_modified(x_center,y_center,mask)
        # print(f"\n\n\n\n point {x_center}, {y_center} is {point_inside} ")
        if point_inside == True:
            
            car_count += 1

    return car_count + truck_count

def draw_boxes_on_image(image, boxes, img_width, img_height):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x_center, y_center, box_width, box_height = box
        # Convert from normalized to pixel values
        x_center *= img_width
        y_center *= img_height
        box_width *= img_width
        box_height *= img_height
        left = x_center - box_width / 2
        top = y_center - box_height / 2
        right = x_center + box_width / 2
        bottom = y_center + box_height / 2
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
    return image



def sendTimer(write_api, org, bucket_name):
    logger.info("Checking queue and sending data to InfluxDB.")
    
    def timer_function():
        t = Timer(TIME_TO_SEND, timer_function)
        t.start()

        queueSize = redisConn.llen(DATA_QUEUE)
        if queueSize > 0:
            for i in range(queueSize):
                try:
                    data = redisConn.rpop(DATA_QUEUE)
                    # logger.info(f"data to send: {data}")
                    p = influxdb_client.Point("detected_cars").tag("pi-id", DEVICE_NAME).field("cars", data)
                    write_api.write(bucket=bucket_name, org=org, record=p,timeout=30_000)
                    logger.info(f"Sent data: {data} to InfluxDB.")
                    redisConn.flushall()

                except Exception as e:
                    logger.error(f"Error while sending data: {e}")
        else:
            # logger.info("No records to send. Skipping...")
            pass
    
    timer_function()


# Initialize TFLite model
interpreter = Interpreter(MODEL, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, width, height, _ = input_details[0]['shape']
type = input_details[0]['dtype']


mask = Image.open('mask_original_img_768_1024_bw.png')
mask = np.array(mask)
# Initialize connections
write_api, redisConn = initialize_connections(org="YOUR ORGANIZATION", token=token, url="YOUR INFLUX URL", redis_hostname=REDIS_HOSTNAME)

if write_api is None or redisConn is None:
    logger.error("Failed to initialize connections. Exiting.")
    # return

# Start the timer to periodically send data
sendTimer(write_api, org="YOUR ORGANIZATION", bucket_name=INFERENCE_BUCKET)

# Main loop to perform inference at regular intervals
temp_index = 0

def periodic_run(output_dir, temp_index=0):
    capture_image(output_dir, temp_index=temp_index)

def create_directories(base_dir):
    # history_dir = os.path.join(base_dir, 'history')
    last_pic_dir = os.path.join(base_dir, 'last_picture')
    
    # os.makedirs(history_dir, exist_ok=True)
    os.makedirs(last_pic_dir, exist_ok=True)
    logging.info(f'Diretórios criados/verificados: {history_dir}, {last_pic_dir}')


def capture_image(output_dir, temp_index=0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_filename = f"{timestamp}.jpg"
    last_pic_filename = "latest.jpg"
    
    try:
        with picamera.PiCamera(resolution='1024x768') as camera:
            camera.rotation = 180
            camera.start_preview()
            time.sleep(2)  # Allow the sensor to adjust
            
            # Capture image and save to history directory
            # last_pic_path = os.path.join(output_dir, 'last_picture', last_pic_filename)
            last_pic_path = os.path.join(output_dir, last_pic_filename)
            camera.capture(last_pic_path)
            logging.info(f'Imagem capturada e salva: {last_pic_path}')
            
    except Exception as e:
        logging.error(f"Erro ao capturar imagem: {str(e)}")

while True:
    # take pic
    periodic_run('/home/pi/parking_2024/camera_output/tempdir/', temp_index%300)
    temp_index +=1
    time.sleep(3)
    # Sleep to avoid overload
    with Image.open(INPUT_PATH) as img_object:
        
        img_width, img_height = img_object.size
        logger.info(f"started inference of image {INPUT_PATH}")
        num_cars_, boxes = count_cars(process_image(img_object),mask, 0.25)
        logger.info(f"Number of cars detected: {num_cars_}")
        num_cars = 16 - num_cars_
        # num_cars = -1
        
        logger.info(f"Number of spots available: {num_cars}")
        
        redisConn.lpush(DATA_QUEUE, num_cars)
        time.sleep(inference_interval)



