from PIL import Image, ImageDraw,ImageFont
import numpy as np
import csv
from tflite_runtime.interpreter import Interpreter
import time
import os
import cv2
import psutil
from datetime import datetime

"""CFG START"""
MODEL = 'lite-model_efficientdet_lite2_detection_default_1.tflite'
IMAGE_DIR = '../../assets/demo_images'
OUTPUT_DIR = '../../assets/results/results_efficientdet_tflite_cnrpark/efficientdetd2lite_cnrpark'
savefigs = 'debug' #choose 'no' to not save images and 'debug' to save images
mask_file = 'cnrpark_mask_original_img_1000_750_bw.png' # 'mask_original_img_768_1024_bw.png' or 'cnrpark_mask_original_img_1000_750_bw.png' or 'all_black_mask.png' to count all cars
"""CFG END"""


# IMAGE_DIR = '../../assets/original'
# OUTPUT_DIR = '../../assets/results/results_efficientdet_tflite_ic2/efficientdetd2lite_ic2'

now = datetime.now()
filename_timestamp = now.strftime("%Y%m%dT%H%M%S")
output_path = f'{OUTPUT_DIR}/batch_{filename_timestamp}'
output_csv_path = output_path
OUTPUT_FILE = f'{output_path}/df_individual_metrics_{filename_timestamp}.csv' 
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(output_csv_path, exist_ok=True)


def process_data(input_data):
    start_time_img = time.time()
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    end_time_img = time.time()
    return result, end_time_img - start_time_img

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




def count_cars_post(boxes, mask):
    """
    Count the number of bounding boxes whose center lies inside the mask.
    boxes: List of bounding boxes in [xmin, ymin, xmax, ymax] format.
    mask: Binary mask where 0 indicates the region of interest.
    """
    car_count = 0

    for box in boxes:
        xmin, ymin, xmax, ymax = box

        # Calculate the center of the bounding box
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        # Check if the center is inside the mask
        point_inside = detection_matrix_modified(x_center, y_center, mask)
        if point_inside:
            car_count += 1

    return car_count

def detection_matrix_modified(x, y, mask):
    """
    Check if a point (x, y) lies inside the mask.
    x, y: Normalized coordinates (between 0 and 1).
    mask: Binary mask where 0 indicates the region of interest.
    """
    mask = mask[:, :, 0]  # Ensure mask is 2D
    mask_x, mask_y = mask.shape

    # Convert normalized coordinates to pixel coordinates
    x_pixel = int(x * mask_y)
    y_pixel = int(y * mask_x)

    # Check if the point is inside the mask
    if 0 <= x_pixel < mask_y and 0 <= y_pixel < mask_x:
        pixel_value = mask[y_pixel, x_pixel]
        return pixel_value != 255  # Return True if inside the mask (pixel_value != 255)
    else:
        return False  # Point is outside the mask boundaries

def count_cars(input_data, th, mask):
    """
    Count the number of detected objects (e.g., cars) whose bounding box centers lie inside the mask.
    input_data: Preprocessed input image.
    th: Confidence threshold.
    mask: Binary mask where 0 indicates the region of interest.
    """
    global output_details
    output, inf_time = process_data(input_data)
    detected_boxes = []
    boxes_idx, classes_idx, scores_idx = 0, 1, 2
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores)):
        if scores[i] >= th and classes[i] == 2:  # Class 2: cars
            box = [boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]]  # [xmin, ymin, xmax, ymax]
            detected_boxes.append(box)

    if detected_boxes:
        detected_boxes = np.array(detected_boxes)
        scores = detected_boxes[:, 3]  # Assuming the score is stored in the height dimension

        # Apply Non-Maximum Suppression (NMS)
        keep = nms(detected_boxes, scores, iou_threshold=0.5)
        detected_boxes = detected_boxes[keep]

        # Count cars inside the mask
        cars = count_cars_post(detected_boxes, mask)

        return cars, detected_boxes, inf_time
    else:
        return 0, detected_boxes, inf_time


def process_image(img_object):
    global type
    
    img = img_object.resize((height, width))
    # img = cv2.cvtColor(img_object, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(np.array(img), axis=0)
    if 'float' in str(type):
        img = img / 255.0
        img = img.astype(type)
    return img

def save_predictions_to_txt(boxes, output_path, img_width, img_height):
    with open(output_path, 'w') as f:
        for box in boxes:
            x_center, y_center, box_width, box_height = box
            # Convert from normalized to pixel values
            x_center *= img_width
            y_center *= img_height
            box_width *= img_width
            box_height *= img_height
            f.write(f'{x_center} {y_center} {box_width} {box_height}\n')


def draw_boxes_on_image(image, boxes, img_width, img_height):
    """
    the model outputs [xmin, ymin, xmax, ymax], not [x_center, y_center, width, height]
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        # Assuming box is in [xmin, ymin, xmax, ymax] format
        xmin, ymin, xmax, ymax = box
        
        # Convert from normalized to pixel values if necessary
        xmin *= img_width
        ymin *= img_height
        xmax *= img_width
        ymax *= img_height
        
        # Draw the rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    counter_text = f"cars: {num_cars}"
    # font = ImageFont.load_default()  # Use default font
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    text_position = (10, 10)  # Top-left corner (x, y)
    
    # Draw the counter text
    draw.text(text_position, counter_text, fill="red", font=font)
    return image

interpreter = Interpreter(MODEL, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, width, height, _ = input_details[0]['shape']
type = input_details[0]['dtype']
print(f"width: {width}, height: {height}")
mask = Image.open(mask_file)
mask = np.array(mask)

with open(OUTPUT_FILE, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name','predicted_cars', 'processing_time','cpu_usage', 'memory_used', 'swap_used'])

    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(IMAGE_DIR, filename)

            with Image.open(image_path) as img_object:
                img_width, img_height = img_object.size
                # print(img_width, img_height)
                num_cars, boxes,inf_time = count_cars(process_image(img_object), 0.25,mask)
                print(f"Number of cars detected: {num_cars}")

                if savefigs == 'debug':
                    # Draw bounding boxes on the original image
                    annotated_image = draw_boxes_on_image(img_object.copy(), boxes, img_width, img_height)

                    # Save the annotated image
                    image_name = f'annotated_{filename}'
                    annotated_image_path = os.path.join(output_csv_path, image_name)
                    annotated_image.save(annotated_image_path)
                    # img_object.save(os.path.join(output_csv_path, f'raw_{image_name}'))
                    print(f"Annotated image saved to {annotated_image_path}")

                cpu_usage = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                swap_info = psutil.swap_memory()
                memory_used = memory_info.used / (1024 ** 2)  # Convert to MB
                swap_used = swap_info.used / (1024 ** 2)  # Convert to MB
                        
                writer.writerow([filename, num_cars, inf_time,cpu_usage, memory_used, swap_used])
print("Processing completed. Results saved to:", OUTPUT_FILE)