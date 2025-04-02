from PIL import Image, ImageDraw
import numpy as np
import csv
from tflite_runtime.interpreter import Interpreter
import time
import os
import psutil
from datetime import datetime

"""CFG START"""

MODEL = 'yolo11m_float16.tflite'
IMAGE_DIR = '../../assets/demo_images'
OUTPUT_DIR = '../../assets/results/results_yolo_tflite_cnrpark/yolov11m_tflite'
savefigs = 'debug' #choose 'no' to not save images and 'debug' to save images
mask_file = 'cnrpark_mask_original_img_1000_750_bw.png' # 'mask_original_img_768_1024_bw.png' or 'cnrpark_mask_original_img_1000_750_bw.png' or 'all_black_mask.png' to count all cars
"""CFG END"""

# IMAGE_DIR = '../../assets/original'
# OUTPUT_DIR = '../../assets/results/results_yolov11mtflite_ic2/yolov11mtflite'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def xywh_to_xyxy(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def nms(boxes, scores, iou_threshold=0.5, max_det=300):
    if len(boxes) == 0:
        return []
    
    boxes = xywh_to_xyxy(boxes)  # Ensure correct format

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # Sort by confidence score

    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep

def process_data(input_data):
    global input_details
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


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
            # if scores[i] >= th and classes[i] == 2:  # class 0: persons, 2 cars, 7 trucks
            if scores[i] >= th and (classes[i] == 2 or classes[i] == 7):
                box = boxes_xywh[i]
                detected_boxes.append(box)
    else:
        for i in output[0]:
            if i[-1] == 1 and i[-2] >= th:  # Assuming [x, y, width, height, score, class_id]
                detected_boxes.append(i[:4])
                
    if detected_boxes:
        detected_boxes = np.array(detected_boxes)
        scores = detected_boxes[:, 3]  # Assuming the score is stored in the height dimension
        
        keep = nms(detected_boxes, scores, iou_threshold=0.45)
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


def process_image(img_object):
    img = img_object.resize((height, width))
    img = np.expand_dims(np.array(img), axis=0)
    if 'float' in str(type):
        img = img / 256.0
        img = img.astype(type)
    return img

def save_predictions_to_csv(data, csv_path):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['image_name','predicted_cars', 'processing_time','cpu_usage', 'memory_used', 'swap_used']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

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

""" Inference """
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
if not image_files:
    raise ValueError("No images found in the specified directory.")


interpreter = Interpreter(MODEL, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, width, height, _ = input_details[0]['shape']
type = input_details[0]['dtype']

mask = Image.open(mask_file)
mask = np.array(mask)
now = datetime.now()
filename_timestamp = now.strftime("%Y%m%dT%H%M%S")
output_path = f'{OUTPUT_DIR}/batch_{filename_timestamp}'
output_csv_path = output_path
output_csv_file = os.path.join(output_csv_path, f'df_individual_metrics_{filename_timestamp}.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(output_csv_path, exist_ok=True)


for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    with Image.open(image_path) as img_object:
        img_width, img_height = img_object.size
        start_time = time.time()
        processed_image = process_image(img_object)
        pre_inference_time = time.time()
        
        num_cars, boxes = count_cars(processed_image,mask, 0.25)
        inference_time = time.time() - pre_inference_time
        total_processing_time = time.time() - start_time

        if savefigs == 'debug':
            # Draw bounding boxes on the original image
            annotated_image = draw_boxes_on_image(img_object.copy(), boxes, img_width, img_height)

            # Save the annotated image
            image_name = f'annotated_image_{MODEL[:-7]}{image_file}'
            annotated_image_path = os.path.join(output_csv_path, image_name)
            annotated_image.save(annotated_image_path)
            print(f"Annotated image saved to {annotated_image_path}")

        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        memory_used = memory_info.used / (1024 ** 2)  # Convert to MB
        swap_used = swap_info.used / (1024 ** 2)  # Convert to MB
        
        save_predictions_to_csv({
            'image_name': image_file,
            'predicted_cars': num_cars,
            'processing_time': total_processing_time,
            # 'inference_time': inference_time,
            'cpu_usage': cpu_usage,
            'memory_used': memory_used,
            'swap_used': swap_used
        }, output_csv_file)
        
        print(f"Processed {image_file}: {num_cars} cars detected, {total_processing_time:.2f}s total, {inference_time:.2f}s inference, CPU {cpu_usage}%, Memory {memory_used:.2f}MB, Swap {swap_used:.2f}MB")
