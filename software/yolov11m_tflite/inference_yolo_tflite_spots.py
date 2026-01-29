from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv
from tflite_runtime.interpreter import Interpreter
import time
import os
import psutil
from datetime import datetime
# NEW
import csv
import ast
from numpy import sqrt

# # config to discover threshold
"""CFG START"""
SPLIT_LARGE = False

MODEL = 'yolo11m_float16.tflite'
IMAGE_DIR = '../../assets/demo_images'

OUTPUT_DIR = '../../assets/results/results_yolo_tflite_spots_discover_area/yolov11m_tflite'
CSV_PATH = '../../assets/labels/custom_labeling/spot_wise_labels.csv'
savefigs = 'debug' #choose 'no' to not save images and 'debug' to save images
mask_file = 'cnrpark_mask_original_img_1000_750_bw.png' # 'mask_original_img_768_1024_bw.png' or 'cnrpark_mask_original_img_1000_750_bw.png' or 'all_black_mask.png' to count all cars
DRAW_SIZES = True  # Draw sizes on boxes
ANNOT_DIST = False  # Annotate distance lines
"""CFG END"""


# # # base config to apply threshold and ABBP
# """CFG START"""
# SPLIT_LARGE = True
# # only if split large is true, define spots and threshold
# spots_id = [0,1,2,3,4,5,6]
# threshold = 5720
# MODEL = 'yolo11m_float16.tflite'
# IMAGE_DIR = '../../assets/demo_images'

# OUTPUT_DIR = '../../assets/results/results_yolo_tflite_spots_split_abbp/yolov11m_tflite'
# CSV_PATH = '../../assets/labels/custom_labeling/spot_wise_labels.csv'
# savefigs = 'debug' #choose 'no' to not save images and 'debug' to save images
# mask_file = 'cnrpark_mask_original_img_1000_750_bw.png' # 'mask_original_img_768_1024_bw.png' or 'cnrpark_mask_original_img_1000_750_bw.png' or 'all_black_mask.png' to count all cars
# DRAW_SIZES = True  # Draw sizes on boxes
# ANNOT_DIST = False  # Annotate distance lines
# """CFG END"""


os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_spot_coordinates(csv_path):
    """Load parking spot coordinates from CSV (one row with all spot boxes)."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return []

            # Get the first row's yolo_boxes
            first_row = rows[0]
            spot_boxes = ast.literal_eval(first_row['yolo_boxes'])

            # Create spot definitions
            spots = []
            for spot_id, box in enumerate(spot_boxes):
                if len(box) >= 3:  # Need at least class, x_center, y_center
                    spots.append({
                        'id': spot_id,
                        'x_center': box[1],  # Normalized x center
                        'y_center': box[2],  # Normalized y center
                        'width': box[3] if len(box) > 3 else 0.05,
                        'height': box[4] if len(box) > 4 else 0.05,
                        'occupied': False
                    })
            return spots
    except Exception as e:
        print(f"Error loading spot coordinates: {e}")
        return []


def assign_detections_to_spots(detections, spots, img_width, img_height,mask, max_distance=0.3):
    """
    Assign detections to nearest parking spots within max_distance (normalized).
    Returns: (updated spots, list of (detection_idx, spot_idx) assignments)
    """
    print("Assigning detections to spots...")
    print(f"Total detections: {len(detections)}, Total spots: {len(spots)}")
    print(f"detections are: {detections}")
    assignments = []
    for spot in spots:
        print(f"Spot {spot['id']} at ({spot['x_center']}, {spot['y_center']})")
        spot['occupied'] = False  # Reset all spots
    
    for det_idx, det in enumerate(detections):
        if len(det) < 4:
            continue
            
        x_center, y_center = det[0], det[1]
        print(f"Detection {det_idx} at ({x_center}, {y_center})")
        
        nearest_spot_idx = None
        min_dist = float('inf')
        
        for spot_idx, spot in enumerate(spots):
            dist = sqrt((x_center - spot['x_center'])**2 + (y_center - spot['y_center'])**2)
            
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                nearest_spot_idx = spot_idx

        if nearest_spot_idx is not None:
            if detection_matrix_modified(x_center,y_center,mask) == True:
                spots[nearest_spot_idx]['occupied'] = True
                assignments.append((det_idx, nearest_spot_idx))
            else:
                print(f"Detection {det_idx} at ({x_center}, {y_center}) is outside the mask, skipping assignment.")
            
    return spots, assignments

def draw_parking_status(image, spots, detections, assignments, img_width, img_height):
    """Draw parking spots and detections with assignment lines."""
    draw = ImageDraw.Draw(image)
    
    # Draw parking spots
    for spot in spots:
        x = spot['x_center'] * img_width
        y = spot['y_center'] * img_height
        w = spot['width'] * img_width / 2
        h = spot['height'] * img_height / 2
        
        color = (255, 0, 0, 150) if spot['occupied'] else (0, 255, 0, 150)
        # draw.rectangle([x-w, y-h, x+w, y+h], outline=color, width=2)
        draw.rectangle([x-w, y-h, x+w, y+h], outline=color, width=2)
        draw.text((x-w+5, y-h+5), f"Spot {spot['id']}", fill="white")
    
    # Draw detections and assignment lines
    for det_idx, spot_idx in assignments:
        det = detections[det_idx]
        spot = spots[spot_idx]
        
        # Detection rectangle
        x1 = (det[0] - det[2]/2) * img_width
        y1 = (det[1] - det[3]/2) * img_height
        x2 = (det[0] + det[2]/2) * img_width
        y2 = (det[1] + det[3]/2) * img_height
        # draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Line from detection to spot center
        det_x = det[0] * img_width
        det_y = det[1] * img_height
        spot_x = spot['x_center'] * img_width
        spot_y = spot['y_center'] * img_height
        # draw.line([(det_x, det_y), (spot_x, spot_y)], fill="magenta", width=5)
        for offset in [(-1,0), (1,0), (0,-1), (0,1)]:
            draw.line(
                [(det_x+offset[0], det_y+offset[1]), (spot_x+offset[0], spot_y+offset[1])],
                fill="black",
                width=7
    )
        draw.line([(det_x, det_y), (spot_x, spot_y)], fill="black", width=7)

        # Main line
        draw.line([(det_x, det_y), (spot_x, spot_y)], fill="orange", width=5)
        if ANNOT_DIST:
            mid_x = (det_x + spot_x) / 2
            mid_y = (det_y + spot_y) / 2
            x_center, y_center = det[0], det[1]
            
            distance = sqrt((x_center - spot['x_center'])**2 + (y_center - spot['y_center'])**2)
            draw.text((mid_x, mid_y), f"{distance:.3f}px", fill="black")
    
    return image

def draw_parking_status_sizes(image, spots, detections, assignments, img_width, img_height):
    """Draw parking spots and detections with assignment lines."""
    draw = ImageDraw.Draw(image)
    areas = []
    spots_id = []
    # Draw parking spots
    for spot in spots:
        x = spot['x_center'] * img_width
        y = spot['y_center'] * img_height
        w = spot['width'] * img_width / 2
        h = spot['height'] * img_height / 2
        
        color = (255, 0, 0, 150) if spot['occupied'] else (0, 255, 0, 150)
        draw.rectangle([x-w, y-h, x+w, y+h], outline=color, width=2)
        draw.text((x-w+5, y-h+5), f"Spot {spot['id']}", fill="white")
    
    # Draw detections and assignment lines
    for det_idx, spot_idx in assignments:
        det = detections[det_idx]
        spot = spots[spot_idx]
        spots_id.append(spot['id'])
        
        # Detection rectangle
        x1 = (det[0] - det[2]/2) * img_width
        y1 = (det[1] - det[3]/2) * img_height
        x2 = (det[0] + det[2]/2) * img_width
        y2 = (det[1] + det[3]/2) * img_height
        draw.rectangle([x1, y1, x2, y2], fill="red")  
        draw.rectangle([x1, y1, x2, y2], outline="black", width=2)
        #  Width and height
        box_w = int(x2 - x1)
        box_h = int(y2 - y1)
        area = box_w * box_h
        areas.append(area)
        text = f"{area}\n{box_w}x{box_h}"
        
        # Use a default font
        # font = ImageFont.load_default()  # or any FreeTypeFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        # Get bounding box of the text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Calculate centered position
        text_x = x1 + (x2 - x1 - text_w) / 2.8
        text_y = y1 + (y2 - y1 - text_h) / 2

        # Draw text
        # text_bg_margin = 3
        # draw.rectangle(
        #     [text_x - text_bg_margin, text_y - text_bg_margin,
        #     text_x + text_w + text_bg_margin, text_y + text_h + text_bg_margin],
        #     fill=(0, 0, 0, 128)  # semi-transparent black background
        # )
        # draw.text((text_x, text_y), text, fill="orange", font=font)
        draw.text(
            (text_x, text_y),
            text,
            fill="black",
            font=font,
            stroke_width=2,         # thickness of outline
            stroke_fill="white"     # outline color
        )
        # draw.text((x1 + 5, y1 + 5), f"{box_w}x{box_h}", fill="yellow")
        # Line from detection to spot center
        det_x = det[0] * img_width
        det_y = det[1] * img_height
        spot_x = spot['x_center'] * img_width
        spot_y = spot['y_center'] * img_height
        # draw.line([(det_x, det_y), (spot_x, spot_y)], fill="cyan", width=2)
    
    return image,areas,spots_id

def xywh_to_xyxy(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

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
        fieldnames = ['image_name','predicted_cars', 'processing_time','cpu_usage', 'memory_used', 'swap_used','areas','spots']
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

def split_large_detection(det, img_width, img_height,threshold=5720):
    """
    Divide uma bounding box no meio vertical se área > 4000.
    det = [x_center, y_center, width, height] (normalizado)
    """
    x_center, y_center, w, h = det
    abs_w = w * img_width
    abs_h = h * img_height
    area = abs_w * abs_h

    # if area <= 4500:
    if area <= threshold:
        return [det]  # mantém igual
    
    # Dividir em 2 caixas no eixo X
    new_w = w / 2
    left_box  = [x_center - new_w/2, y_center, new_w, h]
    right_box = [x_center + new_w/2, y_center, new_w, h]
    return [left_box, right_box]

""" Inference """
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
if not image_files:
    raise ValueError("No images found in the specified directory.")

## load spots
spots = load_spot_coordinates(CSV_PATH)
if not spots:
    logger.info("Failed to load spot coordinates")
## Initialize TFLite model
interpreter = Interpreter(MODEL, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, width, height, _ = input_details[0]['shape']
type = input_details[0]['dtype']
## read mask
mask = Image.open(mask_file)
mask = np.array(mask)
now = datetime.now()
filename_timestamp = now.strftime("%Y%m%dT%H%M%S")
output_path = f'{OUTPUT_DIR}/batch_{filename_timestamp}'
output_csv_path = output_path
output_csv_file = os.path.join(output_csv_path, f'df_individual_metrics_{filename_timestamp}.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(output_csv_path, exist_ok=True)


## Main loop to perform inference
for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    with Image.open(image_path) as img_object:
        img_width, img_height = img_object.size
        start_time = time.time()
        processed_image = process_image(img_object)
        pre_inference_time = time.time()
        
        num_cars, boxes = count_cars(processed_image,mask, 0.25)
        inference_time = time.time() - pre_inference_time
        

        # --- START OF BLOCK THAT DIVIDES BOXES ---
        if SPLIT_LARGE:
            new_boxes = []
            for det in boxes:
                # check if detection is clone to spots 5,6 or 7
                for s in spots:
                    # if s['id'] in [4,5, 6, 7]:
                    if s['id'] in spots_id:
                        dist = sqrt((det[0] - s['x_center'])**2 + (det[1] - s['y_center'])**2)
                        if dist < 0.1:  # mesmo limiar de assign
                            split_boxes = split_large_detection(det, img_width, img_height,threshold)
                            new_boxes.extend(split_boxes)
                            break
                else:
                    new_boxes.append(det)  # se não for spot 5-7, mantém
            boxes = np.array(new_boxes)
        # --- END OF BLOCK THAT DIVIDES BOXES ---
        # end processing time
        # assign to spots
        spots, assignments = assign_detections_to_spots(boxes, spots, img_width, img_height,mask)

        
        ## compute metrics
        parking_status = ''.join(['1' if s['occupied'] else '0' for s in sorted(spots, key=lambda s: s['id'])])
        occupied_count = parking_status.count('1')
        num_cars = occupied_count
        parking_bitmask = int(parking_status, 2)
        parking_bitmask_bin = bin(parking_bitmask)[2:]
        parking_bitmask_bin = parking_bitmask_bin.zfill(16) 
        print(f"\n\nParking status for {image_file}: {parking_status} (Occupied: {occupied_count})\n")
        total_processing_time = time.time() - start_time

        if savefigs == 'debug':
            # Draw bounding boxes on the original image
            # annotated_image = draw_boxes_on_image(img_object.copy(), boxes, img_width, img_height)
            if DRAW_SIZES:
                annotated_image,areas,spots_id = draw_parking_status_sizes(img_object.copy(), spots, boxes, assignments, img_width, img_height)
            else:
                annotated_image = draw_parking_status(img_object.copy(), spots, boxes, assignments, img_width, img_height)
                areas = []
                spots_id=[]

            # Save the annotated image
            image_name = f'annotated_image_{MODEL[:-7]}{image_file}'
            annotated_image_path = os.path.join(output_csv_path, image_name)
            annotated_image.save(annotated_image_path)
            print(f"Annotated image saved to {annotated_image_path}")
        else:
            areas = []
            spots_id=[]

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
            'swap_used': swap_used,
            'areas': areas,
            'spots': spots_id
        }, output_csv_file)
        
        print(f"Processed {image_file}: {num_cars} cars detected, {total_processing_time:.2f}s total, {inference_time:.2f}s inference, CPU {cpu_usage}%, Memory {memory_used:.2f}MB, Swap {swap_used:.2f}MB")
