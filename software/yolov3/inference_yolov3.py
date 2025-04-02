# prev version working but not exactly as expected as the mask needs to be resized also

import cv2
import os
import csv
import numpy as np
from pytorchyolo import detect, models
from PIL import Image
import time
import os
import psutil
from datetime import datetime


"""CFG START"""
model_path = "yolov3.cfg"
weights_path = "yolov3.weights"
model = models.load_model(model_path, weights_path)
image_dir = '../../assets/demo_images'
output_dir = '../../assets/results/results_yolov3_cnrpark/yolov3'
savefigs = 'debug' #choose 'no' to not save images and 'debug' to save images
mask_file = 'cnrpark_mask_resized_img_416_416_bw.png' # ''mask_original_img_416_416_bw.png' or 'cnrpark_mask_resized_img_416_416_bw.png' or 'all_black_mask.png' to count all cars
"""CFG END"""

# image_dir = '../../assets/original'
# output_dir = '../../assets/results/results_yolov3_ic2/yolov3_ic2'

now = datetime.now()
filename_timestamp = now.strftime("%Y%m%dT%H%M%S")
output_path = f'{output_dir}/batch_{filename_timestamp}'
output_csv_path = output_path
OUTPUT_FILE = f'{output_path}/df_individual_metrics_{filename_timestamp}.csv' 
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_csv_path, exist_ok=True)



# Define the class ID for "car" (COCO dataset)
CAR_CLASS_ID = 2


# with Image.open('mask_original_img_768_1024_bw.png') as mask:
with Image.open(mask_file) as mask:
    # mask = mask.resize((640, 640))
    mask = np.array(mask)


def detection_matrix_modified(x, y, mask, img_width, img_height):
    """
    Checks if a point (x, y) is inside the given binary mask.

    Parameters:
        - x, y: Coordinates of the point (bounding box center)
        - mask: Binary mask (numpy array) where 255 is the area of interest
        - img_width, img_height: Original image dimensions
    Returns:
        - True if the point is inside the mask, False otherwise.
    """

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Convert multi-channel mask to single-channel

    # Get mask dimensions
    mask_height, mask_width = mask.shape[:2]
    yolo_image_size = 416  # Make sure this matches your YOLO input size

    # Compute scaling factors
    scale_x = mask_width / yolo_image_size  # Width scaling
    scale_y = mask_height / yolo_image_size  # Height scaling

    # Map bounding box center to the mask space
    x_mask = int(x * scale_x)
    y_mask = int(y * scale_y)

    # Ensure coordinates are within bounds
    x_mask = min(max(x_mask, 0), mask_width - 1)
    y_mask = min(max(y_mask, 0), mask_height - 1)

    # Get the pixel value from the mask
    pixel_value = mask[y_mask, x_mask]

    print(f"\nPoint: (x={x}, y={y}) -> Mapped to Mask: (x={x_mask}, y={y_mask})")
    print(f"Pixel Value: {pixel_value}")

    if pixel_value == 255:
        print("The center of the bounding box is outside the mask.")
        return False
    else:
        print("The center of the bounding box is inside the mask.")
        return True


def count_cars_post(boxes, mask, img_width, img_height):
    """
    Counts the number of cars (class ID 2) inside the mask.

    Parameters:
        - boxes: List of detected objects in YOLOv3 format [x_min, y_min, x_max, y_max, confidence, class_id]
        - mask: Binary mask (numpy array) where 255 is the area of interest
        - img_width, img_height: Original image dimensions
    Returns:
        - Number of cars inside the mask
    """

    car_count = 0

    for box in boxes:
        # Convert box to numpy array (in case it's a list)
        box = np.array(box)

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = box


        # Compute bounding box center
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Check if the bounding box center is inside the mask
        point_inside = detection_matrix_modified(x_center, y_center, mask, 416, 416)
        if point_inside:
            car_count += 1

    return car_count

# Open the CSV file for writing
with open(OUTPUT_FILE, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(['image_name','predicted_cars', 'processing_time','cpu_usage', 'memory_used', 'swap_used'])

    # Loop through every image in the directory
    for image_name in os.listdir(image_dir):
        # Construct the full image path
        image_path = os.path.join(image_dir, image_name)

        print(f"image path is {image_path}")
        with Image.open(image_path) as img:
            img = img.resize((416, 416))
      
            img = np.array(img)          
            # to save resized mask to produce a mask of the same size as the image  
            # resized_image_path = os.path.join(output_csv_path, f"resized_{image_name}")
            # cv2.imwrite(resized_image_path, img)   
            # Skip if the image is not loaded successfully
            if img is None:
                print(f"Error: Unable to load image at {image_path}")
                continue

            # Get image dimensions
            img_height, img_width, _ = img.shape

            img_rgb = img
            
            # Run the YOLO model on the image
            start_time = time.time()
            boxes = detect.detect_image(model, img_rgb)

            # Filter for cars and extract bounding boxes, scores, and class IDs
            car_boxes = []
            scores = []
            for box in boxes:
                print(f'box: {box}')
                x1, y1, x2, y2, confidence, class_id = box
                # if int(class_id) == CAR_CLASS_ID:  # Check if the detected object is a car
                if (int(class_id) == 2 or int(class_id) == 7):
                    car_boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)

            if car_boxes:
                car_boxes = np.array(car_boxes)
                scores =  car_boxes[:, 3]
                # Count cars inside the mask
                car_count = count_cars_post(car_boxes, mask, img_width, img_height)
            else:
                car_count = 0

            # count time
            inference_time = time.time() - start_time

            cpu_usage = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            swap_info = psutil.swap_memory()
            memory_used = memory_info.used / (1024 ** 2)  # Convert to MB
            swap_used = swap_info.used / (1024 ** 2)  # Convert to MB
            # Write the image name and number of cars to the CSV file
            
            csv_writer.writerow([image_name, car_count,inference_time, cpu_usage, memory_used, swap_used])
            if savefigs == 'debug':
                # Draw bounding boxes on the image
                for box in car_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"Car: {confidence:.2f}"
                    cv2.putText(img_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Add the total number of cars detected to the image
                text = f"Total Cars: {car_count}"
                cv2.putText(img_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                # Optionally, save the output image with bounding boxes
                output_image_path = os.path.join(output_csv_path, f"output_{image_name}")
                cv2.imwrite(output_image_path, img_rgb)

            # Print the number of cars detected for the current image
            print(f"Image: {image_name}, Number of cars detected: {car_count}")

print(f"Predictions saved to {output_csv_path}")