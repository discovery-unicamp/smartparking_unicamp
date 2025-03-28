import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import imageio as iio
import numpy as np
import os
import csv
import time
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
import psutil  # For system memory and CPU usage
from datetime import datetime


class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81  # 80 classes + BG

"""CFG START"""
model = mrcnn.model.MaskRCNN(mode="inference", config=SimpleConfig(), model_dir=os.getcwd())
model.load_weights(filepath="mask_rcnn_coco.h5", by_name=True)
input_dir = '../../assets/demo_images'
OUTPUT_DIR = '../../assets/results/results_maskrcnn_cnrpark/markrcnn_cnrpark'
savefigs = 'debug' #choose 'no' to not save images and 'debug' to save images
mask_file = 'cnrpark_mask_original_img_1000_750_bw.png' # 'mask_original_img_768_1024_bw.png' or 'cnrpark_mask_original_img_1000_750_bw.png' or 'all_black_mask.png' to count all cars
"""CFG END"""

# input_dir = '../../assets/original'
# OUTPUT_DIR = '../../assets/results/results_maskrcnn_ic2/markrcnn_ic2'


now = datetime.now()
filename_timestamp = now.strftime("%Y%m%dT%H%M%S")
output_path = f'{OUTPUT_DIR}/batch_{filename_timestamp}'
output_csv_path = output_path
OUTPUT_FILE = f'{output_path}/df_individual_metrics_{filename_timestamp}.csv'
output_csv = OUTPUT_FILE 
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(output_csv_path, exist_ok=True)




CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

data = []
mask = iio.imread(mask_file)

def detection_matrix_modified(x, y, mask, img_width, img_height):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    pixel_value = mask[int(y), int(x)]
    
    return pixel_value != 255


def count_cars_post(boxes, class_ids, mask, img_width, img_height):
    car_count = 0
    for box, class_id in zip(boxes, class_ids):
        # Check if the detected object is a car or truck
        if CLASS_NAMES[class_id] in ['car', 'truck']:
            y1, x1, y2, x2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            if detection_matrix_modified(x_center, y_center, mask, img_width, img_height):
                car_count += 1
    return car_count


# Get list of image files
image_files = [file for file in os.listdir(input_dir) if file.lower().endswith(('jpg', 'jpeg', 'png'))]
total_images = len(image_files)

# Modify the main loop to include progress bar and system metrics
for file in tqdm(image_files, desc="Processing Images", unit="image"):
    image_path = os.path.join(input_dir, file)
    image = iio.imread(image_path)

    image = np.array(image)
    
    start_time = time.time()
    r = model.detect([image], verbose=0)[0]
    inference_time = time.time() - start_time
    
    # Count only cars and trucks
    car_count = sum(1 for i in r['class_ids'] if CLASS_NAMES[i] in ['car', 'truck'])
    masked_car_count = count_cars_post(r['rois'], r['class_ids'], mask, 768, 1024)
    if savefigs == 'debug':
        mrcnn.visualize.display_instances(image=image, boxes=r['rois'], masks=r['masks'], class_ids=r['class_ids'], class_names=CLASS_NAMES, scores=r['scores'],save_fig_path=f'{output_csv_path}/annotated_{file}')
        plt.close() 

        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        memory_used = memory_info.used / (1024 ** 2)  # Convert to MB
        swap_used = swap_info.used / (1024 ** 2)  # Convert to MB
                
    
    data.append([file,masked_car_count, inference_time, cpu_usage,memory_used, swap_used])
    print(f"{file}: {car_count} cars/trucks detected, {masked_car_count} in mask, inference time: {inference_time:.2f} sec, memory used: {memory_used:.2f} MB, CPU usage: {cpu_usage:.2f}%")

with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name','predicted_cars', 'processing_time','cpu_usage', 'memory_used', 'swap_used'])
    writer.writerows(data)

print(f"Results saved to {output_csv}")