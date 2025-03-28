import os
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
from io import BytesIO
import pandas as pd
from statistics import mean
from datetime import datetime
import imageio.v3 as iio
from PIL import Image

def crop(img):
    img = img[:,100:-40]
    # img = cv2.resize(img, (884, 884))
    img = cv2.resize(img, (640, 640))
    return img

def mask_bin(mask):
    mask = crop(mask)
    mask = mask[:,:,0]
    mask[mask>0] = 100
    mask[mask==0] = 1
    mask[mask>1] = 0
    return mask

def detection_matrix(x,y,mask):
    mask = crop(mask)
    mask = mask_bin(mask)
    mask_x,mask_y = mask.shape
    if (x<=1) and (y<=1):
        x = x*mask_x
        y = y*mask_y
        print(f"\n points are {x},{y} \n mask shape is {mask.shape}\n")
    if (x < mask_x) and (y < mask_y):
        return (mask[int(x),int(y)])
    else:
        return None

def detection_matrix_modified(x,y,mask):
    mask = mask[:,:,0]
    print(mask.shape)
    mask_x,mask_y = mask.shape
    # mask_x,mask_y = mask.shape[:2]
    # mask_x, mask_y = (480,640)
    x = x*mask_y
    y = y*mask_x
    pixel_value = mask[int(y),int(x)]
    print(f"\n points are {x},{y} \n pixel value: {pixel_value} and mask shape is {mask.shape}\n mask_x = {mask_x}, mask_y = {mask_y}")
    if pixel_value == 255:
        print("The point is outside the mask.")
        return False
    else:
        print("The point is inside the mask.")
        return True

    


def list_images_subdirectories(directory_path, extensions=None):
    """
    List images in the specified directory with optional extensions filter.

    Parameters:
    - directory_path (str): Path to the directory.
    - extensions (list): List of allowed file extensions. If None, all files are considered.

    Returns:
    - image_list (list): List of image filenames in the directory.
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    image_list = []

    # List all files in the current directory
    current_files = os.listdir(directory_path)
    # Filter files based on extensions
    image_list.extend([os.path.join(directory_path, file) for file in current_files if any(file.lower().endswith(ext) for ext in extensions)])

    # Loop through subdirectories
    for subdir in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdirectory_path):
            # List all files in the subdirectory
            all_files = os.listdir(subdirectory_path)
            # Filter files based on extensions
            image_list.extend([os.path.join(subdir, file) for file in all_files if any(file.lower().endswith(ext) for ext in extensions)])
    image_list.sort()

    return image_list


def extract_timestamp(filename, mode='filename'):
    """
    Extract timestamp from the filename or retrieve it from the file modification_time.

    Parameters:
    - filename (str): Image filename.
    - mode (str): Mode to determine how to extract the timestamp.
                  Possible values: 'filename' or 'modification_time' or 'creation_time'.
                  Defaults to 'filename'.

    Returns:
    - timestamp (str): Extracted timestamp.
    """
    if mode == 'filename':
        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', filename)
        if timestamp_match:
            return timestamp_match.group()
        else:
            return None
    elif mode == 'filename_ic':
        timestamp_match = re.search(r'_(\d+)\.', filename)
        if timestamp_match:
            # Convert the timestamp to a datetime object and format it
            timestamp_str = timestamp_match.group(1)
            timestamp_int = int(timestamp_str)
            timestamp = datetime.fromtimestamp(timestamp_int /  1000).strftime('%Y-%m-%d-%H-%M-%S')
            return timestamp
        else:
            return None
    elif mode == 'filename_ru':
        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', filename)
        if timestamp_match:
            return timestamp_match.group()
        else:
            return None
    elif mode == 'modification_time':
        modification_time = os.path.getctime(filename)
        timestamp = datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d-%H-%M-%S')
        return timestamp
    elif mode == 'creation_time':
        creation_time = os.path.getmtime(filename)  # Use getmtime instead of getctime
        timestamp = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d-%H-%M-%S')
        return timestamp
    else:
        raise ValueError("Invalid mode. Mode must be either 'filename' or 'modification_time' or 'creation_time'.")


def check_camera_type(img_, cars,img_name,language='pt'):

    if language == 'eng':
        text = f'{img_name} - {cars} cars detected'
        cam = 1
        # f'Parking {preench}% full'
    else:
        text = f'{img_name} - {cars} carros detectados'
        cam = 1
    return text,cam

def count_cars_post(lines, class_names_dict,mask):
    # Initialize counters
    car_count = 0
    truck_count = 0

    # Iterate through lines and count cars and trucks
    for line in lines:
        class_index, x_center, y_center, width, height, confidence = map(float, line.split())
        class_name = class_names_dict[int(class_index)]


        if class_name == 'car':
            # point_inside = detection_matrix(x_center,y_center,mask)
            point_inside = detection_matrix_modified(x_center,y_center,mask)
            print(f"\n\n\n\n point {x_center}, {y_center} is {point_inside} ")
            if point_inside == True:
                
                car_count += 1

        elif class_name == 'truck':
            # point_inside = detection_matrix(x_center,y_center,mask)
            point_inside = detection_matrix_modified(x_center,y_center,mask)
            print(f"\n\n\n\n point {x_center}, {y_center} is {point_inside} \n\n\n\n")
            if point_inside == True:
                truck_count += 1

    return car_count + truck_count


def perform_inference_post(input_path, output_path, img_,img_object, model,df,mask, save=False,remove_txt=False,generate_txt=True):

    start_time_img = time.time()
    results = model.predict(img_object)
    end_time_img = time.time()
    elapsed_time = end_time_img - start_time_img

    print(f'image {img_}')
    print(f"Elapsed time to process: {elapsed_time} seconds")
    os.makedirs(output_path, exist_ok=True)

    # Show the results
    for r in results:
        print(f'\n results: \n {r}')
        img_name = img_.split('/')[-1][:-4]

        if generate_txt == True:
            txt_file_path = os.path.join(output_path, f'{img_name}_txt_file.txt')
            # Delete older version of the text file if it exists
            if os.path.exists(txt_file_path):
                os.remove(txt_file_path)
            r.save_txt(txt_file_path, save_conf=True)

            class_names_dict = r.names

            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r') as file:
                    lines = file.readlines()
            else:
                # if no car is detected, produce empty line
                lines = []        

            cars = count_cars_post(lines,class_names_dict,mask)
        else:

            class_names_dict = r.names
            boxes = r.boxes.xywhn.cpu().numpy()  # Get boxes in xyxy format
            confidences = r.boxes.conf.cpu().numpy()  # Get confidence scores
            classes = r.boxes.cls.cpu().numpy()  # Get class labels

            lines = []
            for box, conf, cls in zip(boxes, confidences, classes):
                line = f"{int(cls)} " + " ".join([f"{x:.6f}" for x in box]) + f" {conf:.6f}"
                print(line)
                lines.append(line + "\n")

            # Count the cars directly from the result lines
            cars = count_cars_post(lines,class_names_dict,mask)

        print(f"\n\n {cars} detected")


        text,cam = check_camera_type(img_, cars,img_name,language='eng')

        # Output the count to a new text file
        img_name = img_.split('/')[-1][:-4]

        # get metric
        new_row = {
            'image_name': img_name,
            'predicted_cars': cars,
            'predicted_cars_parking': '',
            'processing_time': '-',
            }

        df.loc[len(df)] = new_row


        if 'no' in save:
            image_with_blank_space = cars
        else:
            for i, result in enumerate(results):
                if result.boxes is not None:
                    # Get the annotated image
                    annotated_image = result.plot(line_width=1)
                    
                    # Convert BGR to RGB
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                    # Create a blank space at the top of the image (white background)
                    blank_space_height = int(annotated_image.shape[0] * 0.1)
                    blank_space = 255 * np.ones((blank_space_height, annotated_image.shape[1], 3), dtype=np.uint8)

                    # Concatenate the blank space and the original image
                    image_with_blank_space = np.vstack((blank_space, annotated_image))

                    # Define your text and font settings
                    text = text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    
                    # Get the text size (width and height)
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_width, text_height = text_size

                    # Adjust text position to center it horizontally in the blank space
                    text_x = (image_with_blank_space.shape[1] - text_width) // 2
                    text_y = (blank_space_height + text_height) // 2

                    # Add text to the blank space
                    cv2.putText(image_with_blank_space, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                    # Save the image with the blank space and text
                    img_file_name = 'results' + os.path.splitext(img_name)[0] + ".jpg"
                    output_file_path = os.path.join(output_path, img_file_name)
                    cv2.imwrite(output_file_path, cv2.cvtColor(image_with_blank_space, cv2.COLOR_RGB2BGR))

        return cars, image_with_blank_space,df,elapsed_time,start_time_img,end_time_img




def perform_inference(input_path, output_path, img_,img_object, model,df, save=False,remove_txt=False,generate_txt=True):

    start_time_img = time.time()
    results = model.predict(img_object)
    end_time_img = time.time()
    elapsed_time = end_time_img - start_time_img

    print(f'image {img_}')
    print(f"Elapsed time to process: {elapsed_time} seconds")
    os.makedirs(output_path, exist_ok=True)

    # Show the results
    for r in results:
        print(f'\n results: \n {r}')
        img_name = img_.split('/')[-1][:-4]

        if generate_txt == True:
            txt_file_path = os.path.join(output_path, f'{img_name}_txt_file.txt')
            # Delete older version of the text file if it exists
            if os.path.exists(txt_file_path):
                os.remove(txt_file_path)
            r.save_txt(txt_file_path, save_conf=True)

            class_names_dict = r.names

            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r') as file:
                    lines = file.readlines()
            else:
                # if no car is detected, produce empty line
                lines = []        

            # cars = count_cars(lines,class_names_dict)
        else:


            # get the model names list
            names = model.names
            # get the 'car' class id
            car_id = list(names)[list(names.values()).index('car')]
            truck_id = list(names)[list(names.values()).index('truck')]
            # count 'car' objects in the results
            boxes = results[0].boxes
            car_objects = results[0].boxes.cls.tolist().count(car_id)
            truck_objects = results[0].boxes.cls.tolist().count(truck_id)
            print(f'\n{boxes}, \n{car_objects} \n\n {truck_objects}')
            cars = car_objects + truck_objects

        print(f"\n\n {cars} detected")


        text,cam = check_camera_type(img_, cars,img_name,language='eng')

        # Output the count to a new text file
        img_name = img_.split('/')[-1][:-4]

        # get metric
        new_row = {
            'image_name': img_name,
            'predicted_cars': cars,
            'predicted_cars_parking': '',
            'processing_time': '-',
            }

        df.loc[len(df)] = new_row



        if 'no' in save:
            image_with_blank_space = cars
        else:
            for i, result in enumerate(results):
                if result.boxes is not None:
                    # Get the annotated image
                    annotated_image = result.plot(line_width=1)
                    
                    # Convert BGR to RGB
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                    # Create a blank space at the top of the image (white background)
                    blank_space_height = int(annotated_image.shape[0] * 0.1)
                    blank_space = 255 * np.ones((blank_space_height, annotated_image.shape[1], 3), dtype=np.uint8)

                    # Concatenate the blank space and the original image
                    image_with_blank_space = np.vstack((blank_space, annotated_image))

                    # Define your text and font settings
                    text = text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    
                    # Get the text size (width and height)
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_width, text_height = text_size

                    # Adjust text position to center it horizontally in the blank space
                    text_x = (image_with_blank_space.shape[1] - text_width) // 2
                    text_y = (blank_space_height + text_height) // 2

                    # Add text to the blank space
                    cv2.putText(image_with_blank_space, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                    # Save the image with the blank space and text
                    img_file_name = 'results' + os.path.splitext(img_name)[0] + ".jpg"
                    output_file_path = os.path.join(output_path, img_file_name)
                    cv2.imwrite(output_file_path, cv2.cvtColor(image_with_blank_space, cv2.COLOR_RGB2BGR))

        return cars, image_with_blank_space,df,elapsed_time,start_time_img,end_time_img



def perform_inference_refact(img_object, model, img_file, output_dir="output",save='no'):
    os.makedirs(output_dir, exist_ok=True)

    # always not saving txt and image as it is done in a custom way
    all_results = model.predict(
        source = img_object,
        save_txt=False,
        save=False,
        classes=[2,7],
        line_width=2
    )

    image_annotations = []
    img_width, img_height = img_object.size

    for i, result in enumerate(all_results):
        if result.boxes is not None:
            for box in result.boxes:
                image_annotations.append([
                    int(box.cls),  # class ID
                    float(box.xywh[0][0] / img_width),  # normalized x center
                    float(box.xywh[0][1] / img_height),  # normalized y center
                    float(box.xywh[0][2] / img_width),  # normalized width
                    float(box.xywh[0][3] / img_height)  # normalized height
                ])

        if ('debug' in save or 'minimal' in save):
            # Annotate the image with its annotations
            if result.boxes is not None:
                annotated_image = result.plot(line_width=2)
                # Convert BGR to RGB
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                annotated_image = Image.fromarray(annotated_image)
    
    if ('no' in save or 'minimal' in save):
        print("No txt file and images will be saved")
    else:
        # Save image annotations
        label_file = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}.txt")
        with open(label_file, 'w') as f:
            for anno in image_annotations:
                f.write(' '.join(map(str, anno)) + '\n')
    
    if ('debug' in save or 'minimal' in save):
        # Save image with annotations
        img_file_name = 'results' + os.path.splitext(img_file)[0] + ".jpg"
        annotated_image.save(os.path.join(output_dir,img_file_name))
    else:
        annotated_image = 'no image'
        print("No image will be saved")
    
    cars = int(len(image_annotations))
    text,cam,preench = check_camera_type(img_file, cars)
    # Return the number of cars, image annotated or not, percentage of filling and annotations
    return cars,annotated_image, preench,image_annotations

def remove_txt_files(directory_path):
    try:
        # Get the list of all files in the directory
        files = os.listdir(directory_path)

        # Iterate through the files and remove .txt files
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(directory_path, file)
                os.remove(file_path)

        print("Text files removed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def divide_list_into_parts(input_list, num_parts):
    """
    Divide a list into a specified number of parts.

    Parameters:
    - input_list (list): The list to be divided.
    - num_parts (int): The number of parts to divide the list into.

    Returns:
    - divided_parts (list of lists): The divided parts of the list.
    """
    # Calculate the number of elements per part
    elements_per_part = len(input_list) // num_parts
    divided_parts = []

    # Divide the list into parts
    for i in range(num_parts):
        start_index = i * elements_per_part
        end_index = (i + 1) * elements_per_part if i < num_parts - 1 else None
        divided_parts.append(input_list[start_index:end_index])

    return divided_parts
