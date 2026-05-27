import os
import re
from datetime import datetime, timedelta
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "Ultralytics YOLO package is not installed. Please install it to use YOLO functionalities."
    )
import time
import cv2
import numpy as np
import json
import argparse

try:
    import imageio.v3 as iio
except ImportError:
    print(
        "imageio package is not installed. Please install it to use image reading functionalities."
    )
import psutil  # For system resource monitoring
import csv


def list_images(directory_path, extensions=None):
    """
    List images in the specified directory with optional extensions filter.

    Parameters:
    - directory_path (str): Path to the directory.
    - extensions (list): List of allowed file extensions. If None, all files are considered.

    Returns:
    - image_list (list): List of image filenames in the directory.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]

    # List all files in the directory
    all_files = os.listdir(directory_path)

    # Filter files based on extensions
    image_list = [
        file
        for file in all_files
        if any(file.lower().endswith(ext) for ext in extensions)
    ]

    return image_list


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
        extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]

    image_list = []
    # Loop through subdirectories
    for subdir in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdirectory_path):
            # List all files in the subdirectory
            all_files = os.listdir(subdirectory_path)
            # Filter files based on extensions
            image_list.extend(
                [
                    os.path.join(subdir, file)
                    for file in all_files
                    if any(file.lower().endswith(ext) for ext in extensions)
                ]
            )
            # image_list.extend([os.path.basename(file) for file in all_files if any(file.lower().endswith(ext) for ext in extensions)])

    return image_list


def extract_timestamp(filename, mode="filename"):
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
    if mode == "filename":
        timestamp_match = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", filename)
        if timestamp_match:
            return timestamp_match.group()
        else:
            return None
    elif mode == "modification_time":
        modification_time = os.path.getctime(filename)
        timestamp = datetime.fromtimestamp(modification_time).strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        return timestamp
    elif mode == "creation_time":
        creation_time = os.path.getmtime(filename)  # Use getmtime instead of getctime
        timestamp = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d-%H-%M-%S")
        return timestamp
    elif mode == "filename_reitoria":
        # match = re.search(
        #     r"cam_reitoria_(\d+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.jpg", filename
        # )
        match = re.search(
            r"cam_reitoria_(\d+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?: \([^)]*\))?\.jpg",
            filename
        )

        if match:
            return match.group(2)
        else:
            return None
    else:
        raise ValueError(
            "Invalid mode. Mode must be either 'filename' or 'modification_time' or 'creation_time'."
        )


def get_images_from_dir(directory):
    """
    Retrieves all image file names from a specified directory.

    Args:
        directory (str): Path to the directory containing image files.

    Returns:
        list: List of image file names with '.jpg' or '.png' extensions.
    """
    images = [
        f for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")
    ]
    return images


def check_image_sizes(images, directory):
    """
    Checks if all images in a directory have the same size.

    Args:
        images (list): List of image file names.
        directory (str): Path to the directory containing images.

    Returns:
        bool: True if all images have the same size, False otherwise.
    """
    sizes = []
    for image in images:
        img = Image.open(os.path.join(directory, image))
        sizes.append(img.size)
    print(sizes)
    return len(set(sizes)) == 1


def divide_and_save_images(images, input_directory, output_dir):
    """
    Divides each image in the input directory into upper and lower halves
    and saves them in the output directory.

    Args:
        images (list): List of image file names.
        input_directory (str): Path to the directory containing original images.
        output_dir (str): Path to the directory to save cropped images.

    Returns:
        None
    """
    for image in images:
        img = Image.open(os.path.join(input_directory, image))
        width, height = img.size

        # Define the crop size
        upper_height = int(height * 0.5)
        lower_height = height - upper_height

        # Define the coordinates for the upper half
        upper_coords = (0, 0, width, upper_height)
        upper_half = img.crop(upper_coords)

        upper_half.save(os.path.join(output_dir, "upper_" + image))

        # Define the coordinates for the lower half
        lower_coords = (0, upper_height, width, height)
        lower_half = img.crop(lower_coords)
        lower_half.save(os.path.join(output_dir, "lower_" + image))


def process_images(directory, output_dir):
    """
    Processes images in a directory by verifying uniform sizes and
    dividing them into halves if sizes match.

    Args:
        directory (str): Path to the directory containing original images.
        output_dir (str): Path to the directory to save processed images.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    images = get_images_from_dir(directory)
    if check_image_sizes(images, directory):
        divide_and_save_images(images, directory, output_dir)
    else:
        print("Not all images have the same size.")


def plot_images(images, text):
    """
    Plot a grid of images.

    Parameters:
    - images (list): List of PIL Image objects.
    """
    matplotlib.use("Agg")
    # Ensure we have at least 4 images
    while len(images) < 4:
        images.append(
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        )  # Add blank images if not enough

    # Create a 2x2 grid for the subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten the array of axes for easier iteration
    axs = axs.flatten()

    # Plot each image in a subplot
    for i, image in enumerate(images):
        axs[i].imshow(image)

        axs[i].axis("off")  # Hide the axis
    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    fig.tight_layout()

    # Add text on top of the entire plot
    plt.figtext(
        0.5,
        0.98,
        text,
        ha="center",
        va="top",
        fontsize=12,
        color="black",
        fontweight="bold",
    )

    plt.subplots_adjust(top=0.95)

    return fig


def categorize_parking_status(occupancy_percentage):
    """
    Categorizes the parking lot's status based on occupancy percentage.

    Args:
        occupancy_percentage (float): Percentage of occupied parking spaces.

    Returns:
        str: Parking status ('Vazio', 'Médio', 'Cheio', 'Muito Cheio').
    """
    if occupancy_percentage <= 40:
        return "Vazio"
    elif 40 <= occupancy_percentage <= 65:
        return "Médio"
    elif 65 <= occupancy_percentage <= 80:
        return "Cheio"
    else:
        return "Muito Cheio"


def is_within_time_range(timestamp, start_time, end_time):
    """
    Checks if a timestamp is within a specified time range.

    Args:
        timestamp (str): Timestamp in the format '%Y-%m-%d-%H-%M-%S'.
        start_time (str): Start time in the format '%H-%M-%S'.
        end_time (str): End time in the format '%H-%M-%S'.

    Returns:
        bool: True if the timestamp is within the range, False otherwise.
    """
    target_format = "%H-%M-%S"
    timestamp_datetime = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
    timestamp_datetime_converted = timestamp_datetime.strftime(target_format)
    print(timestamp_datetime_converted)
    start_time_datetime = datetime.strptime(start_time, target_format).strftime(
        target_format
    )
    end_time_datetime = datetime.strptime(end_time, target_format).strftime(
        target_format
    )
    print(
        f"original: {timestamp_datetime}, converted: {timestamp_datetime_converted}, start time: {start_time_datetime}, end time:{end_time_datetime}"
    )
    return start_time_datetime <= timestamp_datetime_converted <= end_time_datetime


def is_timestamp_valid(image_timestamp, most_recent_timestamp, desired_range):
    """
    Validates if the time difference between two timestamps is within a desired range.

    Args:
        image_timestamp (str): Timestamp of the image in '%Y-%m-%d-%H-%M-%S'.
        most_recent_timestamp (str): Most recent timestamp in the same format.
        desired_range (timedelta): Allowed range for time difference.

    Returns:
        bool: True if the time difference is within the desired range, False otherwise.
    """
    # Convert timestamp strings to datetime objects
    image_timestamp_dt = datetime.strptime(image_timestamp, "%Y-%m-%d-%H-%M-%S")
    most_recent_timestamp_dt = datetime.strptime(
        most_recent_timestamp, "%Y-%m-%d-%H-%M-%S"
    )
    print(image_timestamp, most_recent_timestamp)
    # Calculate the time difference
    time_difference = abs(image_timestamp_dt - most_recent_timestamp_dt)

    # Check if the time difference is within the desired range
    if time_difference <= desired_range:
        return True
    else:
        return False


def count_cars(lines, class_names_dict):
    """
    Counts the number of cars and trucks in a detection result.

    Args:
        lines (list): List of detection results containing class indices.
        class_names_dict (dict): Mapping of class indices to class names.

    Returns:
        int: Total count of cars and trucks.
    """
    # Initialize counters
    car_count = 0
    truck_count = 0

    # Iterate through lines and count cars and trucks
    for line in lines:
        class_index, *_ = line.split()
        class_name = class_names_dict[int(class_index)]

        if class_name == "car":
            car_count += 1
        elif class_name == "truck":
            truck_count += 1

    return car_count + truck_count


def padronize_filename(img_, timestamp):
    """
    Standardizes the filename of an image based on the camera identifier and timestamp.

    Args:
        img_ (str): The image filename which contains the camera identifier.
        timestamp (str): A timestamp string to be included in the new filename.

    Returns:
        str: A standardized filename in the format 'cam_reitoria_<camera_id>-<timestamp>.jpg'.
    """
    print(img_)
    if "cam1" in img_:
        txt_cam = "cam_reitoria_1"

    elif "cam2" in img_:
        txt_cam = "cam_reitoria_2"

    elif "cam3" in img_:
        txt_cam = "cam_reitoria_3"

    elif "cam4" in img_:
        txt_cam = "cam_reitoria_4"

    img_ = f"{txt_cam}-{timestamp}.jpg"
    print(img_)
    return img_


def check_camera_type(img_, cars):
    """
    Determines the camera type and parking lot occupancy based on the number of detected cars.

    Args:
        img_ (str): The image filename which contains the camera identifier.
        cars (int): The number of cars detected in the image.

    Returns:
        tuple: A tuple containing a message about the detected cars, camera type, and occupancy percentage:
            - str: A message indicating the number of cars detected, the camera type, and parking status.
            - int: The camera type (1, 2, 3, or 4).
            - float: The occupancy percentage of the parking lot.
    """

    if "cam_reitoria_1" in img_:
        cam = 1
        preench = round((cars) / 68 * 100, 2)
        status = categorize_parking_status(preench)

    elif "cam_reitoria_2" in img_:
        cam = 2
        preench = round((cars) / 18 * 100, 2)
        status = categorize_parking_status(preench)

    elif "cam_reitoria_3" in img_:
        cam = 3
        preench = round((cars) / 31 * 100, 2)
        status = categorize_parking_status(preench)

    elif "cam_reitoria_4" in img_:
        cam = 4
        preench = round((cars) / 44 * 100, 2)
        status = categorize_parking_status(preench)

    text = f"Número de carros detectados: {cars} - Câmera {cam} - {status}, {preench}% de ocupação"
    print(text)
    return text, cam, preench


def resize_and_divide_image(img_object, block_size=(640, 640)):
    """
    Resize an image to a target size and divide it into smaller blocks.

    Parameters:
    - img_object (PIL.Image): The input image to be resized and divided.
    - block_size (tuple): The size of each block (width, height) in pixels. Default is (640, 640).

    Returns:
    - blocks (dict): A dictionary where the keys are (x, y) coordinates of the top-left corner of each block,
                     and the values are the corresponding cropped image blocks (PIL.Image).
    - resized_image (PIL.Image): The resized image after scaling to the target size.
    """
    target_size = (1920, 1280)
    resized_image = img_object.resize(target_size, Image.Resampling.LANCZOS)
    width, height = resized_image.size

    blocks = {}
    for i in range(0, width, block_size[0]):
        for j in range(0, height, block_size[1]):
            box = (i, j, i + block_size[0], j + block_size[1])
            block = resized_image.crop(box)
            blocks[(i, j)] = block

    return blocks, resized_image


def adjust_annotations_for_full_image(
    annotations, block_position, block_size, img_size
):
    """
    Adjust bounding box annotations from block coordinates to normalized full image coordinates.

    Parameters:
    - annotations (list): List of bounding box annotations in block coordinates.
                           Format: [[class_id, center_x, center_y, width, height], ...].
    - block_position (tuple): (x, y) position of the block's top-left corner in the full image.
    - block_size (tuple): (width, height) dimensions of the block.
    - img_size (tuple): (width, height) dimensions of the full image.

    Returns:
    - adjusted_annotations (list): List of annotations adjusted to normalized full image coordinates.
                                    Format: [[class_id, norm_center_x, norm_center_y, norm_width, norm_height], ...].
    """
    block_x, block_y = block_position
    block_width, block_height = block_size
    img_width, img_height = img_size

    adjusted_annotations = []
    for anno in annotations:
        class_id, center_x, center_y, width, height = anno
        # Convert to absolute coordinates
        center_x = center_x * block_width + block_x
        center_y = center_y * block_height + block_y
        width = width * block_width
        height = height * block_height

        # Normalize for full image
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height

        adjusted_annotations.append([class_id, center_x, center_y, width, height])

    return adjusted_annotations


def apply_blur_to_boxes(image, boxes, block_position, block_size):
    """
    Apply a blur effect to specified bounding boxes in the image.

    Parameters:
    - image (PIL.Image): The image where the boxes are located.
    - boxes (list): List of bounding box coordinates [(x1, y1, x2, y2), ...].
    - block_position (tuple): (x, y) position of the block in the original image.
    - block_size (tuple): (width, height) of the block.

    Returns:
    - PIL.Image: Image with blurred boxes applied.
    """
    image_array = np.array(image)

    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # Extract the region to blur
        region = image_array[y1:y2, x1:x2]
        # Apply Gaussian blur
        blurred_region = cv2.GaussianBlur(region, (31, 31), 70)
        # Replace the original region with the blurred one
        image_array[y1:y2, x1:x2] = blurred_region

    # Convert back to PIL Image
    return Image.fromarray(image_array)


def apply_blur_to_image(image, blur_boxes=None, kernel_size=(31, 31), sigma=70):
    """
    Apply a Gaussian blur to the entire image or specified bounding boxes.

    Parameters:
    - image (PIL.Image): Input image.
    - blur_boxes (list): List of bounding box coordinates [(x1, y1, x2, y2), ...].
    - kernel_size (tuple): Kernel size for Gaussian blur.
    - sigma (float): Standard deviation for Gaussian blur.

    Returns:
    - PIL.Image: Blurred image.
    """
    image_array = np.array(image)

    if blur_boxes is None:
        # Apply blur to the entire image
        blurred_image_array = cv2.GaussianBlur(image_array, kernel_size, sigma)
    else:
        blurred_image_array = image_array.copy()
        for box in blur_boxes:
            x1, y1, x2, y2 = map(int, box)
            region = image_array[y1:y2, x1:x2]
            blurred_region = cv2.GaussianBlur(region, kernel_size, sigma)
            blurred_image_array[y1:y2, x1:x2] = blurred_region

    return Image.fromarray(blurred_image_array)


def perform_inference_blocks(
    img_object, model, img_file, df, output_dir="output", save="no"
):
    """
    Divides an image into blocks, performs inference on each block using a model, and saves the results.

    This function resizes the input image, divides it into blocks, performs inference on each block,
    and saves the results (annotations and/or images) based on the specified `save` option.
    It also returns information about the number of detected cars, the full annotated image,
    the parking occupancy percentage, and the image annotations.

    Args:
        img_object (PIL.Image): The input image to be processed.
        model (YOLO model): The YOLO model to be used for inference.
        img_file (str): The file path of the image being processed.
        df (pandas.DataFrame): A DataFrame that may be used for additional processing (not used in this function).
        output_dir (str, optional): The directory where the results (images and annotations) will be saved. Defaults to "output".
        save (str, optional): Determines the level of saving. Can be 'no', 'minimal', or 'debug'.
                              'no' means no saving, 'minimal' saves only essential files, 'debug' saves additional debugging files. Defaults to 'no'.

    Returns:
        tuple: A tuple containing:
            - int: The number of cars detected in the image (sum of cars and trucks).
            - PIL.Image: The full annotated image.
            - float: The parking lot occupancy percentage based on detected cars.
            - list: A list of annotations for each object detected in the full image,
                    with each annotation containing class ID, normalized coordinates, and dimensions.
    """
    blocks, full_image = resize_and_divide_image(img_object)
    # Save resized image
    os.makedirs(output_dir, exist_ok=True)
    # img_file_name = os.path.splitext(img_file)[0] + ".jpg"
    img_file_name = img_file.split("/")[-1]
    print(img_file_name)
    if "debug" in save:
        # save the resized input image only in debug mode
        full_image.save(os.path.join(output_dir, "resized_" + img_file_name))

    # Perform inference on blocks

    if "no" in save or "minimal" in save:
        try:
            all_results = model.predict(
                source=list(blocks.values()),
                save_txt=False,
                save=False,
                classes=[2, 7, 0],  # 0 person 2 cars 7 truck
                line_width=3,
            )
        except:
            all_results = []
            for block in blocks.values():
                result = model.predict(block, save_txt=False, save=False, classes=[0])
                all_results.append(result[0])
    else:
        all_results = model.predict(
            source=list(blocks.values()),
            save_txt=True,
            save=True,
            classes=[2, 7, 0],
            line_width=3,
            project=output_dir,  # directory to save
            name="annot",  # subdirectory to save
        )

    full_image_annotations = []

    for i, result in enumerate(all_results):
        block_position = list(blocks.keys())[i]
        block_width, block_height = 640, 640
        img_size = (1920, 1280)

        # Extract block annotations
        block_annotations = []
        if result.boxes is not None:
            for box in result.boxes:
                block_annotations.append(
                    [
                        int(box.cls),  # class ID
                        float(box.xywh[0][0] / block_width),  # normalized x center
                        float(box.xywh[0][1] / block_height),  # normalized y center
                        float(box.xywh[0][2] / block_width),  # normalized width
                        float(box.xywh[0][3] / block_height),  # normalized height
                    ]
                )

        # Adjust annotations for full image
        adjusted_annotations = adjust_annotations_for_full_image(
            block_annotations, block_position, (block_width, block_height), img_size
        )
        full_image_annotations.extend(adjusted_annotations)

        if "debug" in save or "minimal" in save:

            if result.boxes is not None:
                block_boxes = [
                    (
                        int(box.xyxy[0][0]),
                        int(box.xyxy[0][1]),
                        int(box.xyxy[0][2]),
                        int(box.xyxy[0][3]),
                    )
                    for box in result.boxes
                    if int(box.cls) in [2, 7, 0]
                ]
                # annotated_block = result.plot(conf=False,line_width=2,labels=False)
                annotated_block = result.plot(conf=False, line_width=3, labels=False)
                annotated_block = cv2.cvtColor(annotated_block, cv2.COLOR_BGR2RGB)
                annotated_block = Image.fromarray(annotated_block)
                # Blur detected persons in the current block
                # annotated_block = apply_blur_to_boxes(
                #     annotated_block,
                #     block_boxes,
                #     block_position,
                #     (block_width, block_height),
                # )
                full_image.paste(annotated_block, block_position)

    if "no" in save or "minimal" in save:
        print("No txt file and images will be saved")
    else:

        label_file = os.path.join(output_dir, f"{img_file.split('/')[-1][:-4]}.txt")

        print(f"\n\n\nlabel file: {label_file}")
        with open(label_file, "w") as f:
            for anno in full_image_annotations:
                f.write(" ".join(map(str, anno)) + "\n")

    if "debug" in save or "minimal" in save:

        img_file_name = "results" + img_file.split("/")[-1]
        print(f"\n\n\nimg_file_name: {img_file_name}")
        # Save the full annotated image
        print(
            f"Saving full annotated image to {os.path.join(output_dir,img_file_name)}"
        )
        full_image.save(os.path.join(output_dir, img_file_name))
    else:
        print("No image will be saved")

    cars = len([anno for anno in full_image_annotations if anno[0] in [2, 7]])

    text, cam, preench = check_camera_type(img_file_name, cars)

    # Return the number of cars, image annotated or not, percentage of filling and annotations
    return cars, full_image, preench, full_image_annotations


def perform_inference(img_object, model, img_file, output_dir="output", save="no"):
    """
    Perform inference on an image and optionally blur detections or the entire image.

    Parameters:
    - img_object (PIL.Image): The input image object.
    - model: The detection model.
    - img_file (str): The input image file name.
    - output_dir (str): Directory to save outputs.
    - save (str): Save mode ('no', 'minimal', 'debug').

    Returns:
    - cars (int): Number of detected cars.
    - annotated_image (PIL.Image or str): Annotated image or 'no image'.
    - preench (float): Percentage of parking lot filling (if applicable).
    - image_annotations (list): List of annotations for detected objects.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Perform inference
    all_results = model.predict(
        source=img_object,
        save_txt=False,
        save=False,
        classes=[2, 7, 0],  # 0: person, 2: cars, 7: truck
        line_width=None,
    )

    image_annotations = []
    img_width, img_height = img_object.size

    for result in all_results:
        if result.boxes is not None:
            for box in result.boxes:
                image_annotations.append(
                    [
                        int(box.cls),  # class ID
                        float(box.xywh[0][0] / img_width),  # normalized x center
                        float(box.xywh[0][1] / img_height),  # normalized y center
                        float(box.xywh[0][2] / img_width),  # normalized width
                        float(box.xywh[0][3] / img_height),  # normalized height
                    ]
                )

    if "debug" in save or "minimal" in save:
        if all_results[0].boxes is not None:
            # Annotate the image
            # annotated_image = all_results[0].plot(line_width=2)
            annotated_image = all_results[0].plot(
                conf=False, line_width=3, labels=False
            )
            # ,color_mode="white"

            # Convert BGR to RGB
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(annotated_image)

            # Optionally blur specific regions (e.g., persons)
            block_boxes = [
                (
                    int(box.xyxy[0][0]),
                    int(box.xyxy[0][1]),
                    int(box.xyxy[0][2]),
                    int(box.xyxy[0][3]),
                )
                for box in all_results[0].boxes
                if int(box.cls) in [2, 7, 0]  # Class 0: person
            ]
            if block_boxes:
                # annotated_image = apply_blur_to_image(annotated_image, block_boxes)
                print("skipping blur")

            # Save annotated image
            img_file_name = "results_" + os.path.basename(img_file)
            annotated_image.save(os.path.join(output_dir, img_file_name))
        else:
            annotated_image = "no image"

    else:
        annotated_image = "no image"
        print("No image will be saved")

    # Save annotations to a file
    # if ('no' not in save):
    if "debug" in save:
        label_file = os.path.join(
            output_dir, f"{os.path.splitext(os.path.basename(img_file))[0]}.txt"
        )
        with open(label_file, "w") as f:
            for anno in image_annotations:
                f.write(" ".join(map(str, anno)) + "\n")

    cars = len(
        [anno for anno in image_annotations if anno[0] in [2, 7]]
    )  # Count detected cars (class ID = 2)
    text, cam, preench = check_camera_type(img_file, cars)

    return cars, annotated_image, preench, image_annotations


def detection_matrix_modified(x, y, mask):
    mask = mask[:, :, 0]
    print(mask.shape)
    mask_x, mask_y = mask.shape
    # mask_x,mask_y = mask.shape[:2]
    # mask_x, mask_y = (480,640)
    x = x * mask_y
    y = y * mask_x
    pixel_value = mask[int(y), int(x)]
    print(
        f"\n points are {x},{y} \n pixel value: {pixel_value} and mask shape is {mask.shape}\n mask_x = {mask_x}, mask_y = {mask_y}"
    )
    if pixel_value == 255:
        print("The point is outside the mask.")
        return False
    else:
        print("The point is inside the mask.")
        return True


def count_cars_post(lines, mask):
    # Initialize counters
    car_count = 0
    truck_count = 0

    # Iterate through lines and count cars and trucks
    for line in lines:

        line_ = np.array(line)
        class_name, x_center, y_center, width, height = line_
        print(
            f"\n class name: {class_name}, x_center: {x_center}, y_center: {y_center}, width: {width}, height: {height}"
        )

        if class_name == 2.0:
            # point_inside = detection_matrix(x_center,y_center,mask)
            point_inside = detection_matrix_modified(x_center, y_center, mask)
            print(f"\n\n\n\n point {x_center}, {y_center} is {point_inside} ")
            if point_inside == True:

                car_count += 1

        elif class_name == 7.0:
            # point_inside = detection_matrix(x_center,y_center,mask)
            point_inside = detection_matrix_modified(x_center, y_center, mask)
            print(f"\n\n\n\n point {x_center}, {y_center} is {point_inside} \n\n\n\n")
            if point_inside == True:
                truck_count += 1

    return car_count + truck_count


def perform_inference_post(
    img_object, model, img_file, mask, output_dir="output", save="no"
):
    """
    TODO add postprocessed docstring
    Perform inference on an image

    Parameters:
    - img_object (PIL.Image): The input image object.
    - model: The detection model.
    - img_file (str): The input image file name.
    - mask (numpy.ndarray): The mask to be applied to the image.
    - output_dir (str): Directory to save outputs.
    - save (str): Save mode ('no', 'minimal', 'debug').

    Returns:
    - cars (int): Number of detected cars.
    - annotated_image (PIL.Image or str): Annotated image or 'no image'.
    - preench (float): Percentage of parking lot filling (if applicable).
    - image_annotations (list): List of annotations for detected objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Perform inference
    all_results = model.predict(
        source=img_object,
        save_txt=False,
        save=False,
        classes=[2, 7, 0],  # 0: person, 2: cars, 7: truck
        line_width=None,
    )

    image_annotations = []
    img_width, img_height = img_object.size

    for result in all_results:
        if result.boxes is not None:
            for box in result.boxes:
                image_annotations.append(
                    [
                        int(box.cls),  # class ID
                        float(box.xywh[0][0] / img_width),  # normalized x center
                        float(box.xywh[0][1] / img_height),  # normalized y center
                        float(box.xywh[0][2] / img_width),  # normalized width
                        float(box.xywh[0][3] / img_height),  # normalized height
                    ]
                )

    if "debug" in save or "minimal" in save:
        if all_results[0].boxes is not None:
            # Annotate the image
            # annotated_image = all_results[0].plot(line_width=2)
            annotated_image = all_results[0].plot(
                conf=False, line_width=3, labels=False
            )

            # Convert BGR to RGB
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(annotated_image)

            # Optionally blur specific regions (e.g., persons)
            block_boxes = [
                (
                    int(box.xyxy[0][0]),
                    int(box.xyxy[0][1]),
                    int(box.xyxy[0][2]),
                    int(box.xyxy[0][3]),
                )
                for box in all_results[0].boxes
                if int(box.cls) in [2, 7, 0]  # Class 0: person
            ]
            if block_boxes:
                # annotated_image = apply_blur_to_image(annotated_image, block_boxes)
                print("skipping blur")
            # Save annotated image
            img_file_name = "results_" + img_file[:-4] + ".png"
            annotated_image.save(os.path.join(output_dir, img_file_name))
        else:
            annotated_image = "no image"

    else:
        annotated_image = "no image"
        print("No image will be saved")

    # Save annotations to a file
    if "no" not in save:
        label_file = os.path.join(
            output_dir, f"{os.path.splitext(os.path.basename(img_file))[0]}.txt"
        )
        with open(label_file, "w") as f:
            for anno in image_annotations:
                f.write(" ".join(map(str, anno)) + "\n")

    print(
        f"image annotations: {image_annotations}\n\n\n type: {type(image_annotations)}"
    )

    cars = count_cars_post(image_annotations, mask)

    text, cam, preench = check_camera_type(img_file, cars)

    return cars, annotated_image, preench, image_annotations


def get_system_metrics():
    """Get current system metrics (CPU, memory, swap)"""
    return {
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().used / (1024 * 1024),  # Convert to MB
        "swap": psutil.swap_memory().used / (1024 * 1024),  # Convert to MB
    }
