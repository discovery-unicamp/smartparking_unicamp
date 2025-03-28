# YOLO Ultralytics Inference


The files here correspond to performing benchmarks with different deep learning models and measure inference time, as long as machine learning metrics for YOLOv8 to YOLOv11, as reported in the paper available at [arxiv](https://arxiv.org/abs/2412.01983) cited by: 

```
@article{da2024smart,
  title={Smart Parking with Pixel-Wise ROI Selection for Vehicle Detection Using YOLOv8, YOLOv9, YOLOv10, and YOLOv11},
  author={da Luz, Gustavo PCP and Sato, Gabriel Massuyoshi and Gonzalez, Luis Fernando Gomez and Borin, Juliana Freitag},
  journal={arXiv preprint arXiv:2412.01983},
  year={2024}
}

```

---


## Requirements
### Python & Virtual Environment

Before running the script, ensure that Python is installed on your system. The script has been tested with Python 3.11.7 and pip 23.2.1. It is recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python packages on your system.

### Creating a Virtual Environment

To create and activate a virtual environment, follow these steps:

1. **Create the virtual environment**:  
   In the terminal, navigate to the project directory and run:
   ```bash
   python -m venv yoloultralyticsenv
   ```
   This will create a directory named `yoloultralyticsenv` in your project directory, which will contain the isolated Python environment.

2. **Activate the virtual environment**:
   - **macOS/Linux**:
     ```bash
     source yoloultralyticsenv/bin/activate
     ```

   Once activated, the terminal prompt should change to indicate that the virtual environment is active, e.g., `(yoloultralyticsenv)`.

3. **Install required libraries**:  
   With the virtual environment active, run the following command to install all necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   This will install the required libraries

4. **Deactivate the virtual environment**:  
   After you're done working, you can deactivate the virtual environment by running:
   ```bash
   deactivate
   ```

### Required File

### Available Mask Files:
- **`mask_original_img_768_1024_bw.png`**: A predefined mask that allows object counting in specific regions for IC2 images.
- **`cnrpark_mask_original_img_1000_750_bw.png`**: A mask designed for the **CNRPark dataset**, with specific dimensions.
- **`all_black_mask.png`**: A fully black mask that allows counting **all** cars in the image, with no region restrictions.

### How Mask Files Work:
The mask file is a **grayscale image** where:
- **Black areas**: These regions are considered for counting.
- **White areas**: These regions are ignored.

If you want to count **all** vehicles in the image, use **`all_black_mask.png`**.

### Modifying the Mask:
You can edit the mask using tools like **GIMP** or **Photoshop** to define custom counting areas. Simply paint the regions where objects should be counted in **black**.

---

## Files

1. **inference_yolo_ultralytics.py**  
   This script processes images, runs inference using a YOLO model, and returns results in a csv.

   Example test images can be found on [docs folder](../../assets/demo_images) extracted from [CNRPark](http://cnrpark.it/) named CNR-EXT_FULL_IMAGE_1000x750.tar (1.1 GB) and are from the under CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY/2015-11-12/camera1. The images used for the reference paper are in shape 728x1024 but other shapes should also be compatible.


## Running the Script

To execute the script, you need to provide some command-line arguments. Here’s a summary of the required arguments:

- `input_path`: Path to the directory containing the images to be processed.
- `output_path`: Path to the directory where results (such as processed images or result files) will be saved.
- `--savefigs`: Option to save figures. Available options:
  - `debug`: Saves all figures generated during the process.
  - `no`: Does not save any figures.
  - `partial`: Saves only essential figures.
- `--model`: YOLO model to be used, e.g., `yolov8n`, `yolov8x`, `yolov9t`, `yolov9x`, `yolov10n`, `yolov10x`,`yolo11n`,`yolo11x`. The model will be automatically downloaded. You can also **Downloadsome  Pre-Trained model from our Google Drive Folder**: [Link](https://drive.google.com/drive/folders/1D_88IY0JBwUdi3EKsSAzLj1hxN6SJGit?usp=sharing)  
- `mask_file`: Selected desired mask file. Default is all black, or not use a mask.

### Example usage:


```
python3 inference_yolo_ultralytics.py --input_path ../../assets/demo_images --output_path ../../assets/results/results_yolo_ultralytics/yolov8n --savefigs debug --model yolov8n --mask_file cnrpark_mask_original_img_1000_750_bw.png
```

---
  

### Output Directory Structure

When you run the script, it creates an output directory (specified by `--output_path`). Inside this directory, you will find:

```
output_path/
│
├── batch_<timestamp>/  # A new batch directory is created for each run
│   ├── df_individual_metrics_<batch_number>_<timestamp>.csv  # Metrics for each processed image
│   ├── df_full_metrics.csv  # Overall metrics for the batch
│   ├── results/  # Processed images with bounding boxes (if savefigs is enabled)
│   ├── images/  # Original images (if configured)
│
```

### Description of Output Files

`batch_<timestamp>/`

Each time the script is executed, a new batch directory is created. The `<timestamp>` is based on the execution time to ensure uniqueness.

- **`df_individual_metrics_<batch_number>_<timestamp>.csv`**: This file contains per-image detection results, including:
  - Image name
  - Timestamp
  - Predicted number of cars
  - Inference time
  - Blank fields for metrics


- **`df_full_metrics.csv`**: This file aggregates the results from all processed images in the batch, including:
  - Number of images processed
  - Average processing time per image

### `results/`
If saving figures is enabled (`--savefigs debug` or `--savefigs partial`), this directory contains processed images with bounding boxes drawn over detected objects.

---

2. **bench.sh.py**  
   This shell script is just and easy way to run multiple tests at once.

### Metrics: 

To compute the result you will need the standardized script that can be found at [software/benchmarks](../benchmarks/README.md)

---
