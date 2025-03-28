# YOLO TFLite Inference

## Overview

This repository provides a script for running inference using a YOLO model in TensorFlow Lite format. The model is pre-trained on the COCO dataset with an input shape of 640x640. The environment setup follows the same requirements as the EfficientDet models.

---

## Requirements
### Python & Virtual Environment
Ensure that Python is installed. The script has been tested with Python 3.11.7 and pip 23.2.1. Using a virtual environment is recommended to manage dependencies and prevent conflicts.

### Setting Up a Virtual Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv tfliteenv
   ```

2. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     tfliteenv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source tfliteenv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivate the virtual environment**:
   ```bash
   deactivate
   ```

---

## Required Files

Before running the script, ensure you have the correct **mask file** to define the counting regions

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

You need to have an **yolo11m (TFLite) Model**: The model used for car detection.  
  - 📥 **Download `yolo11m_float16.tflite` Pre-Trained model from our Google Drive Folder**: [Link](https://drive.google.com/drive/folders/1D_88IY0JBwUdi3EKsSAzLj1hxN6SJGit?usp=sharing) . There is also an option to export yourself the model. Details in the end of this README

---

## Script Details

### Main Script: `inference_yolo_tflite.py`
This script processes images using a YOLO TFLite model and outputs results in a CSV file.

**Example demo images**: Available in the [docs folder](../../assets/demo_images), extracted from [CNRPark](http://cnrpark.it/). The dataset includes images in shape 728x1024, but other shapes should also be compatible.

### Running the Script

Run the script using the following command:
```bash
python3 inference_yolo_tflite.py
```

#### Manual Configuration Required Dependng on User Needs
Before running the script, ensure that the following variables in the script are correctly set according to your needs:

```python
MODEL = 'yolo11m_float16.tflite'
IMAGE_DIR = '../../assets/demo_images'
OUTPUT_DIR = '../../assets/results/results_yolo_tflite/yolov11m_tflite'
savefigs = 'debug'  
mask_file= 'cnrpark_mask_original_img_1000_750_bw.png' 
```

Make sure `MODEL`, `IMAGE_DIR`, `mask_file` and `OUTPUT_DIR` are correctly defined before executing the script. The `savefigs` variable should be set to `'debug'` if you want to save images or `'no'` if you prefer not to save them. 



## Output Structure

```
output_path/
│
├── batch_<timestamp>/  # New batch directory for each run
│   ├── df_individual_metrics_<batch_number>_<timestamp>.csv  # Per-image detection results
│   ├── annotated_image.jpg/  # Processed images with bounding boxes (if enabled)
```
---

## Benchmarking

To evaluate model performance, use the benchmarking script in [software/benchmarks](../benchmarks/README.md).

---


### Exporting YOLO Model to TFLite  

The script `export_tflite.py` loads a pre-trained **YOLO** model and exports it to the **TFLite** format.  

#### How the Script Works:
1. **Imports Required Libraries**  
   - `ultralytics`: Used to load and export the YOLO model, used version Ultralytics 8.3.2 🚀 Python-3.11.7 torch-2.3.0

2. **Defines Configuration (`CFG` Class)**  
   - Specifies the base YOLO model (`yolo11n`, `yolo11m`, `yolov8n`, etc.).  
   - Sets the model weight file (`yolo11m.pt` in this case).  

3. **Loads the YOLO Model**  
   - The script initializes a YOLO model with the specified weights.  

4. **Exports the Model to TFLite**  
   - Converts the model into **TFLite** format for deployment on mobile and edge devices.  
   - Saves the exported model in a new folder:  
   yolo11n_saved_model/ ├── assets ├── fingerprint.pb ├── metadata.yaml ├── saved_model.pb ├── variables │ ├── variables.data-00000-of-00001 │ └── variables.index ├── yolo11n_float16.tflite └── yolo11n_float32.tflite


#### Floating-Point Precision Options:
- **Float32 (`yolo11n_float32.tflite`)**: Standard 32-bit precision.  
- **Float16 (`yolo11n_float16.tflite`)**: Lower memory usage, faster inference.  

The float16 model is preferred for deployment on resource-constrained devices. 🚀  
