# YOLOV3 Darknet Pytorch Inference

## Overview

This repository provides a script for running inference using a YOLO model in TensorFlow Lite format. The model is pre-trained on the COCO dataset with an input shape of 416x416. The environment setup follows an unique setup as this codebase was extracted from [A minimal PyTorch implementation of YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

More info of this version at https://pjreddie.com/darknet/yolo/ and https://smartcampus.prefeitura.unicamp.br/pub/artigos_relatorios/PFG_Joao_Victor_Estacionamento_Inteligente.pdf


---

## Requirements
### Python & Virtual Environment
Ensure that Python is installed. The script has been tested with Python 3.11.7 and pip 23.2.1. Using a virtual environment is recommended to manage dependencies and prevent conflicts.

### Setting Up a Virtual Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv yolov3env
   ```

2. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     yolov3env\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source yolov3env/bin/activate
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
- **`mask_original_img_416_416_bw.png`**: A predefined mask that allows object counting in specific regions for IC2 images.
- **`cnrpark_mask_resized_img_416_416_bw.png`**: A mask designed for the **CNRPark dataset**, with specific dimensions.
- **`all_black_mask.png`**: A fully black mask that allows counting **all** cars in the image, with no region restrictions.

### How Mask Files Work:
The mask file is a **grayscale image** where:
- **Black areas**: These regions are considered for counting.
- **White areas**: These regions are ignored.

If you want to count **all** vehicles in the image, use **`all_black_mask.png`**.

### Modifying the Mask:
You can edit the mask using tools like **GIMP** or **Photoshop** to define custom counting areas. Simply paint the regions where objects should be counted in **black**.

You need to have an **YOLOV3 Model**: The model used for car detection.  
  - 📥 **Download `yolov3.weights` Pre-Trained model from OSF**: [Link](https://osf.io/yq6cn/)
---

## Script Details

### Main Script: `inference_yolov3.py`
This script processes images using a YOLO TFLite model and outputs results in a CSV file.

**Example demo images**: Available in the [docs folder](../../assets/demo_images), extracted from [CNRPark](http://cnrpark.it/). The dataset includes images in shape 728x1024, but other shapes should also be compatible.

### Running the Script

Run the script using the following command:
```bash
python3 inference_yolov3.py
```

#### Manual Configuration Required Dependng on User Needs
Before running the script, ensure that the following variables in the script are correctly set according to your needs:

```python
model_path = "yolov3.cfg"
weights_path = "yolov3.weights"
model = models.load_model(model_path, weights_path)
image_dir = '../../assets/demo_images'
output_dir = '../../assets/results/results_yolov3/yolov3'
savefigs = 'debug'
mask_file = 'cnrpark_mask_resized_img_416_416_bw.png'
```

Make sure `MODEL`, `image_dir`, `mask_file` and `output_dir` are correctly defined before executing the script. The `savefigs` variable should be set to `'debug'` if you want to save images or `'no'` if you prefer not to save them. 



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
