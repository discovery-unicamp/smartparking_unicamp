# Mask R-CNN TensorFlow 2.0 Inference

## Overview

This repository provides a script for running inference using a Mask R-CNN. The model is pre-trained on the COCO dataset with an input shape of 1024x1024. The environment setup follows an unique setup as this codebase was extracted from [Mask R-CNN for Object Detection and Segmentation using TensorFlow 2.0](https://github.com/ahmedfgad/Mask-RCNN-TF2/tree/master)

This was not possible to execute at the Raspberry Pi 3B+.

---

## Requirements
### Python & Virtual Environment
Ensure that Python is installed. The script has been tested with Python 3.6.13 and pip 21.2.2. Using a virtual environment is recommended to manage dependencies and prevent conflicts. To manage it easier, we used conda for it. If you are not fammiliar with conda read [Getting started with conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) first

### Setting Up a Virtual Environment

1. **Create a virtual environment**:
   ```bash
   conda create -n condamaskenv36 python=3.6.13
   ```

2. **Activate the virtual environment**:
     ```bash
     conda activate condamaskenv36
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivate the virtual environment**:
   ```bash
   conda deactivate
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

You need to have an **Mask R-CNN Model**: The model used for car detection.  
  - 📥 **Download `mask_rcnn_coco.h5` Pre-Trained model from OSF**: [Link](https://osf.io/yq6cn/)

---

## Script Details

### Main Script: `inference_maskrcnn.py`
This script processes images using a Mask R-CNN model and outputs results in a CSV file.

**Example demo images**: Available in the [docs folder](../../assets/demo_images), extracted from [CNRPark](http://cnrpark.it/). The dataset includes images in shape 728x1024, but other shapes should also be compatible.

### Running the Script

Run the script using the following command:
```bash
python3 inference_maskrcnn.py
```

#### Manual Configuration Required Dependng on User Needs
Before running the script, ensure that the following variables in the script are correctly set according to your needs:

```python
model = mrcnn.model.MaskRCNN(mode="inference", config=SimpleConfig(), model_dir=os.getcwd())
model.load_weights(filepath="mask_rcnn_coco.h5", by_name=True)
input_dir = '../../assets/demo_images'
OUTPUT_DIR = '../../assets/results/results_maskrcnn/markrcnn'
savefigs = 'debug' 
```

Make sure `MODEL`, `input_dir`, and `OUTPUT_DIR` are correctly defined before executing the script. The `savefigs` variable should be set to `'debug'` if you want to save images or `'no'` if you prefer not to save them.



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
