# CNRPark+EXT Label File Generation

## Overview
This script processes the **CNRPark+EXT** csv file and converts it into a **standard label file** names **`labels_cnrpark.csv`** used in our project. It extracts relevant information from the dataset and structures it into a CSV format that can be used for model training and evaluation.

## Mandatory Columns for a New Label File
If you want to build a new label file, ensure it contains the following necessary columns:
- **`image_name`**: The formatted image name. For CNRPark it is in `YYYY-MM-DD_HHMM` format.
- **`real_cars`**: The actual number of cars present in the image.

### Placeholders (Can Be Removed or Updated Later)
These columns are currently included as placeholders for further model evaluation to be developed but can be removed if unnecessary:
- **`predicted_cars`**: Placeholder for model-predicted car count.
- **`predicted_cars_parking`**: Placeholder for predicted cars in parking spots.
- **`start_time`, `end_time`, `processing_time`, `Timestamp`**: Timing placeholders.
- **`predicted_background`, `predicted_background_parking`, `real_background`**: Background object counts.
- **`TP, TN, FP, FN, accuracy`**: Performance evaluation placeholders.
- **`first_datetime`**: The first timestamp in the aggregated group.
- **`last_datetime`**: The last timestamp in the aggregated group.
- **`timestamp`**: The datetime of capture (from the dataset).

## Dataset Information
The images used in this project are **full-frame** images from the **CNR-EXT** subset. The images follow the naming convention:

FULL_IMAGE_1000x750/<WEATHER>/<CAPTURE_DATE>/camera<CAM_ID>/<CAPTURE_DATE>_<CAPTURE_TIME>.jpg

For this demonstration, the images were taken from:
CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY/2015-11-12/camera1

The camera in this subset has **35 parking spots**. This can be confirmed at **Figure 2(c)** from the paper:


```
@article{amato2017deep, title={Deep learning for decentralized parking lot occupancy detection}, author={Amato, Giuseppe and Carrara, Fabio and Falchi, Fabrizio and Gennaro, Claudio and Meghini, Carlo and Vairo, Claudio}, journal={Expert Systems with Applications}, volume={72}, pages={327--334}, year={2017}, publisher={Pergamon} }
```


## Dataset & Labels Download
You can download the full **CNRPark+EXT** dataset and its corresponding labels from the following links:
- **CNRPark+EXT.csv** (18.1 MB) - Labels file  
  [Download](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNRPark+EXT.csv) and place it into labels folder
- **CNR-EXT_FULL_IMAGE_1000x750.tar** (1.1 GB) - Dataset images  
  [Download](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT_FULL_IMAGE_1000x750.tar)

### Demo Images

This [folder](../demo_images/) contains sample images extracted from the **CNRPark+EXT** dataset. These images are used to demonstrate the labeling format and the occupancy detection process.
