# Compute Metrics Script

This script computes evaluation metrics for model predictions based on labeled data and inference CSV files.
Before running it, ensure to have executed one of the codes in each of the folders corresponding to the desired model:

- [YOLOv8 to YOLOv11](../yolov8_to_v11/)
- [YOLOv11m (TFLite)](../yolov11m_tflite/)
- [EfficientDet-D2 Lite](../efficientdet_lite/)
- [YOLOv3](../yolov3/)
- [Mask R-CNN](../maskrcnn/)

## Features
- Loads labeled data and model predictions
- Supports multiple datasets (`cnrpark`, `ic2`).
- Processes data to calculate key evaluation metrics
- Computes confusion matrix and performance metrics such as accuracy, recall, precision, F1-score, sensitivity, specificity, and balanced accuracy
- Computes average inference time and standard deviation
- Saves evaluation results as CSV and generates a confusion matrix plot
- Supports evaluation for all images and separately for images containing at least one vehicle

## Requirements
- **Required Python packages**: `pandas`, `seaborn`, `matplotlib`, `numpy`, `argparse`
- It is possible to use the same environment from [instructions of yolov8 to yolov11](../yolov8_to_v11/)

### Required Files
Ensure the following files and directories exist before running the script:
- **Labeled data CSV**: A file named `labels_DATASETNAME.csv` inside the `base_dir_labeled_data` directory. This file should contain ground truth labels, including the number of vehicles per image.
- **Inference CSV files**: The script looks for any `df_individual_metrics*.csv` files recursively in `base_dir_inference_csvs`. Ensure these files contain the predicted number of vehicles per image.
- **Output directory**: A valid directory path where results will be saved.

## Usage

Run the script with the following arguments:

```bash
python compute_metrics.py --model MODEL_NAME \
    --base_dir_labeled_data PATH_TO_LABELED_DATA \
    --base_dir_inference_csvs PATH_TO_INFERENCE_CSVS \
    --base_dir_results PATH_TO_SAVE_RESULTS
    --dataset <dataset_name>
```

### Arguments
- `--model` : Name of the model
- `--base_dir_labeled_data` : Path to the directory containing labeled data CSV file (`combined_metrics.csv`)
- `--base_dir_inference_csvs` : Path to the directory containing inference CSV files
- `--base_dir_results` : Path to the directory where results will be saved

## Example
YOLO from ultralytics
```bash
python compute_metrics.py --model yolov8n     --base_dir_labeled_data ../../assets/labels     --base_dir_inference_csvs ../../assets/results/results_yolo_ultralytics/yolov8n     --base_dir_results ../../assets/results/results_yolo_ultralytics/yolov8n --dataset cnrpark
```

YOLO (TFLite)
```bash
python compute_metrics.py --model yolov11m_tflite     --base_dir_labeled_data ../../assets/labels     --base_dir_inference_csvs ../../assets/results/results_yolo_tflite/yolov11m_tflite     --base_dir_results ../../assets/results/results_yolo_tflite/yolov11m_tflite --dataset cnrpark
```

Efficientdet (TFLite)
```bash
python compute_metrics.py --model efficientdetd2lite     --base_dir_labeled_data ../../assets/labels     --base_dir_inference_csvs ../../assets/results/results_efficientdet_tflite/efficientdetd2lite     --base_dir_results ../../assets/results/results_efficientdet_tflite/efficientdetd2lite --dataset cnrpark
```

YOLOV3
```bash
python compute_metrics.py --model yolov3   --base_dir_labeled_data ../../assets/labels     --base_dir_inference_csvs ../../assets/results/results_yolov3/yolov3     --base_dir_results ../../assets/results/results_yolov3/yolov3 --dataset cnrpark
```

Mask R-CNN
```bash
python compute_metrics.py --model maskrcnn_tf2 --base_dir_labeled_data ../../assets/labels --base_dir_inference_csvs ../../assets/results/results_maskrcnn_tf2/markrcnn_tf2/ --base_dir_results ../../assets/results/results_maskrcnn_tf2/markrcnn_tf2/ --dataset cnrpark
```

## Outputs
- `combined_metrics_all_MODEL_NAME.csv` : CSV file with computed evaluation metrics for all images
- `combined_metrics_cars_only_MODEL_NAME.csv` : CSV file with computed evaluation metrics for images containing at least one vehicle
- `confusion_matrix_all_MODEL_NAME.png` : Confusion matrix visualization for all images
- `confusion_matrix_cars_only_MODEL_NAME.png` : Confusion matrix visualization for images containing at least one vehicle

### Important Notes
- If you have multiple batches, ensure that the images were not processed more than once, as the script looks for any files with `df_individual_metrics.csv` recursively in the inference directory.
- The confusion matrices and metrics are computed for both all images and separately for images that contain at least one vehicle.
- For more information about labels follow the [labels folder](../../assets/labels/)
