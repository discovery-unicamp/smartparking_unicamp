import ultralytics
print(ultralytics.__version__)

import warnings
warnings.filterwarnings("ignore")

import os
from ultralytics import YOLO


class CFG:    
    BASE_MODEL = 'yolo11m' # yolo11m, yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov9c, yolov9e
    BASE_MODEL_WEIGHTS = f'{BASE_MODEL}.pt'



### Load pre-trained YOLO model
model = YOLO(CFG.BASE_MODEL_WEIGHTS)


# Export the model
model.export(
    format = 'tflite', # openvino, onnx, engine, tflite
    imgsz = (640, 640),
    half = False,
    int8 = False,
    simplify = False,
    nms = True,
)
