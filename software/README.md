# Smart Parking System with Deep Learning at Unicamp

The files here correspond to performing benchmarks with different deep learning models and measure inference time, as long as machine learning metrics for the models. First choose one of the following models options:
- [YOLOv8 to YOLOv11](yolov8_to_v11/)
- [YOLOv11m (TFLite)](yolov11m_tflite/)
- [EfficientDet-D2 Lite](efficientdet_lite/)
- [YOLOv3](yolov3/)
- [Mask R-CNN](maskrcnn/)

Then, to evaluate model performance, use the benchmarking script in [software/benchmarks](benchmarks/).

We also provide code for:
- [Setting a Telegram Bot](influx/bot/)