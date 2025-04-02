# Smart Parking System with Deep Learning at Unicamp

This repository contains the complete implementation of a real-time smart parking monitoring system using edge computing and deep learning, representing a decade of research iterations at Unicamp.

### Architecture:
![System Overview](assets/docs/system_architecture.png)


### 📜 Project Timeline & Key Events
![System Overview](assets/docs/system_evolution.png)


The system has evolved through multiple research phases, leading to significant improvements in deep learning-based parking detection.

#### 🚀 2025 (Second Deployment)  
- The **YOLOv11m** model, optimized with TensorFlow Lite, was selected for deployment.  
- Achieved the best balance between **accuracy** and **inference speed** for real-time edge computing. 
- Results and a comparison between the other research phases and deployment were made and submitted to the Urban Computing Workshop (IX Workshop de Computação Urbana - CoUrb 2025) in the paper **10 Years of Deep Learning for Vehicle Detection at a Smart Parking : What has Changed?**

#### 🔍 2024 (Research Phase 3)  
- Conducted a **benchmark study** of the latest **YOLO models (YOLOv8 to YOLOv11)**.  
- Tested across multiple devices to assess performance and efficiency.  
- Results were published at the arxiv paper submitted to Elsevier Internet of Things **[Smart Parking with Pixel-Wise ROI Selection for Vehicle Detection Using YOLOv8, YOLOv9, YOLOv10, and YOLOv11](https://arxiv.org/abs/2412.01983)**.

#### 🔍 2020-2024 (Research Phase 2)  
- Evaluated **YOLO models (YOLOv3)** and **two-stage detectors (Mask R-CNN)**.  
- Focused on improving accuracy and inference speed for real-time detection. 
- Results of **YOLOv3** were published at the technical report **[SmartParking A smart solution using Deep Learning](https://smartcampus.prefeitura.unicamp.br/pub/artigos_relatorios/PFG_Joao_Victor_Estacionamento_Inteligente.pdf)**.

#### 🚀 2019 (First Deployment)  
- The system was first deployed using the **SSD-based EfficientDet d2** model optimized with TensorFlow Lite.  
- The project appeared in the media: **[Inova Campinas 2019](https://youtu.be/_cFjeLJ9SOI?t=105)**.


#### 🔍 2015-2019 (Research Phase 1)  
- Initial research focused on CNN-based object detection for parking space identification.  
- Experimented with **GoogleLeNet** and **Xception** for feature extraction.  
- A presentation on the project was delivered at **[PAPIs.io LATAM 2018](https://www.youtube.com/watch?v=vRXgc0Bvbx8)**.  
- Showcased early findings on smart parking and deep learning applications.  


### Real Time Demonstration Video:
![System Overview](assets/docs/demo_system.gif)

[Full video link](https://youtu.be/7rofjEfX5fA)

---

## 🛠 Hardware Setup
For complete hardware instructions go to 
[📖 Hardware Documentation](hardware/)

Key components:
- [Parking totem assembly](hardware/totem/)
- [Raspberry Pi 3B+ configuration](hardware/pi_and_camera/)

---


## 💻 Software Implementation
For detailed software documentation go to 
[📖 Software Documentation](software/)

Key components:
- [A benchmark of different deep learning models for accuracy and inference time](software/)
- [Instructions to set and monitor InfluxDB](software/influx/)

---

