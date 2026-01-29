#### 1. `labeling.ipynb` 
**Labeling Tool (Bounding Boxes)**
Used for counting the amount of cars.



#### 2. `spot_wise_labeling.ipynb` 
**Labeling Tool (Bounding Boxes)**
Used for selecting the center of each spot. Unlike simple counters, this tool generates bounding box coordinates.

* **Function:** Interactive widget to draw bounding boxes around cars.
* **Key Features:**
    * **Bounding Box Mode:** Click Top-Left → Click Bottom-Right to draw a box.
    * **Mask Visualization:** Automatically loads spots masks  and overlays them in **red**.
    * **Safety Checks:** Warns the user if a box is drawn inside a masked (ignored) area.
    * **Output:** CSV containing `image_name`, `real_cars`, and `yolo_boxes` (List of normalized $x, y, w, h$ coordinates).