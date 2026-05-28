# 🔪 Slicing & Patching Inference (SAHI) for Smart Parking Adaptation

This folder contains the **adaptation code** that integrates
**slicing / patching inference** into three different object detectors:
**DEIMv2**, **Florence-2**, and **YOLO**, for the Smart Parking use case at Unicamp.

The goal of these experiments is to detect **small / distant vehicles** in
high-resolution parking-lot frames (1920×1080), where running the detector over
the full image at once tends to miss small objects. To address this we use
**[SAHI – Slicing Aided Hyper Inference](https://github.com/obss/sahi)**: the
image is split into overlapping tiles ("slices"/"patches"), each tile is run
through the detector independently, and the per-tile detections are merged back
into the full-frame coordinate space.

---

## ⚠️ About reproducibility

> This is **adaptation / glue code**, *not* a package.

The notebooks here show **how** we wrapped each detector in SAHI and how we
evaluated the results. Running them as-is will **not** reproduce our exact
numbers, because the full pipeline also depends on assets that are **not**
shipped in this repository:

1. **Image data** : the parking-lot frames (1920×1080) used as input.
2. **Annotated data** : the ground-truth labels (e.g. `labels.csv`) used by the
   metrics scripts.
3. **Base / upstream code & weights** : the original DEIMv2 and Florence-2 model
   code and the trained checkpoints. The notebooks reference paths such as
   `../../configs/deimv2/deimv2_dinov3_x_coco.yml` and model weights that come
   from those upstream projects (see [Base code & weights](#-base-code--weights)).

We publish this code so the **method is transparent and reusable**, even though
the private dataset cannot be redistributed here.

---

## 📂 Contents

| Path | Description |
| --- | --- |
| [`deimv2/`](deimv2/) | SAHI-wrapped inference demo for **DEIMv2** (`DEMO_inference_DEIMv2_SAHI_045_DEFAULT.ipynb`). |
| [`florence2/`](florence2/) | SAHI-wrapped inference demo for **Florence-2** (`DEMO_inference_Florence_DEFAULT_SAHI.ipynb`). |
| [`yolo/`](yolo/) | Slicing/patching inference demo for **YOLO** (`DEMO_inference_YOLO.ipynb`). |
| [`compute_metrics.py`](compute_metrics.py) | Computes evaluation metrics (we just used MAE) by comparing predictions against the labeled data. |
| [`utils_yolo.py`](utils_yolo.py) | Helper functions shared by the YOLO pipeline (image listing, drawing, timing, resource monitoring). |

Each model folder ships its own `requirements.txt` (pinned versions, including
`sahi==0.11.23`) and the license of the corresponding upstream project.

---

## 🌐 Base code & weights

These adaptations build on top of **external** projects. To run the full
pipeline you must obtain the original code and weights from the sources below.
We pin the **exact commit / revision** we used so the behaviour is reproducible:

### DEIMv2
- **Source code:** <https://github.com/Intellindust-AI-Lab/DEIMv2/tree/8241e8d79fd1225694445aec347fa39c708cc7e8>
- Provides the model definition and the config files referenced by the notebook
  (e.g. `configs/deimv2/deimv2_dinov3_x_coco.yml`). Clone this repo and place our
  `deimv2/` notebook so that the relative `../../configs/...` paths resolve, or
  adjust the paths to match your layout.
- Licensed under the **Apache License** (see [`deimv2/LICENCE`](deimv2/LICENCE)).

### Florence-2
- **Model & weights:** <https://huggingface.co/microsoft/Florence-2-base-ft/tree/f6c1a25888ffc1d945ee8a1a77ac833c7303d46e>
- The `florence2/` notebook loads this checkpoint via the Hugging Face
  `transformers` API and runs it under SAHI.

### YOLO (Ultralytics)
- Installed directly from PyPI (`ultralytics`, pinned in
  [`yolo/requirements.txt`](yolo/requirements.txt)).
- Ultralytics is distributed under **AGPL-3.0** (see
  [`yolo/LICENSE.afgplv3`](yolo/LICENSE.afgplv3)).

### SAHI
- The slicing engine: [`obss/sahi`](https://github.com/obss/sahi), `sahi==0.11.23`
  (MIT, see [`LICENSE`](LICENSE)).

---

## ▶️ How to run (general flow)

Each model folder follows the same pattern. Using `deimv2/` as an example:

```bash
# 1. Create and activate a virtual environment
python -m venv deimv2env
source deimv2env/bin/activate        # macOS/Linux

# 2. Install the pinned dependencies
pip install -r deimv2/requirements.txt

# 3. Get the upstream base code & weights (see "Base code & weights" above)

# 4. Provide your input images + annotations, then open the demo notebook
jupyter notebook deimv2/DEMO_inference_DEIMv2_SAHI_045_DEFAULT.ipynb
```

> Each notebook was tested with **Python 3.11.7** and **pip 23.2.1**.

The notebooks produce a `results.csv` with per-image predictions; you can then
score them against your labels with [`compute_metrics.py`](compute_metrics.py).

---

## 🔁 Adapting this to your own use case

This code is intentionally generic and can be reused well beyond smart parking.
The core idea is: **wrap any detector in a `sahi.DetectionModel` subclass** so
SAHI can call it per slice. In the DEIMv2 notebook this is the
`DEIMv2DetectionModel(DetectionModel)` class: use it as a template to plug in
**your own model**.

To adapt it:
- **Swap the detector** : implement a `DetectionModel` subclass that loads your
  weights and converts your model's outputs into SAHI's prediction format.
- **Tune the slicing** : change the slice size, overlap ratio and the merge
  (postprocess) thresholds to match your image resolution and object scale.
- **Change the target classes** : the demos filter for vehicle classes (cars,
  trucks); replace these with the classes relevant to your domain.
- **Reuse the metrics** : `compute_metrics.py` is dataset-agnostic; point it at
  your own predictions CSV and label CSV.


---

## 📚 Related

- Main project documentation: [`software/`](../)
- Standardized metrics across models: [`software/benchmarks/`](../benchmarks/)
- YOLOv8–YOLOv11 baselines: [`software/yolov8_to_v11/`](../yolov8_to_v11/)

---

## 📝 License

The bundled `LICENSE` (MIT) and the per-model license files cover the
respective upstream projects (SAHI, DEIMv2 / Apache, Ultralytics / AGPL-3.0,
Florence-2). When you build on this work, **comply with the
license of each upstream component** you use.
