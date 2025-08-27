# 🔬 R-NET: Light-Sheet Fluorescence Microscopy (LSFM) Segmentation & 3D Reconstruction

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)  
[![Deep Learning](https://img.shields.io/badge/Successive%20Subspace%20Learning-PixelHop-orange)]()  
[![3D Visualization](https://img.shields.io/badge/3D%20&%20VR-Enabled-green)]()  

---

## 📌 Overview
**R-NET** is an open-source framework for **automated segmentation** of light-sheet fluorescence microscopy (LSFM) images and subsequent **3D volumetric reconstruction**. The algorithm leverages **PixelHop-based Successive Subspace Learning (SSL)** for efficient segmentation with minimal training data.

Developed by **Vinay Kadam** under the guidance of **Dr. Yichen Ding** at the Ding Incubator, University of Texas at Dallas, the framework aims to bridge the gap between raw microscopy data and **interactive 3D/VR analysis**.

**Objective:** Develop an automated segmentation pipeline for cardiac trabeculae in LSFM images using PixelHop with minimal training images.  
**Outcome:** Achieved **92% IoU** and **94% F1 score** using only 19 training images — outperforming U-Net. Delivered **7× faster GPU processing** and an integrated GUI tool for **3D model visualization and VR-based analysis**【42†Thesis】.

---

## 🧪 Pipeline Overview
![Pipeline](https://drive.google.com/uc?export=view&id=1lAhobxT2CabMv4FB9Qh0ely2RsGGiQU1)

## ⚙️ Algorithm Overview
![Algorithm](https://drive.google.com/uc?export=view&id=1cB7Z-bznt2hNAsGFeWX0ryfDAy4IYOEk)

---

## 📊 Results & Analysis
### Qualitative Results
![Results](https://drive.google.com/uc?export=view&id=1OKMQUmXL5gL5sAfkqpxfXgLC_tPivW1A)

### Quantitative Results
![Results](https://drive.google.com/uc?export=view&id=1mEV8AZ4I_XncQtRDP_zvdtrvdKEWRsz6)

### VR-Enabled 3D Models
![Results](https://drive.google.com/uc?export=view&id=1khC5tkY3OveJUbBQgd269FdONCY5c7Ao)

**Key Insight:** R-NET demonstrates superior segmentation performance with minimal data compared to U-Net, enabling scalable and interactive biomedical analysis【42†Thesis】.

---

## 🛠️ Installation & Dependencies
### Python Setup
Ensure **Python 3.11+** is installed and added to your [PATH](https://www.machinelearningplus.com/python/add-python-to-path-how-to-add-python-to-the-path-environment-variable-in-windows/).

### Required Libraries
Install dependencies via pip:
```bash
pip install os multiprocessing glob random subprocess PyQt5 platform sys numpy opencv-python matplotlib shutil time webbrowser re IPython nbformat json
pip install ipynb-py-convert  # must be installed manually
```

---

## 🏋️ Training Module
### Train Tab Interface
![trainTab](https://drive.google.com/uc?export=view&id=119LspquAOys2FMjHfuuUWai9p89NBZt1)

#### Preparing Training Data
- **Format:** PNG, grayscale only.
- **Naming Convention:**  
  - `image_raw_##.png`  
  - `image_seg_##.png`
- Use **Fiji** for image conversion if required.
- Files must be in the selected folder with case-sensitive names.

#### Configurable Fields
- **Number of Classes:** Binary / Multi-class (default = 2).
- **Variance (Energy %):** Recommended 0.96 – 0.98.
- **Training Images:** More improves accuracy, but beware overfitting.
- **Multiprocessing Option:** Enable for GPU acceleration.
- **Output Folder:** Saves masks + trained models.

---

## 🧪 Testing Module
### Test Tab Interface
![testTab](https://drive.google.com/uc?export=view&id=1yG60d0BITpk5J7Si5ednVVWVJt3Dg0PL)

#### Required Inputs
- **Test Images:** PNG, grayscale.
- **Model Files:** `pixelhop1.pkl`, `pixelhop2.pkl` (from training).
- **Extra Files:** Generated during training.
- **Save Path:** Directory for predicted masks.

---

## 📤 Export Module
### Export Tab Interface
![exportTab](https://drive.google.com/uc?export=view&id=1wIrNXDZlnRxarn0Em2Oo3eoXCRI7DJoR)

#### Prerequisites
- Install & run **Docker Desktop**.
- Remove `Slicerdocker.dll` from directory (avoids runtime issues).

#### Configurable Fields
- **Segmented Images:** Folder of `.tif` files.
- **Spacing/Dimensions (Z, Y, X):** Match original imaging specs.
- **VTI Method (Recommended):** Single scalable `.obj` model.
- **Non-VTI Method:** Creates `.obj` + `.mtl` per slice.

---

## 📜 License
R-NET is released under an **open-source license** (add license type, e.g., MIT/GPL).

---

✨ If you find this project useful, please ⭐ the repo and share it with the community!
