# R-NET: Light-Sheet Fluorescence Microscopy Image Segmentation & 3D Reconstruction
## Overview

R-NET is an open-source framework for segmenting light-sheet fluorescence microscopy (LSFM) images and reconstructing them into 3D volumetric representations.

The model leverages PixelHop-based successive subspace learning (SSL) for efficient, accurate segmentation.
R-NET was developed by Vinay Kadam under the guidance of Dr. Yichen Ding at the Ding Incubator, UT Dallas.

# Visual Results
## Qualitative Analysis
![Results](https://drive.google.com/uc?export=view&id=1OKMQUmXL5gL5sAfkqpxfXgLC_tPivW1A)

## VR-Based 3D Analysis
![Results](https://drive.google.com/uc?export=view&id=1khC5tkY3OveJUbBQgd269FdONCY5c7Ao)

# Dependencies + Install
## Python
Firstly, make sure that you have python installed on your computer. We recommend installing python 3.11.3 and adding it to your [path](https://www.machinelearningplus.com/python/add-python-to-path-how-to-add-python-to-the-path-environment-variable-in-windows/) and following the basic setup expectations. This will make sure all the libraries run accurately and are compatible. 

### Associated Libraries 
Most libraries will come pre-installed within the exe file and should run properly. If not, make sure to have the following libraries. You can use the pip install command to install these libraries. You can follow this [tutorial](https://packaging.python.org/en/latest/tutorials/installing-packages/) if you are unsure about “pip install”. 
Here are needed libraries: **os, multiprocessing, glob, random, subprocess, PyQt5, platform, sys, numpy, opencv-python, matplotlib, shutil, time, webbrowser, re, IPython, nbformat, json, ipynb-py-convert (INSTALL THIS BY HAND)**

Make sure, you install these variables to the python interpreter or within the directory of the folder. You can type “cmd” into the address bar of the directory to do this. 

# Training Module
## Train Tab
![trainTab](https://drive.google.com/uc?export=view&id=119LspquAOys2FMjHfuuUWai9p89NBZt1)

## Preparing Training Data

● **Format**: PNG, grayscale only (not RGB).

● **Naming convention**:
  ○ image_raw_##.png
  ○ image_seg_##.png

● Use Fiji to convert images if needed.

● Ensure files are inside the selected folder (case-sensitive names).

## Preparing Testing Data

● Same format, naming, and grayscale requirement as training images.

## Configurable Fields

● **Number of Classes** → Supports binary and Multi-class segmentation → 2.

● **Variance (Energy %)** → Recommended: 0.96 – 0.98.

● **Number of Training Images** → More images = better accuracy (beware of overfitting).

● **Multiprocessing Option** → Enable for GPU/multi-image runs.

● **Output Folder** → Directory for masks + trained models.


# Testing Module
## Test Tab
![testTab](https://drive.google.com/uc?export=view&id=1yG60d0BITpk5J7Si5ednVVWVJt3Dg0PL)

## Required Inputs

● **Test Images** → Raw input folder (PNG, grayscale).

● **Main Model Files** → pixelhop1.pkl & pixelhop2.pkl (from training).

● **Extra Model Files** → Located in the output folder from training.

● **Predicted Images Save Path** → Directory where generated masks will be saved.

# Export Module
## Export Tab
![exportTab](https://drive.google.com/uc?export=view&id=1wIrNXDZlnRxarn0Em2Oo3eoXCRI7DJoR)

## Prerequisites

● Install & log in to **Docker Desktop**.

● Ensure Docker is running in the background.

● Remove Slicerdocker.dll from the current directory (avoids runtime issues).

## Configurable Fields

● **Segmented Images** → Folder of segmented .tif files.

● **Spacing / Dimensions (Z, Y, X)** → Match as closely as possible to original image dimensions.

● **VTI Method (Recommended for large datasets)** → Produces a single .obj model, scalable via spacing.

● **Non-VTI Method** → Creates one .obj & .mtl per slice (e.g., 95 images → 95 models).






